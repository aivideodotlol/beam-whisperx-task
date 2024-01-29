from beam import App, Runtime, Image, Volume, Output
import base64
import whisperx
import json
import os
from tempfile import NamedTemporaryFile
import torch

device = "cuda"

app = App(
    name="whisper",
    runtime=Runtime(
        cpu=1,
        memory="8Gi",
        gpu="T4",
        image=Image(
            python_packages=["git+https://github.com/m-bain/whisperx.git"],
            commands=["apt-get update && apt-get install -y ffmpeg"],
        ),
    ),
    volumes=[Volume(path="./cache", name="cache")],
)


def load_models():
    model = whisperx.load_model(
        "medium", download_root="./cache", device=device, compute_type="float16"
    )
    return model


@app.rest_api(
    outputs=[Output(path="output.json")], loader=load_models, keep_warm_seconds=60
)
def transcribe_audio(**inputs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a temporary file.
    with NamedTemporaryFile() as temp:
        # Write the user's uploaded file to the temporary file.
        audio_file = base64.b64decode(inputs["audio_file"].encode("utf-8"))
        temp.write(audio_file)

        # Retrieve model from loader
        model = inputs["context"]

        # Inference
        audio = whisperx.load_audio(temp.name)
        result = model.transcribe(audio, batch_size=16)

        lang = result["language"]

        model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=os.getenv("HUGGINGFACE_TOKEN"), device=device
        )
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    
    # Fixing symbols not having timestamps
    segments = result["segments"]
    for i, segment in enumerate(segments):
        past_word = None
        
        for x, word in enumerate(segment["words"]):
            if "start" not in word:
                if past_word is None:
                    word["start"] = 0
                else:
                    word["start"] = past_word["end"]
                word["speaker"] = segment["speaker"]

            if "end" not in word:
                if len(segment["words"]) > x + 1:
                    word["end"] = segment["words"][x + 1]["start"]
                elif segment["end"] is not None:
                    word["end"] = segment["end"]
                elif len(result["segments"]) > i + 1:
                    word["end"] = result["segments"][i + 1]["start"]

                if word["end"] is None:
                    word["end"] = word["start"] + 0.5

            past_word = word

    result = {
        "lang": lang,
        "segments": segments,
    }
    # Write transcription to file output
    with open("output.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    """'
    *** Testing Locally ***

    > beam start app.py
    > python app.py

    """
    import os

    mp3_filepath = os.path.abspath("test.mp3")
    transcribe_audio(
        audio_file=base64.b64encode(open(mp3_filepath, "rb").read()).decode("UTF-8"),
    )
