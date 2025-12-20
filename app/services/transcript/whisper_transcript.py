import whisper

# Load the Whisper model once (global)
# "tiny" is fastest, "base" is more accurate but larger
model = whisper.load_model("tiny")

def generate_whisper_transcript(audio_path: str) -> str:
    """
    Convert extracted audio into text using OpenAI Whisper (tiny model).
    """

    try:
        result = model.transcribe(audio_path)
        transcript = result.get("text", "")

        # Clean transcript
        transcript = transcript.strip().replace("\n", " ")

        return transcript

    except Exception as e:
        raise Exception(f"Whisper transcription failed: {str(e)}")
