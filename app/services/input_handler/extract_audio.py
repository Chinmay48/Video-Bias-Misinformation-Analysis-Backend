import ffmpeg
import os

async def extract_audio(video_path):
    """
    Extracts audio from the downloaded video using FFmpeg.
    Saves audio as WAV for Whisper processing.
    """

    audio_path = "temp_files/audio.wav"

    # Remove old audio file if exists
    if os.path.exists(audio_path):
        os.remove(audio_path)

    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', ac=1, ar='16000')  # mono audio, 16k sample rate
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        raise Exception(f"Audio extraction failed: {str(e)}")

    return audio_path
