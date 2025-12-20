import re
import os
from youtube_transcript_api import YouTubeTranscriptApi

YOUTUBE_REGEX = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+"


async def detect_input_type(video_url=None, file=None):
    """
    Detect if input is:
    1. YouTube link with transcript
    2. YouTube link without transcript
    3. Direct video URL
    4. Uploaded video file
    """

    # CASE 1: Uploaded file (highest priority)
    if file:
        os.makedirs("temp_files", exist_ok=True)

        file_path = f"temp_files/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        return {
            "type": "file_upload",
            "video_path": file_path
        }

    # CASE 2: No input
    if not video_url:
        raise Exception("No video URL or file provided.")

    # CASE 3: YouTube URL
    if re.match(YOUTUBE_REGEX, video_url):
        video_id = extract_youtube_id(video_url)

        try:
            YouTubeTranscriptApi.get_transcript(video_id)
            return {
                "type": "youtube_with_transcript",
                "video_id": video_id,
                "url": video_url   # ✅ STANDARDIZED
            }
        except:
            return {
                "type": "youtube_no_transcript",
                "video_id": video_id,
                "url": video_url   # ✅ STANDARDIZED
            }

    # CASE 4: Direct video URL
    return {
        "type": "video_url",
        "url": video_url       # ✅ STANDARDIZED
    }


def extract_youtube_id(url: str) -> str:
    """
    Extract YouTube video ID from different formats.
    """

    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"embed/([^?]+)"
    ]

    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)

    raise Exception("Unable to extract YouTube video ID.")
