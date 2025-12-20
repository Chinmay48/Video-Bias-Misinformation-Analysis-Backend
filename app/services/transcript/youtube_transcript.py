from youtube_transcript_api import YouTubeTranscriptApi

def get_youtube_transcript(video_id: str) -> str:
    """
    Fetches YouTube transcript using YouTubeTranscriptApi.
    Returns a clean text transcript.
    """

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

        # Merge all transcript chunks into single string
        transcript_text = " ".join([entry["text"] for entry in transcript_list])

        # Clean line breaks and extra spaces
        transcript_text = transcript_text.replace("\n", " ").strip()

        return transcript_text

    except Exception as e:
        raise Exception(f"Failed to fetch YouTube transcript: {str(e)}")
