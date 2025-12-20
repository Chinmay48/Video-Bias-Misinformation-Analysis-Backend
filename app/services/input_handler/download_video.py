import yt_dlp
import os

from app.services.utils.constants import TEMP_DIR


async def download_video(input_info):
    """
    Downloads video using yt-dlp.
    Works for YouTube, short links, playlists, etc.
    """

    os.makedirs(TEMP_DIR, exist_ok=True)

    output_path = os.path.join(TEMP_DIR, "video.mp4")

    ydl_opts = {
        "outtmpl": output_path,
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "noplaylist": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([input_info["url"]])

    return output_path
