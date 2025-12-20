from pydantic import BaseModel
from typing import Optional

class VideoInput(BaseModel):
    video_url: Optional[str] = None
    language: str = "en"
