from fastapi import APIRouter, UploadFile, File
from app.pipeline.run_pipeline import run_full_pipeline

router = APIRouter()

@router.post("/analyze-video")
async def analyze_video(
    video_url: str = None,
    file: UploadFile = None
):
    result = await run_full_pipeline(video_url, file)
    return result
