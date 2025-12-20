from app.services.input_handler.detect_input_type import detect_input_type
from app.services.input_handler.download_video import download_video
from app.services.input_handler.extract_audio import extract_audio

from app.services.transcript.youtube_transcript import get_youtube_transcript
from app.services.transcript.whisper_transcript import generate_whisper_transcript

from app.services.ocr.frame_extractor import extract_frames
from app.services.ocr.ocr_reader import read_text_from_frames

from app.services.nlp.merge_text import merge_text
from app.services.nlp.text_processing import preprocess_text
from app.services.nlp.bias_detection import analyze_bias
from app.services.nlp.misinformation_detection import detect_misinformation  # NEW

from app.services.utils.file_utils import cleanup_temp_files


async def run_full_pipeline(video_url=None, file=None):

    # ------------------------------------
    # 1. Detect input type
    # ------------------------------------
    input_info = await detect_input_type(video_url, file)

    transcript_text = ""
    video_path = None

    # ------------------------------------
    # 2. YouTube link with auto captions
    # ------------------------------------
    if input_info["type"] == "youtube_with_transcript":
        transcript_text = get_youtube_transcript(input_info["video_id"])

        # Still need the video file for OCR
        video_path = await download_video(input_info)

    else:
        # ------------------------------------
        # 3. Download video directly
        # ------------------------------------
        video_path = await download_video(input_info)

        # ------------------------------------
        # 4. Extract audio from video
        # ------------------------------------
        audio_path = await extract_audio(video_path)

        # ------------------------------------
        # 5. Speech-to-text using Whisper
        # ------------------------------------
        transcript_text = generate_whisper_transcript(audio_path)

    # ------------------------------------
    # 6. OCR — Extract frames + read text
    # ------------------------------------
    frame_paths = extract_frames(video_path)
    ocr_text = read_text_from_frames(frame_paths)

    # ------------------------------------
    # 7. Merge transcript + OCR text
    # ------------------------------------
    merged_text = merge_text(transcript_text, ocr_text)

    # ------------------------------------
    # 8. Preprocess text (clean + tokenize)
    # ------------------------------------
    clean_text, sentences = preprocess_text(merged_text)

    # ------------------------------------
    # 9. Bias Detection
    # ------------------------------------
    bias_report = analyze_bias(sentences)

    # ------------------------------------
    # 10. Misinformation Detection (NEW)
    # ------------------------------------
    misinfo_report = detect_misinformation(clean_text, sentences)

    # ------------------------------------
    # 11. Cleanup temporary files
    # ------------------------------------
    cleanup_temp_files()

    # ------------------------------------
    # 12. Final combined response
    # ------------------------------------
    return {
        "transcript": transcript_text,
        "ocr_text": ocr_text,
        "clean_text": clean_text,

        "bias_report": bias_report,

        "misinformation": misinfo_report["misinformation"],
        "misinformation_score": misinfo_report["misinformation_score"],
        "final_reliability_score": misinfo_report["final_reliability_score"]
    }
