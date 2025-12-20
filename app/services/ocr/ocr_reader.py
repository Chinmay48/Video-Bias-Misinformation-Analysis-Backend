import easyocr
import os

# Load EasyOCR reader once (global)
reader = easyocr.Reader(['en'])

def read_text_from_frames(frame_paths):
    """
    Run OCR on all extracted frames.
    Returns combined OCR text.
    """

    ocr_results = []

    for frame in frame_paths:
        try:
            text = reader.readtext(frame, detail=0)  # detail=0 returns plain text
            if text:
                # Merge text from this frame
                combined = " ".join(text)
                ocr_results.append(combined)
        except Exception as e:
            print(f"OCR failed on frame {frame}: {e}")

    # Merge all OCR text from all frames
    ocr_final = "\n".join(ocr_results)

    return ocr_final
