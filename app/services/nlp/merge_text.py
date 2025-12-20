def merge_text(transcript_text: str, ocr_text: str) -> str:
    """
    Merges transcript text + OCR extracted text.
    Ensures both are combined cleanly for NLP processing.
    """

    if not transcript_text:
        transcript_text = ""

    if not ocr_text:
        ocr_text = ""

    # Clean whitespace
    transcript_text = transcript_text.strip()
    ocr_text = ocr_text.strip()

    # Merge both with separation
    merged_text = transcript_text + "\n\n" + ocr_text

    return merged_text.strip()
