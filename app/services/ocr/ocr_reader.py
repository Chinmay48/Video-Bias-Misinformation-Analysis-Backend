from paddleocr import PaddleOCR
import time
import hashlib

# Initialize once (angle classifier enabled here)
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def frame_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def read_text_from_frames(frame_paths, skip_every=3):

    ocr_results = []
    seen_hashes = set()

    frames = frame_paths[::skip_every]

    print(f"\n🔍 OCR STARTED (PaddleOCR)")
    print(f"📉 Frames reduced: {len(frame_paths)} → {len(frames)}\n")

    start = time.time()

    for idx, frame in enumerate(frames, start=1):
        try:
            h = frame_hash(frame)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            result = ocr.ocr(frame)

            if result:
                for line in result:
                    text = " ".join([word[1][0] for word in line])
                    ocr_results.append(text)

        except Exception as e:
            print(f"OCR error on frame {idx}: {e}")

        if idx % 10 == 0:
            print(f"📸 OCR progress: {idx}/{len(frames)}")

    print(f"\n✅ OCR DONE in {round(time.time() - start, 2)} sec\n")

    return "\n".join(ocr_results)
