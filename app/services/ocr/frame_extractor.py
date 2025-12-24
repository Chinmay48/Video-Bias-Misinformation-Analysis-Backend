import cv2
import os
import hashlib

def _frame_hash(frame, size=16):
    """
    Lightweight perceptual hash for duplicate detection
    """
    small = cv2.resize(frame, (size, size))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


def extract_frames(
    video_path: str,
    frame_rate: int = 3,        # 1 frame every 3 seconds (OCR-friendly)
    max_frames: int = 120,
    resize_width: int = 960     # resize for faster OCR
):
    """
    Optimized frame extraction for OCR.

    - frame_rate: seconds between frames
    - max_frames: hard safety limit
    - resize_width: downscale frames for OCR speed
    """

    frames_dir = "temp_files/frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Clear old frames
    for file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, file))

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Failed to open video for frame extraction")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    frame_step = int(fps * frame_rate)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths = []
    seen_hashes = set()
    saved = 0
    current_frame = 0

    while current_frame < total_frames and saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = cap.read()
        if not success:
            break

        # Resize (keep aspect ratio)
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / w
            frame = cv2.resize(frame, (resize_width, int(h * scale)))

        # Skip duplicate frames
        hsh = _frame_hash(frame)
        if hsh in seen_hashes:
            current_frame += frame_step
            continue

        seen_hashes.add(hsh)

        frame_file = f"{frames_dir}/frame_{saved}.jpg"
        cv2.imwrite(frame_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        frame_paths.append(frame_file)
        saved += 1
        current_frame += frame_step

    cap.release()

    print(
        f"🎞️ Frames extracted: {len(frame_paths)} "
        f"(every {frame_rate}s, max {max_frames})"
    )

    return frame_paths
