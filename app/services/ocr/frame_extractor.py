import cv2
import os

def extract_frames(video_path: str, frame_rate: int = 1):
    """
    Extracts frames from video.
    Default: 1 frame per second.
    Saves frames in temp_files/frames/
    """

    # Create frame directory if not exists
    frames_dir = "temp_files/frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Clear old frames
    for file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, file))

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise Exception("Failed to open video for frame extraction.")

    fps = video.get(cv2.CAP_PROP_FPS)  # actual FPS of video
    frame_gap = int(fps * frame_rate)  # how many frames to skip

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break

        # Save only 1 frame per second
        if frame_count % frame_gap == 0:
            frame_file = f"{frames_dir}/frame_{saved_count}.jpg"
            cv2.imwrite(frame_file, frame)
            frame_paths.append(frame_file)
            saved_count += 1

        frame_count += 1

    video.release()
    return frame_paths
