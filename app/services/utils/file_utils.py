import os
import shutil

TEMP_DIR = "temp_files"


def ensure_temp_dirs():
    """
    Ensure required temp directories exist.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(os.path.join(TEMP_DIR, "frames"), exist_ok=True)


def cleanup_temp_files():
    """
    Delete all temporary files and folders inside temp_files/.
    Called after pipeline execution.
    """
    if not os.path.exists(TEMP_DIR):
        return

    try:
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)

            if os.path.isfile(item_path):
                os.remove(item_path)

            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    except Exception as e:
        print(f"[WARN] Temp cleanup failed: {e}")
