# ------------------------------
# Video / OCR
# ------------------------------
FRAME_EXTRACTION_RATE = 1  # frames per second

# ------------------------------
# Paths
# ------------------------------
TEMP_DIR = "temp_files"
FRAMES_DIR = "temp_files/frames"

# ------------------------------
# NLP thresholds
# ------------------------------
EMOTION_CONFIDENCE_THRESHOLD = 0.75
TOXICITY_THRESHOLD = 0.80
POLITICAL_BIAS_THRESHOLD = 0.75
SUBJECTIVITY_THRESHOLD = 0.75

# ------------------------------
# Scoring weights
# ------------------------------
MISINFO_PENALTY = 20
UNCERTAIN_PENALTY = 5
BIAS_MAX_SCORE = 100
