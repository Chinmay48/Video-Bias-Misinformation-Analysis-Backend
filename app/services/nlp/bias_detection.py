import os
import requests
from collections import Counter
from typing import List, Tuple, Any

# =====================================================
# CONFIG
# =====================================================

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set in environment variables")

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models"

# =====================================================
# HF API CALL
# =====================================================

def hf_inference(model_name: str, payload: dict) -> Any:
    url = f"{HF_ROUTER_URL}/{model_name}"

    response = requests.post(
        url,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(
            f"HF API error ({model_name}): {response.status_code} {response.text}"
        )

    return response.json()


# =====================================================
# HF OUTPUT NORMALIZER (VERY IMPORTANT)
# =====================================================

def get_top_label(result):
    """
    Handles ALL HF formats:
    - list[{label, score}]
    - list[list[{label, score}]]
    - {labels:[], scores:[]}
    """

    if isinstance(result, dict):
        return result["labels"][0], float(result["scores"][0])

    if isinstance(result, list):
        while isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
            result = result[0]

        if isinstance(result, list) and isinstance(result[0], dict):
            top = max(result, key=lambda x: x.get("score", 0))
            return top.get("label", "unknown"), float(top.get("score", 0))

    raise ValueError(f"Unexpected HF response format: {result}")


# =====================================================
# RULE-BASED PREFILTER (SPEED BOOST 🚀)
# =====================================================

IMPORTANT_KEYWORDS = {
    "should", "must", "never", "always", "truth", "lie", "fake",
    "government", "media", "people", "danger", "fear", "hate",
    "agenda", "propaganda", "corrupt", "exposed"
}

def is_candidate_sentence(sentence: str) -> bool:
    words = set(sentence.split())
    if len(words) < 6:
        return False
    return not words.isdisjoint(IMPORTANT_KEYWORDS)


# =====================================================
# OPTIMIZED DETECTION FUNCTIONS
# =====================================================

def detect_emotion_and_manipulation(sentence: str) -> Tuple[str, float]:
    """
    Single model replaces:
    - sentiment
    - emotion
    - partial toxicity
    """
    result = hf_inference(
        "j-hartmann/emotion-english-distilroberta-base",
        {"inputs": sentence}
    )
    return get_top_label(result)


def detect_bias_and_subjectivity(sentence: str) -> Tuple[str, float]:
    """
    Single MNLI call replaces:
    - political bias
    - subjectivity
    """
    result = hf_inference(
        "facebook/bart-large-mnli",
        {
            "inputs": sentence,
            "parameters": {
                "candidate_labels": [
                    "left-leaning political opinion",
                    "right-leaning political opinion",
                    "neutral factual statement",
                    "subjective opinion"
                ]
            }
        }
    )
    return get_top_label(result)


# =====================================================
# MAIN ANALYSIS FUNCTION (OPTIMIZED ⚡)
# =====================================================

def analyze_bias(sentences: List[str]) -> dict:

    emotional_flags = 0
    manipulative_sentences = []
    political_biases = []
    opinion_sentences = []

    for sentence in sentences:

        # ---------- FAST RULE FILTER ----------
        if not is_candidate_sentence(sentence):
            continue

        try:
            # ---------- EMOTION + MANIPULATION ----------
            emotion_label, emotion_score = detect_emotion_and_manipulation(sentence)

            if emotion_label in {"anger", "fear", "disgust"} and emotion_score > 0.75:
                manipulative_sentences.append(sentence)

            if emotion_score > 0.85:
                emotional_flags += 1

            # ---------- BIAS + SUBJECTIVITY ----------
            bias_label, bias_score = detect_bias_and_subjectivity(sentence)

            if "left-leaning" in bias_label or "right-leaning" in bias_label:
                if bias_score > 0.75:
                    political_biases.append(bias_label)

            if "subjective" in bias_label and bias_score > 0.75:
                opinion_sentences.append(sentence)

        except Exception as e:
            print(f"[WARN] Bias skipped: {e}")

    # =================================================
    # FINAL AGGREGATION
    # =================================================

    political_bias = most_common(political_biases)

    bias_score = calculate_bias_score(
        emotional_flags=emotional_flags,
        manipulative=len(manipulative_sentences),
        political=len(political_biases),
        opinions=len(opinion_sentences)
    )

    return {
        "emotional_tone": "emotional" if emotional_flags > 0 else "neutral",
        "manipulative_language": len(manipulative_sentences) > 0,
        "political_bias": political_bias or "neutral",
        "opinion_disguised_as_fact": opinion_sentences[:5],
        "bias_score": bias_score
    }


# =====================================================
# UTILITIES
# =====================================================

def most_common(items: List[str]):
    if not items:
        return None
    return Counter(items).most_common(1)[0][0]


def calculate_bias_score(
    emotional_flags: int,
    manipulative: int,
    political: int,
    opinions: int
) -> int:
    score = 0
    score += emotional_flags * 5
    score += manipulative * 10
    score += political * 10
    score += opinions * 5
    return min(score, 100)
