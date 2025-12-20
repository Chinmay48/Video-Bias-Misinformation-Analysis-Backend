from transformers import pipeline

# =====================================================
# Lazy-loaded HuggingFace models (SAFE for uvicorn reload)
# =====================================================

_sentiment_model = None
_emotion_model = None
_toxicity_model = None
_political_model = None
_subjectivity_model = None


def get_sentiment_model():
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = pipeline("sentiment-analysis")
    return _sentiment_model


def get_emotion_model():
    global _emotion_model
    if _emotion_model is None:
        _emotion_model = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            top_k=1
        )
    return _emotion_model


def get_toxicity_model():
    global _toxicity_model
    if _toxicity_model is None:
        _toxicity_model = pipeline(
            "text-classification",
            model="unitary/unbiased-toxic-roberta"
        )
    return _toxicity_model


def get_political_model():
    global _political_model
    if _political_model is None:
        _political_model = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
    return _political_model


def get_subjectivity_model():
    global _subjectivity_model
    if _subjectivity_model is None:
        _subjectivity_model = pipeline(
            "text-classification",
            model="julien-c/bert-base-uncased-subjectivity"
        )
    return _subjectivity_model


# =====================================================
# Helper detection functions
# =====================================================

def detect_emotional_tone(sentence):
    result = get_sentiment_model()(sentence)[0]
    return result["label"], result["score"]


def detect_emotion(sentence):
    result = get_emotion_model()(sentence)[0][0]
    return result["label"], result["score"]


def detect_toxicity(sentence):
    result = get_toxicity_model()(sentence)[0]
    return result["label"], result["score"]


def detect_political_bias(sentence):
    result = get_political_model()(
        sentence,
        candidate_labels=["left-leaning", "right-leaning", "neutral"]
    )
    return result["labels"][0], result["scores"][0]


def detect_subjectivity(sentence):
    result = get_subjectivity_model()(sentence)[0]
    return result["label"], result["score"]


# =====================================================
# MAIN ANALYSIS FUNCTION
# =====================================================

def analyze_bias(sentences):
    """
    Input: List of cleaned sentences
    Output: Bias analysis report
    """

    emotional_levels = []
    manipulative_flags = []
    political_bias_results = []
    opinion_sentences = []

    for sentence in sentences:

        # -------- Emotional Tone --------
        emo_label, emo_score = detect_emotional_tone(sentence)
        if emo_score > 0.85:
            emotional_levels.append(emo_label)

        # -------- Emotion Manipulation --------
        emotion_label, emotion_score = detect_emotion(sentence)
        if emotion_label in ["anger", "fear", "disgust"] and emotion_score > 0.75:
            manipulative_flags.append(sentence)

        # -------- Toxicity --------
        toxic_label, toxic_score = detect_toxicity(sentence)
        if toxic_label == "toxic" and toxic_score > 0.80:
            manipulative_flags.append(sentence)

        # -------- Political Bias --------
        pol_label, pol_score = detect_political_bias(sentence)
        if pol_label != "neutral" and pol_score > 0.75:
            political_bias_results.append(pol_label)

        # -------- Subjectivity --------
        subj_label, subj_score = detect_subjectivity(sentence)
        if subj_label == "SUBJECTIVE" and subj_score > 0.75:
            opinion_sentences.append(sentence)

    # =================================================
    # FINAL AGGREGATION
    # =================================================

    emotional_tone = most_common(emotional_levels)
    political_bias = most_common(political_bias_results)

    bias_score = calculate_bias_score(
        emotional_flags=len(emotional_levels),
        manipulative=len(manipulative_flags),
        political=len(political_bias_results),
        opinions=len(opinion_sentences)
    )

    return {
        "emotional_tone": emotional_tone or "neutral",
        "manipulative_language": len(manipulative_flags) > 0,
        "political_bias": political_bias or "neutral",
        "opinion_disguised_as_fact": opinion_sentences[:5],
        "bias_score": bias_score
    }


# =====================================================
# Utility functions
# =====================================================

def most_common(items):
    if not items:
        return None
    return max(set(items), key=items.count)


def calculate_bias_score(emotional_flags, manipulative, political, opinions):
    score = 0
    score += emotional_flags * 5
    score += manipulative * 10
    score += political * 10
    score += opinions * 5
    return min(score, 100)
