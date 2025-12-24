import re
import os
import requests
import time
from functools import lru_cache

# =====================================================
# Hugging Face API setup
# =====================================================

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HF_API_TOKEN:
    raise RuntimeError("HF_API_TOKEN not set in environment variables")

HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

HF_URL = "https://router.huggingface.co/hf-inference/models"


# =====================================================
# HF API CALL (retry + safety)
# =====================================================

def hf_inference(model_name, payload, retries=3):
    url = f"{HF_URL}/{model_name}"

    for attempt in range(retries):
        response = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()

        if response.status_code in (429, 503):
            time.sleep(3)
            continue

        raise Exception(f"HF API error ({model_name}): {response.text}")

    raise Exception(f"HF API failed after {retries} retries for {model_name}")


# =====================================================
# HF OUTPUT NORMALIZER (MNLI SAFE)
# =====================================================

def get_mnli_result(result):
    """
    Handles:
    - [{label, score}]
    - [[{label, score}]]
    """
    if isinstance(result, list):
        while isinstance(result[0], list):
            result = result[0]
        top = max(result, key=lambda x: x.get("score", 0))
        return top["label"], float(top["score"])

    raise ValueError("Unexpected MNLI response format")


# =====================================================
# STEP 1 — SMART CLAIM EXTRACTION
# =====================================================

FACTUAL_TRIGGERS = (
    " is ", " are ", " was ", " were ",
    " has ", " have ", " caused ", " leads ",
    " results ", " increases ", " decreases "
)

def extract_claims(sentences):
    """
    Faster + cleaner factual claim extraction
    """
    claims = []

    for s in sentences:
        s = s.strip()

        if len(s.split()) < 6:
            continue

        s_lower = s.lower()

        if any(t in s_lower for t in FACTUAL_TRIGGERS) or re.search(r"\d", s):
            claims.append(s)

    return claims[:20]  # HARD LIMIT (very important ⚡)


# =====================================================
# STEP 2 — Wikipedia Evidence (CACHED)
# =====================================================

@lru_cache(maxsize=128)
def get_wikipedia_summary(query):
    """
    Cached Wikipedia summary fetch
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data.get("extract")

    except Exception:
        return None

    return None


def extract_wiki_query(claim):
    """
    Better Wikipedia query extraction
    """
    tokens = claim.split()
    return " ".join(tokens[:5])


# =====================================================
# STEP 3 — CLAIM VERIFICATION (MNLI)
# =====================================================

def classify_claim(claim, evidence):
    """
    Returns verdict + confidence
    """

    if not evidence:
        return "uncertain", 0.0

    result = hf_inference(
        "facebook/bart-large-mnli",
        {
            "inputs": {
                "premise": evidence,
                "hypothesis": claim
            }
        }
    )

    label, score = get_mnli_result(result)

    if label == "ENTAILMENT":
        return "supported", score

    if label == "CONTRADICTION":
        return "misinformation", score

    return "uncertain", score


# =====================================================
# STEP 4 — MAIN PIPELINE (OPTIMIZED ⚡)
# =====================================================

def detect_misinformation(clean_text, sentences):

    claims = extract_claims(sentences)

    misinformation_results = []
    misinformation_score = 0

    for claim in claims:

        try:
            wiki_query = extract_wiki_query(claim)
            evidence = get_wikipedia_summary(wiki_query)

            verdict, confidence = classify_claim(claim, evidence)

            if verdict == "misinformation":
                misinformation_score += 20
            elif verdict == "uncertain":
                misinformation_score += 5

            misinformation_results.append({
                "claim": claim,
                "verdict": verdict,
                "confidence": round(confidence, 2),
                "evidence_snippet": evidence[:200] if evidence else None
            })

        except Exception as e:
            print(f"[WARN] Claim skipped: {e}")

    misinformation_score = min(misinformation_score, 100)
    final_reliability = max(0, 100 - misinformation_score)

    return {
        "misinformation": misinformation_results[:10],
        "misinformation_score": misinformation_score,
        "final_reliability_score": final_reliability
    }
