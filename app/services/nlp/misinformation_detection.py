import re
import requests
from transformers import pipeline

# ------------------------------
# Load fact-checking model (MNLI)
# ------------------------------
# This model can classify:
#  - ENTAILMENT (supports)
#  - CONTRADICTION (refutes / misinformation)
#  - NEUTRAL (uncertain)
mnli_model = pipeline("text-classification", model="facebook/bart-large-mnli")


# ------------------------------
# STEP 1 — Simple Claim Extraction
# ------------------------------

def extract_claims(sentences):
    claims = []

    for s in sentences:
        # Very simple rules to detect factual claims
        if any(x in s for x in ["is", "are", "was", "were", "has", "have"]):
            if len(s.split()) > 5:  # avoid very small sentences
                claims.append(s)

        # Sentences with numbers are usually factual statements
        elif re.search(r"\d", s):
            claims.append(s)

    return claims



# ------------------------------
# STEP 2 — Wikipedia Search
# ------------------------------

def get_wikipedia_summary(query):
    """
    Search Wikipedia for the topic and return a summary paragraph.
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
        response = requests.get(url).json()

        if "extract" in response:
            return response["extract"]
    except:
        return None

    return None



# ------------------------------
# STEP 3 — Compare Claim vs Evidence (MNLI)
# ------------------------------

def classify_claim(claim, evidence):
    """
    Returns:
        verdict: "supported" | "misinformation" | "uncertain"
        score: confidence score
    """

    if evidence is None:
        return "uncertain", 0.0

    result = mnli_model({
        "text": evidence,
        "text_pair": claim
    })[0]

    label = result["label"]
    score = result["score"]

    if label == "ENTAILMENT":
        return "supported", score

    elif label == "CONTRADICTION":
        return "misinformation", score

    else:
        return "uncertain", score



# ------------------------------
# STEP 4 — MAIN MISINFORMATION DETECTION LOGIC
# ------------------------------

def detect_misinformation(clean_text, sentences):
    """
    Main misinformation detection function.
    Returns:
        misinformation: list of claims flagged
        misinformation_score: 0–100
        final_reliability_score: combination of bias + misinfo
    """

    # 1. Extract factual claims
    claims = extract_claims(sentences)

    misinformation_results = []
    misinformation_score = 0

    for claim in claims:

        # 2. Use first 3 keywords in the claim to search Wikipedia
        keywords = claim.split()[:3]
        wiki_query = " ".join(keywords)
        evidence = get_wikipedia_summary(wiki_query)

        # 3. Classify using MNLI
        verdict, confidence = classify_claim(claim, evidence)

        # 4. Score logic
        if verdict == "misinformation":
            misinformation_score += 20  # high penalty

        elif verdict == "uncertain":
            misinformation_score += 5  # small penalty

        # Store results
        misinformation_results.append({
            "claim": claim,
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "evidence_snippet": evidence[:200] if evidence else None
        })

    # Limit max score
    misinformation_score = min(misinformation_score, 100)

    # Combine with bias score (handled in run_pipeline)
    # For now, final_reliability_score = 100 - misinfo_score (placeholder)
    # Later, you'll update using bias + misinfo both
    final_reliability = max(0, 100 - misinformation_score)

    return {
        "misinformation": misinformation_results[:10],  # return top 10 claims
        "misinformation_score": misinformation_score,
        "final_reliability_score": final_reliability
    }
