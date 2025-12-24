# test_hf.py
from  app.services.nlp.bias_detection import hf_inference

result = hf_inference(
    "facebook/bart-large-mnli",
    {
        "inputs": "This is a great movie!",
        "parameters": {
            "candidate_labels": ["positive", "negative", "neutral"]
        }
    }
)
print(result)
