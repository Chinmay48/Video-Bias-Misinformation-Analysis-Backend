from pydantic import BaseModel
from typing import Dict, List

class BiasReport(BaseModel):
    emotional_tone: str
    manipulative_language: bool
    political_bias: str
    opinion_disguised_as_fact: List[str]
    bias_score: int

class AnalysisResponse(BaseModel):
    transcript: str
    ocr_text: str
    clean_text: str
    bias_report: dict
    misinformation: list
    misinformation_score: int
    final_reliability_score: int
