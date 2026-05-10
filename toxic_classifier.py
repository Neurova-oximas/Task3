"""
toxic_classifier.py  —  CONTRACT FILE
======================================
This is the STUB / CONTRACT that defines what the ML team must implement.
Replace the function bodies with your actual model code.
Do NOT change:
  - function signatures
  - return types
  - PredictionResult field names or types
"""

from dataclasses import dataclass
from typing import Dict
from PIL import Image


# ── Shared data contract ───────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Return type for both predict_text_class and predict_image_class.

    Fields
    ------
    label       : str
        The name of the top predicted class (e.g. "Class A").

    confidence  : float
        Confidence score for the top class. Must be in [0.0, 1.0].

    all_scores  : Dict[str, float]
        Confidence score for EVERY class your model outputs.
        Keys   → class name strings (consistent across calls)
        Values → floats in [0.0, 1.0]
        Should ideally sum to ~1.0 (softmax output), but not enforced.

    Example
    -------
    PredictionResult(
        label      = "Severe",
        confidence = 0.87,
        all_scores = {
            "Clean":   0.05,
            "Mild":    0.08,
            "Severe":  0.87,
        }
    )
    """
    label: str
    confidence: float
    all_scores: Dict[str, float]

example_prediction = PredictionResult("example Class",0.90,{"Toxic":0.15,"example Class":0.9,"Safe":0.05})

# ── Text pipeline ──────────────────────────────────────────────────────────────

def predict_text_class(text: str) -> PredictionResult:
    """
    Run the text classification pipeline.

    Parameters
    ----------
    text : str
        Raw input string from the user. Already stripped of leading/trailing
        whitespace by the app. Do NOT assume any further preprocessing.

    Returns
    -------
    PredictionResult
        Populated with the top label, its confidence, and all class scores.
    """
    # ── REPLACE THIS STUB WITH YOUR MODEL CODE ────────────────────────────────
    return example_prediction
    #raise NotImplementedError("predict_text_class not implemented yet.")


# ── Image pipeline ─────────────────────────────────────────────────────────────

def predict_image_class(image: Image.Image) -> PredictionResult:
    """
    Run the image classification pipeline.
    (Internally: caption the image → run text classifier, or direct vision model.)

    Parameters
    ----------
    image : PIL.Image.Image
        RGB image object. Already converted to RGB by the app.
        Size is NOT guaranteed — handle any resolution inside this function.

    Returns
    -------
    PredictionResult
        Populated with the top label, its confidence, and all class scores.
    """
    # ── REPLACE THIS STUB WITH YOUR MODEL CODE ────────────────────────────────
    return example_prediction
    #raise NotImplementedError("predict_image_class not implemented yet.")