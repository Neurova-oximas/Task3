"""
toxic_classifier.py
--------------------
ML pipeline contract + implementation stubs.

Two public functions:
    predict_text_class(text)   -> PredictionResult
    predict_image_class(image) -> tuple[PredictionResult, str]
                                  (result, caption) — caption needed for DB logging

Wire your models into the stubs below. Do NOT change function signatures or
the PredictionResult dataclass — app.py and database.py depend on this contract.
"""

from dataclasses import dataclass
from typing import Dict
from PIL import Image

from imagecaption import caption_image


# ── Shared data contract ───────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Return type for both predict_text_class and predict_image_class.

    Fields
    ------
    label       : str
        Name of the top predicted class.

    confidence  : float
        Confidence score for the top class. Must be in [0.0, 1.0].
        Must equal all_scores[label].

    all_scores  : Dict[str, float]
        Confidence for EVERY class your model outputs.
        Keys   → class name strings (must be consistent across every call)
        Values → floats in [0.0, 1.0]
        Should sum to ~1.0 (softmax output).

    Example
    -------
    PredictionResult(
        label      = "Severe",
        confidence = 0.87,
        all_scores = {
            "Clean":  0.05,
            "Mild":   0.08,
            "Severe": 0.87,
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
        Raw input string. Already stripped of whitespace by the caller.

    Returns
    -------
    PredictionResult
    """
    # ── INSERT YOUR MODEL CODE HERE ───────────────────────────────────────────
    #
    # Example skeleton:
    #
    #   inputs = tokenizer(text, return_tensors="pt", truncation=True)
    #   with torch.no_grad():
    #       logits = model(**inputs).logits
    #   probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    #   class_names = ["Clean", "Mild", "Severe"]   # your actual class names
    #   all_scores = dict(zip(class_names, probs))
    #   top_label = max(all_scores, key=all_scores.get)
    #   return PredictionResult(
    #       label      = top_label,
    #       confidence = all_scores[top_label],
    #       all_scores = all_scores,
    #   )
    #
    #
    return example_prediction
    #raise NotImplementedError("predict_text_class: plug in your model here.")


# ── Image pipeline ─────────────────────────────────────────────────────────────

def predict_image_class(image: Image.Image) -> tuple[PredictionResult, str]:
    """
    Run the image classification pipeline.

    Internally:
        1. Caption the image via BLIP-1  (imagecaption.py)
        2. Pass the caption to predict_text_class()

    Parameters
    ----------
    image : PIL.Image.Image
        RGB image. Already converted to RGB by app.py before this is called.

    Returns
    -------
    tuple[PredictionResult, str]
        (result, caption) — app.py needs the caption separately to log it to DB.
    """
    return (example_prediction,"this is an example caption")
    # caption = caption_image(image)
    # result = predict_text_class(caption)
    # return result, caption