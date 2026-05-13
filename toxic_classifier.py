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

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from imagecaption import caption_image


# ── Model config ───────────────────────────────────────────────────────────────

MODEL_PATH  = r"C:\Users\Mega Store\Desktop\Task3\models\toxic_classifier_best.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    'Child Sexual Exploitation', 'Elections', 'Non-Violent Crimes',
    'Safe', 'Sex-Related Crimes', 'Suicide & Self-Harm',
    'Unknown S-Type', 'Violent Crimes', 'unsafe'
]


# ── Model definition ───────────────────────────────────────────────────────────

class ToxicClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=9):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64),         nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# ── Load once at startup ───────────────────────────────────────────────────────
# These are expensive to initialize — loaded here at module import time so
# Streamlit only pays the cost once across all reruns.

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

_model = ToxicClassifier().to(DEVICE)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
_model.eval()


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
    """
    label: str
    confidence: float
    all_scores: Dict[str, float]


# ── Core inference (shared by both pipelines) ──────────────────────────────────

def _classify(query: str = "", image_desc: str = "") -> PredictionResult:
    """
    Internal function — builds a combined embedding from query + image description
    and runs inference. Mirrors the integration guide's classify() exactly.
    """
    q_emb = embedder.encode("query " + query,      convert_to_tensor=True)
    d_emb = embedder.encode("image " + image_desc, convert_to_tensor=True)
    x     = torch.cat([q_emb, d_emb]).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs    = torch.softmax(_model(x), dim=1).squeeze()
        pred_idx = probs.argmax().item()

    all_scores = {CLASS_NAMES[i]: round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))}
    top_label  = CLASS_NAMES[pred_idx]

    return PredictionResult(
        label      = top_label,
        confidence = round(probs[pred_idx].item(), 4),
        all_scores = all_scores,
    )


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
    return _classify(query=text, image_desc="")


# ── Image pipeline ─────────────────────────────────────────────────────────────

def predict_image_class(image: Image.Image) -> tuple[PredictionResult, str]:
    """
    Run the image classification pipeline.

    Internally:
        1. Caption the image via BLIP-1  (imagecaption.py)
        2. Pass the caption to _classify() as image_desc

    Parameters
    ----------
    image : PIL.Image.Image
        RGB image. Already converted to RGB by app.py before this is called.

    Returns
    -------
    tuple[PredictionResult, str]
        (result, caption) — app.py needs the caption separately to log it to DB.
    """
    caption = caption_image(image)
    result  = _classify(query="", image_desc=caption)
    return result, caption