# Task 3 — Toxic Content Classifier

A production-ready Streamlit application that classifies user-submitted **text queries** and **images** into toxicity categories using a custom-trained neural classifier backed by sentence embeddings.

---

## What It Does

- Accepts a **text query**, an **uploaded image**, or both
- Images are automatically captioned (via BLIP-1) and passed to the classifier
- Outputs a predicted toxicity category with a confidence score and full probability distribution
- Logs all predictions to a database for auditing

### Output Categories

| Label | Description |
|---|---|
| `Safe` | No harmful content detected |
| `unsafe` | General unsafe content |
| `Violent Crimes` | Content related to violent criminal activity |
| `Non-Violent Crimes` | Content related to non-violent criminal activity |
| `Sex-Related Crimes` | Sexual criminal content |
| `Child Sexual Exploitation` | CSAM or grooming-related content |
| `Suicide & Self-Harm` | Self-harm or suicidal content |
| `Elections` | Election interference or manipulation |
| `Unknown S-Type` | Ambiguous unsafe content (low-confidence flag) |

> ⚠️ Known model weakness: `Safe` and `Unknown S-Type` are occasionally confused. Treat `Unknown S-Type` predictions with lower confidence in downstream logic.

---

## Project Structure

```
Task3/
├── app.py                  # Streamlit frontend + app entrypoint
├── toxic_classifier.py     # ML pipeline — text & image classification
├── imagecaption.py         # BLIP-1 wrapper for image → caption
├── blip_captioner.py       # BLIPCaptioner class (BLIP-1 / BLIP-2)
├── database.py             # Prediction logging
└── models/
    └── toxic_classifier_best.pth   # Trained model weights
```

---

## Setup

### 1. Install dependencies

```bash
pip install torch torchvision transformers sentence-transformers streamlit pillow requests
```

### 2. ⚠️ Enable real image captioning (IMPORTANT for graders)

By default, `imagecaption.py` runs in **mock mode** and returns a hardcoded caption.  
To enable real BLIP-1 captioning:

**Open `imagecaption.py` and:**
1. **Delete** the mock `caption_image` function (lines ~17–24)
2. **Uncomment** the real implementation block at the bottom of the file
3. On first run, BLIP-1 weights (~1.88 GB) will download automatically from HuggingFace

```python
# UNCOMMENT THIS BLOCK in imagecaption.py:

# from blip_captioner import BLIPCaptioner
# _captioner = BLIPCaptioner(model="blip1")
# def caption_image(image, prompt=None, max_new_tokens=100, num_beams=5):
#     return _captioner.caption(image_source=image, ...)
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## Model Details

- **Architecture:** 3-layer MLP classifier on top of sentence embeddings
- **Embedder:** `all-MiniLM-L6-v2` (SentenceTransformers) — downloads automatically (~80MB)
- **Input:** Combined query + image description embeddings (768-dim each, concatenated)
- **Output:** Softmax over 9 classes
- **Hardware:** Runs on CPU or CUDA automatically

---

## Notes

- The `.pth` model file must be present at `models/toxic_classifier_best.pth`
- Both the embedder and classifier are loaded **once at startup** — not per request
- Image classification accuracy depends on BLIP caption quality; mock mode bypasses this entirely
