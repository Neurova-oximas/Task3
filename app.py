"""
Toxic Classifier — Streamlit App
Multimodal: accepts text or image input.
ML backend: toxic_classifier module (predict_text_class / predict_image_class)
"""

import streamlit as st
from PIL import Image
from dataclasses import dataclass
from typing import Dict

# ── ML backend import ──────────────────────────────────────────────────────────
from toxic_classifier import predict_text_class, predict_image_class


# ── Data contract ──────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Agreed contract between app.py and toxic_classifier.py

    Attributes
    ----------
    label       : str   — the top predicted class name
    confidence  : float — confidence for the top class, in [0.0, 1.0]
    all_scores  : Dict[str, float] — {class_name: confidence} for ALL classes,
                                     values in [0.0, 1.0], should sum to ~1.0
    """
    label: str
    confidence: float
    all_scores: Dict[str, float]


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Toxic Classifier",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;600&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
    background-color: #080b10 !important;
    color: #c8d6e5 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.block-container {
    max-width: 740px !important;
    padding: 2rem 1.5rem 5rem !important;
}

/* ── Scanline overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,255,200,0.012) 2px,
        rgba(0,255,200,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Header ── */
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00ffc8;
    text-shadow: 0 0 20px rgba(0,255,200,0.4), 0 0 60px rgba(0,255,200,0.15);
    margin: 0;
    line-height: 1.1;
}
.app-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    color: #3a5068;
    margin-top: 0.35rem;
    text-transform: uppercase;
}
.header-rule {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, #00ffc8 0%, #0d4a3a 60%, transparent 100%);
    margin: 1.2rem 0 2rem;
}

/* ── Section labels ── */
.section-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #00ffc8;
    opacity: 0.7;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Panel / card ── */
.panel {
    background: #0d1520;
    border: 1px solid #1a2d42;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    position: relative;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00ffc8, transparent);
    border-radius: 4px 0 0 4px;
}

/* ── Radio buttons ── */
div[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 0.75rem;
}
div[data-testid="stRadio"] label {
    background: #0d1520 !important;
    border: 1px solid #1a2d42 !important;
    border-radius: 3px !important;
    padding: 0.5rem 1.4rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5a7a96 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
div[data-testid="stRadio"] label:hover {
    border-color: #00ffc8 !important;
    color: #00ffc8 !important;
}
/* hide the actual radio circle */
div[data-testid="stRadio"] label > div:first-child {
    display: none !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label[aria-checked="true"] {
    border-color: #00ffc8 !important;
    color: #00ffc8 !important;
    background: rgba(0,255,200,0.06) !important;
    box-shadow: 0 0 12px rgba(0,255,200,0.12) !important;
}

/* ── Text area ── */
textarea {
    background: #060a0f !important;
    border: 1px solid #1a2d42 !important;
    border-radius: 3px !important;
    color: #c8d6e5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    resize: vertical !important;
}
textarea:focus {
    border-color: #00ffc8 !important;
    box-shadow: 0 0 0 1px rgba(0,255,200,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #060a0f !important;
    border: 1px dashed #1a2d42 !important;
    border-radius: 3px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00ffc8 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: transparent !important;
    border: 1px solid #00ffc8 !important;
    border-radius: 3px !important;
    color: #00ffc8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.75rem !important;
    transition: all 0.25s !important;
    margin-top: 0.5rem;
}
.stButton > button:hover:not(:disabled) {
    background: rgba(0,255,200,0.08) !important;
    box-shadow: 0 0 20px rgba(0,255,200,0.2) !important;
}
.stButton > button:disabled {
    border-color: #1a2d42 !important;
    color: #2a3f52 !important;
    cursor: not-allowed !important;
    opacity: 1 !important;
}

/* ── Result panel ── */
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #00ffc8;
    text-shadow: 0 0 16px rgba(0,255,200,0.5);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.result-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #3a5068;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.class-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}
.class-name { color: #7a9ab5; letter-spacing: 0.05em; }
.class-score { color: #00ffc8; min-width: 3.5rem; text-align: right; }

/* ── Progress bars (native override) ── */
[data-testid="stProgress"] > div > div {
    background: rgba(0,255,200,0.12) !important;
    border-radius: 2px !important;
}
[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #00ffc8, #00b8a0) !important;
    border-radius: 2px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #00ffc8 !important;
}

/* ── Image preview ── */
[data-testid="stImage"] img {
    border: 1px solid #1a2d42;
    border-radius: 3px;
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<p class="app-title">⬡ Toxic Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Multimodal classification system // v1.0</p>', unsafe_allow_html=True)
st.markdown('<hr class="header-rule">', unsafe_allow_html=True)


# ── Mode selector ─────────────────────────────────────────────────────────────

st.markdown('<span class="section-tag">// input modality</span>', unsafe_allow_html=True)

mode = st.radio(
    label="Input mode",
    options=["Text", "Image"],
    horizontal=True,
    label_visibility="collapsed",
)


# ── Input panel ───────────────────────────────────────────────────────────────

st.markdown('<div class="panel">', unsafe_allow_html=True)

input_ready = False
user_text = ""
user_image = None

if mode == "Text":
    st.markdown('<span class="section-tag">// paste or type input</span>', unsafe_allow_html=True)
    user_text = st.text_area(
        label="Text input",
        placeholder="Enter content to classify...",
        height=160,
        label_visibility="collapsed",
    )
    input_ready = len(user_text.strip()) > 0

else:  # Image mode
    st.markdown('<span class="section-tag">// upload image</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Image upload",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        user_image = Image.open(uploaded_file).convert("RGB")
        st.image(user_image, caption="Preview", use_column_width=True)
        input_ready = True

st.markdown('</div>', unsafe_allow_html=True)


# ── Predict button ────────────────────────────────────────────────────────────

predict_clicked = st.button(
    label="⬡  PREDICT TOXIC CLASS",
    disabled=not input_ready,
    use_container_width=True,
)


# ── Prediction logic ──────────────────────────────────────────────────────────

if predict_clicked and input_ready:
    with st.spinner("RUNNING INFERENCE..."):
        try:
            if mode == "Text":
                result: PredictionResult = predict_text_class(user_text.strip())
            else:
                result: PredictionResult = predict_image_class(user_image)

        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    # ── Results ───────────────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown('<span class="section-tag">// classification result</span>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # Top prediction
    st.markdown(
        f'<p class="result-label">{result.label}</p>'
        f'<p class="result-confidence">top confidence &nbsp;·&nbsp; {result.confidence:.1%}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-tag">// all class scores</span>', unsafe_allow_html=True)

    # All class breakdown — sorted descending by score
    sorted_scores = sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)

    for class_name, score in sorted_scores:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                f'<div class="class-row">'
                f'  <span class="class-name">{class_name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(score))
        with col2:
            st.markdown(
                f'<div style="padding-top:0.6rem; font-family: JetBrains Mono, monospace; '
                f'font-size:0.82rem; color:#00ffc8; text-align:right;">'
                f'{score:.1%}</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)