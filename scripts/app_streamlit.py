from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from snapcheck.triage import QualityGate

st.set_page_config(page_title="TeleDerm SnapCheck", layout="wide")

st.title("TeleDerm SnapCheck Demo")

checkpoint_path = st.sidebar.text_input("Quality model checkpoint", "models/snapcheck_quality.pt")
threshold = st.sidebar.slider("Fail threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

quality_gate = None
if Path(checkpoint_path).exists():
    thresholds = {label: threshold for label in [
        "blur",
        "motion_blur",
        "low_brightness",
        "high_brightness",
        "low_contrast",
        "high_contrast",
        "noise",
        "shadow",
        "obstruction",
        "framing",
        "low_resolution",
        "overall_fail",
    ]}
    quality_gate = QualityGate(Path(checkpoint_path), thresholds)
else:
    st.warning("Checkpoint not found. Upload an image to visualize only.")

uploaded = st.file_uploader("Upload a derm image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)
    if quality_gate:
        scores = quality_gate.assess(image)
        st.subheader("Quality probabilities")
        st.json(scores)
        st.markdown(f"**Retake recommended:** {'Yes' if quality_gate.should_retake(scores) else 'No'}")
