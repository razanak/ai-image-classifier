"""
AI Image Classifier
--------------------
A simple web app that uses a pretrained MobileNetV2 model (trained on
ImageNet, 1000 everyday object categories) to classify images you upload.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

# ---------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI Image Classifier", page_icon="🤖", layout="centered")
st.title("🤖 AI Image Classifier")
st.write(
    "Upload a photo and this app will tell you what it thinks is in it, "
    "using a pretrained deep learning model (MobileNetV2)."
)


# ---------------------------------------------------------------------
# Load the model once and cache it so it doesn't reload on every click
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


model = load_model()

# ---------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your image", use_container_width=True)

    with st.spinner("Thinking..."):
        # MobileNetV2 expects 224x224 images
        resized = image.resize((224, 224))
        img_array = np.array(resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        results = decode_predictions(predictions, top=5)[0]

    st.subheader("Predictions")
    for _, label, confidence in results:
        label = label.replace("_", " ").title()
        st.write(f"**{label}** — {confidence * 100:.2f}%")
        st.progress(float(confidence))
else:
    st.info("👆 Upload a JPG or PNG image to get started.")

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit • Model: MobileNetV2 (ImageNet)")
