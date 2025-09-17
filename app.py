import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU (Streamlit Cloud has no GPU)

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests

MODEL_PATH = "cat_vs_dog_model.keras"

# -----------------------------
# Download model directly from Hugging Face
# -----------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model from Hugging Face..."):
        url = "https://huggingface.co/Abdulmoiz123/cat-dog-classifier/resolve/main/cat_vs_dog_model.keras?download=true"
        r = requests.get(url, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# -----------------------------
# Load model (cache for speed)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Image preprocessing
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = np.asarray(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.set_page_config(page_title="🐶🐱 Dog vs Cat Classifier", layout="centered")
st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image of a **Dog** or **Cat** and let the CNN predict!")

uploaded_file = st.file_uploader("📂 Upload a Dog/Cat Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_img = preprocess_image(image)
    pred = my_model.predict(processed_img)
    probability = float(pred[0][0])

    st.subheader("🔍 Prediction Result:")
    if probability > 0.5:
        st.success(f"**Dog 🐶** (Confidence: {probability*100:.2f}%)")
        st.progress(int(probability * 100))
    else:
        st.success(f"**Cat 🐱** (Confidence: {(1-probability)*100:.2f}%)")
        st.progress(int((1-probability) * 100))




