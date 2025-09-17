import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Streamlit Cloud has no GPU


import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# -----------------------------
# Download model from Hugging Face
# -----------------------------
MODEL_PATH = hf_hub_download(
    repo_id="https://huggingface.co/Abdulmoiz123/cat-dog-classifier/tree/main",  # Your Hugging Face repo
    filename="cat_vs_dog_model.keras",                        # Model file name inside HF repo
    local_dir="."                               # Save locally in Streamlit
)

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



