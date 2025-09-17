import os
import urllib.request
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Disable GPU to avoid CUDA errors if no GPU is available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Hugging Face direct model link and local path
MODEL_URL = "https://huggingface.co/Abdulmoiz123/cat-dog-classifier/resolve/main/cat_vs_dog_model.keras"
MODEL_PATH = "cat_vs_dog_model.keras"

# Load model (cached)
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model (~475MB). Please wait...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully!")
    model = load_model(MODEL_PATH, compile=False)
    return model

my_model = load_my_model()

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
