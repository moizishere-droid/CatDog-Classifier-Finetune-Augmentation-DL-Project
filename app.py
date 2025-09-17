import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

@st.cache_resource
def load_cnn_model():
    model = load_model("cat_vs_dog_model.keras")
    return model

my_model = load_cnn_model()


def preprocess_image(image: Image.Image):
    # Resize image to model input size
    image = image.resize((224, 224))

    # Convert to numpy
    img_array = np.asarray(image).astype("float32")

    # Normalize (if trained with /255)
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


st.set_page_config(page_title="🐶🐱 Dog vs Cat Classifier", layout="centered")

st.title("🐶🐱 Dog vs Cat Classifier")
st.write("Upload an image of a **Dog** or **Cat** and let the CNN predict!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a Dog/Cat Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    processed_img = preprocess_image(image)

    # Predict
    pred = my_model.predict(processed_img)
    probability = float(pred[0][0])  # convert tensor to float

    # Display result
    st.subheader("🔍 Prediction Result:")
    if probability > 0.5:
        st.success(f"**Dog 🐶** (Confidence: {probability*100:.2f}%)")
    else:
        st.success(f"**Cat 🐱** (Confidence: {(1-probability)*100:.2f}%)")

    # Show probability bar
    st.progress(int(probability * 100) if probability > 0.5 else int((1-probability) * 100))
