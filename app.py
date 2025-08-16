import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image

MODEL_ID = "1_ME01LJP56yiwFzRWRE0Kl5nxNY9bRjQ"
MODEL_PATH = "best_model_transfer.h5"

class_names = {0: "Covid", 1: "Normal", 2: "Viral Pneumonia"}

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = download_and_load_model()

# Streamlit UI
st.title("🩺 Chest X-ray Classification")
st.write("Upload a chest X-ray image and let the model predict the condition.")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    img = image_data.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.subheader("🧠 Prediction Result")
    st.success(f"**{class_names[predicted_class]}** — Confidence: {confidence:.2%}")

    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob:.2%}")
