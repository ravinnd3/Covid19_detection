import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
from PIL import Image

# Constants
MODEL_URL = "https://drive.google.com/uc?export=download&id=1_ME01LJP56yiwFzRWRE0Kl5nxNY9bRjQ"
MODEL_PATH = "best_model_transfer.h5"
IMG_SIZE = (224, 224)  # Adjust if your model expects a different size

# Download and load model
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = download_and_load_model()

# Streamlit UI
st.title("🧠 Image Classification App")
st.write("Upload an image and let the model predict what it sees!")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image_data = Image.open(uploaded_file).convert("RGB")
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image_data.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0) / 255.0  # Normalize if needed

    # Predict
    predictions = model.predict(img_array)

    # Display predictions
    st.subheader("🔍 Prediction Output")
    st.write(predictions)

    # Optional: If you have class labels
    # class_names = ['cat', 'dog', 'car', ...]  # Replace with your actual labels
    # predicted_class = class_names[np.argmax(predictions)]
    # st.success(f"Predicted Class: {predicted_class}")
