import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 📦 Model download info
MODEL_ID = "1_ME01LJP56yiwFzRWRE0Kl5nxNY9bRjQ"
MODEL_PATH = "best_model_transfer.h5"

# 🏷 Class labels
class_names = {0: "Covid", 1: "Normal", 2: "Viral Pneumonia"}

# 🧠 Load model with caching
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
    if os.path.getsize(MODEL_PATH) < 100000:  # sanity check
        st.error("❌ Downloaded file seems incomplete. Please verify the Drive link.")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model. Please check the model file format.\n\n**Error:** {e}")
        st.stop()

model = download_and_load_model()

# 🎯 App title and instructions
st.title("🩺 Chest X-ray Classification")
st.markdown("""
Upload a chest X-ray image and let the model predict the condition.  
Supported classes: **Covid**, **Normal**, **Viral Pneumonia**
""")

# 📁 File uploader
uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image_data = Image.open(uploaded_file).convert("RGB")
        st.image(image_data, caption="🖼️ Uploaded Image", use_column_width=True)

        # 🧼 Preprocess image
        img = image_data.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

        # ✅ Debug input shape
        st.write(f"🔍 Input shape: {img_array.shape}")
        st.write(f"🔍 Input type: {type(img_array)}")

        # 🔮 Make prediction
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        # 📊 Display results
        st.subheader("🧠 Prediction Result")
        st.success(f"**{class_names[predicted_class]}** — Confidence: {confidence:.2%}")

        # 📊 Class Probabilities as Bar Chart
        st.subheader("📊 Class Probabilities")
        fig, ax = plt.subplots()
        ax.bar(class_names.values(), predictions[0], color=["red", "green", "blue"])
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing the image or making prediction.\n\n**Error:** {e}")
