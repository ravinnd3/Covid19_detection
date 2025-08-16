import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import PIL

# Load model architecture
@st.cache_resource
def load_model():
    model = MobileNetV2(weights=None)  # No weights loaded initially
    model.load_weights('model_weights.weights.h5')  # Load your saved weights
    return model

model = load_model()

# Streamlit UI
st.title("🔍 Image Classification with Transfer Learning")
st.write("Upload an image and let the model predict what it sees!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]

    st.subheader("📊 Top Predictions:")
    for i, (imagenetID, label, prob) in enumerate(decoded):
        st.write(f"{i+1}. **{label}** — {prob:.2%}")
