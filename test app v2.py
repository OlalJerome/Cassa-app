import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

st.set_page_config(page_title="Cassava Disease Detector", page_icon="🌿", layout="centered")

st.title("🌿 Cassava Disease Detector")
st.markdown("MobileNet • Custom Trained")

class_names = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy"
]

advice_dict = {
    0: "🟡 Remove affected leaves and avoid overhead watering.",
    1: "🔴 Uproot and destroy infected plants immediately.",
    2: "🟡 Apply miticides and improve soil nutrition.",
    3: "🔴 Use clean planting materials and control whiteflies.",
    4: "🟢 Plant is healthy. Maintain good farming practices."
}

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("best_mobilenet_cassava.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()
st.success("✅ Model loaded successfully!")

# Image Input
option = st.radio("Choose input method:", ["📤 Upload Image", "📸 Take Photo"], horizontal=True)

image = None
if option == "📤 Upload Image":
    uploaded = st.file_uploader("Upload cassava leaf image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
elif option == "📸 Take Photo":
    camera = st.camera_input("Capture image")
    if camera:
        image = Image.open(camera).convert("RGB")

if image:
    st.image(image, caption="Selected Image", use_container_width=True)

    if st.button("🔍 Predict Disease", type="primary"):
        with st.spinner("Analyzing..."):
            try:
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array, verbose=0)[0]
                pred_idx = np.argmax(predictions)
                confidence = float(predictions[pred_idx] * 100)

                st.success(f"**Prediction:** {class_names[pred_idx]}")
                st.info(f"**Confidence:** {confidence:.2f}%")

                st.markdown("### 📌 Advice")
                st.write(advice_dict[pred_idx])

            except Exception as e:
                st.error(f"Prediction error: {e}")
else:
    st.info("Please upload or capture a cassava leaf image.")

st.caption("Made with ❤️ for Farmers")
