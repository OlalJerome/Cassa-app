import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

st.set_page_config(page_title="Cassava Disease Detector", page_icon="🌿", layout="centered")

st.title("🌿 Cassava Disease Detector")
st.markdown("MobileNet Model • Currently Training")

# ====================== CLASS NAMES ======================
# IMPORTANT: Update this order according to your dataset folders
class_names = [
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Bacterial Blight (CBB)",
    "Healthy",
    "Cassava Mosaic Disease (CMD)",
    "Cassava Green Mottle (CGM)"
    
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
    model_path = "best_mobilenet_cassava.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found: {model_path}")
        st.stop()
    
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# Show model summary info
st.info(f"Model Input Shape: {model.input_shape}")

# ====================== IMAGE INPUT ======================
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
        with st.spinner("Analyzing image..."):
            try:
                # Preprocess
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
                predictions = model.predict(img_array, verbose=0)[0]
                pred_idx = np.argmax(predictions)
                confidence = float(predictions[pred_idx] * 100)

                st.success(f"**Prediction:** {class_names[pred_idx]}")
                st.info(f"**Confidence:** {confidence:.2f}%")

                st.markdown("### 📌 Advice")
                st.write(advice_dict[pred_idx])

                # Show all probabilities
                st.markdown("### 📊 All Probabilities")
                for i, prob in enumerate(predictions):
                    st.write(f"{class_names[i]} → **{prob*100:.2f}%**")

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload or capture a cassava leaf image.")

st.caption("Powered by MobileNet • Training in Progress")