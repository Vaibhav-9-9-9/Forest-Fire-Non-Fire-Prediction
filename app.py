import streamlit as st
import pickle
import numpy as np
import cv2

st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🔥",
    layout="centered"
)

st.title("🔥 Forest Fire vs Non-Fire Detection")

@st.cache_resource
def load_model():
    with open("forest_fire_nonfire.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
st.caption(f"Model classes: {model.classes_}")

def preprocess_image(uploaded_file, img_size=(128, 128)):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, img_size)
    X = img.flatten().reshape(1, -1)

    return img, X

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        img, X = preprocess_image(uploaded_file)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        prediction = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            confidence = np.max(proba) * 100
        else:
            confidence = None

        st.subheader("🧠 Prediction Result")

        if prediction in ["Fire", 1]:
            st.error("🔥 Fire Detected")
        else:
            st.success("🌲 No Fire Detected")

        if confidence:
            st.info(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.warning("⚠️ Image processing failed")
        st.info("🌲 No Fire Detected")
        st.caption(f"Debug: {e}")
