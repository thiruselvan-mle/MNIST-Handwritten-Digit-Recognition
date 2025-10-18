import streamlit as st
import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# =========================
# Load the Trained Model
# =========================

Base_Dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_TYPE = "CNN"  

if MODEL_TYPE == "CNN":
    model = load_model(os.path.join(Base_Dir, "models/cnn_model.keras"))
else:
    model = load_model(os.path.join(Base_Dir, "models/mlp_model.keras"))


# =========================
# Streamlit Page Setup
# =========================

st.set_page_config(page_title="MNIST Digit Recognition", layout="centered")

# Custom Styling (Professional Header)
st.markdown("""
<style>
    .main-title {
        font-size:36px;
        text-align:center;
        font-weight:bold;
        color:#00BFFF;
    }
    .sub-text {
        text-align:center;
        font-size:18px;
        color:gray;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h3><p class='main-title'>MNIST Handwritten Digit Recognition</p></h3>", unsafe_allow_html=True)
st.markdown("<h5><p class='sub-text'>Draw or upload a digit (0–9) below to see the model prediction.</p></h5>", unsafe_allow_html=True)


# =========================
# Sidebar Input Option
# =========================

st.sidebar.header("⚙️ Input Options")
input_method = st.sidebar.radio("Choose input type:", ["Draw Digit", "Upload Image"])

st.sidebar.markdown("---")
st.sidebar.info("""
**Model:** CNN  
**Dataset:** MNIST  
**Developer:** Thiruselvan M  
""")


# =========================
# Draw Digit Interface
# =========================

if input_method == "Draw Digit":
    st.write("🎨 Draw your digit below:")
    canvas_result = st_canvas(
        fill_color="#000000",  # Black fill
        stroke_width=10,
        stroke_color="#FFFFFF",  # White stroke
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Check if user actually drew something
    if canvas_result.image_data is not None:
        img = canvas_result.image_data.astype("uint8")
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # If canvas is empty (mostly black)
        if np.mean(gray) < 7:
            st.warning("⚠️ Please draw a digit to activate prediction.")
        else:
            # Preprocess drawn image
            img = cv2.resize(gray, (28, 28))
            img = img / 255.0
            img = img.reshape(1, 28, 28, 1)

            # Show Predict button only after drawing
            if st.button("🔍 Predict"):
                with st.spinner("🤖 Analyzing your digit..."):
                    probs = model.predict(img)
                    pred = np.argmax(probs)
                    confidence = np.max(probs) * 100

                st.success(f"Predicted Digit: **{pred}** ({confidence:.2f}% confidence)")
                st.balloons()


# =========================
# Upload Image Interface
# =========================

elif input_method == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload a digit image (28x28 or larger):", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("L")
        img = np.array(img)
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = img.reshape(1, 28, 28, 1)

        st.image(img.reshape(28, 28), caption="🖼️ Processed Image", width=150)

        if st.button("🔍 Predict"):
            with st.spinner("🤖 Analyzing your image..."):
                probs = model.predict(img)
                pred = np.argmax(probs)
                confidence = np.max(probs) * 100

            st.success(f"Predicted Digit: **{pred}** ({confidence:.2f}% confidence)")
            st.balloons()


# =========================
# Footer Section
# =========================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>👨‍💻 Developed by <b>Thiruselvan M</b> | MNIST Digit Recognition Project</div>",
    unsafe_allow_html=True
)
