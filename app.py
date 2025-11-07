import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pickle
import io

# ----------------------------
# Load your trained model
# ----------------------------
with open("gbc_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🖌️ Handwritten Digit Recognition")
st.write("You can either **draw a digit** or **upload an image** and click Predict!")

# ----------------------------
# Image uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

# ----------------------------
# Drawing section
# ----------------------------
st.write("Or draw a digit manually:")

# Canvas dimensions
canvas_size = 280
st.write("Draw your digit in the canvas below (white on black) and click 'Save Drawing'")

# Simple drawing using Streamlit's `st.image` and numpy
if 'drawing' not in st.session_state:
    st.session_state.drawing = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

def reset_canvas():
    st.session_state.drawing = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

if st.button("Clear Canvas"):
    reset_canvas()

# Display current drawing
st.image(st.session_state.drawing, width=canvas_size)

# File uploader takes precedence over drawing
image_to_predict = None
if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file).convert("L")
elif st.session_state.drawing is not None:
    # Use drawing if no uploaded file
    image_to_predict = Image.fromarray(st.session_state.drawing).convert("L")

# ----------------------------
# Prediction
# ----------------------------
if image_to_predict is not None:
    # Resize to 8x8 like sklearn digits dataset
    img_resized = image_to_predict.resize((8, 8), Image.ANTIALIAS)
    img_array = np.array(img_resized)
    
    # Scale 0-16 (sklearn digits dataset scale)
    img_array = 16 - (img_array / 255.0 * 16)
    img_array = img_array.flatten().reshape(1, -1)
    
    if st.button("Predict"):
        pred = model.predict(img_array)
        st.write(f"Predicted digit: **{pred[0]}**")
        st.image(img_resized.resize((100, 100)), caption="Resized 8x8 input", width=100)
