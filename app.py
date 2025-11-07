import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import pickle

# Load the trained Gradient Boosting model
with open("gbc_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below and click Predict!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",  # Black brush
    stroke_width=15,
    stroke_color="white",  # White drawing
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert drawn image to PIL image
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")  # grayscale
    # Resize to 8x8 pixels (like sklearn digits dataset)
    img_resized = img.resize((8, 8), Image.ANTIALIAS)
    img_array = np.array(img_resized)
    
    # Invert colors and scale to 0-16
    img_array = 16 - (img_array / 255.0 * 16)
    img_array = img_array.flatten().reshape(1, -1)
    
    # Predict
    if st.button("Predict"):
        pred = model.predict(img_array)
        st.write(f"Predicted digit: **{pred[0]}**")
        st.image(img_resized.resize((100, 100)), caption="Resized 8x8 input", width=100)
