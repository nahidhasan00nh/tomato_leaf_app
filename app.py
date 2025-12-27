import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

# Load model
MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Streamlit UI
st.title("Tomato Leaf Disease Classifier")

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    st.success(f"Predicted Class: {class_names[class_idx]}")
