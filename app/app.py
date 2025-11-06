import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(page_title="Fashion-MNIST Classifier", page_icon="", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/model.h5")
    with open("models/class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

# --- NEW: your preprocess_image function (used below) ---
def preprocess_image(image, target_size, dataset_type=''):
    """
    Preprocess uploaded image:
    - Converts RGB to grayscale for Fashion-MNIST
    - Normalizes pixels to [0,1]
    - Resizes to model input shape
    """
    if dataset_type == 'fashion_mnist':
        # Convert to grayscale (Fashion-MNIST is 1 channel)
        if image.mode != 'L':
            image = image.convert('L')
            st.sidebar.info("📸 Converted image to grayscale for Fashion-MNIST model.")

        # Equalize brightness & contrast for clearer features
        image = ImageOps.autocontrast(image)
        image = ImageOps.equalize(image)

    # Resize to model input size
    image = image.resize(target_size)

    # Convert to array and normalize
    img_array = np.array(image).astype('float32') / 255.0

    # Add missing grayscale channel
    if dataset_type == 'fashion_mnist' and len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

st.title("Fashion-MNIST CNN")
st.write("Upload an image (grayscale). This app is trained on grayscale data of FASHIONMNIST, "
         "so it’s optimized for grayscale input. The app normalizes to Fashion-MNIST style for better accuracy.")

try:
    model, class_names = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model. Train it first. Error: {e}")

uploaded = st.file_uploader("Upload PNG/JPG", type=["png","jpg","jpeg"])
if uploaded and model_loaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_column_width=True)

    # --- CHANGED: use preprocess_image instead of custom preprocess ---
    x = preprocess_image(image, target_size=(28, 28), dataset_type='fashion_mnist')

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    st.subheader(f"Prediction: {class_names[idx]}")

    # Keep your probabilities view (unchanged behavior)
    st.write("Probabilities:")
    for i, p in enumerate(probs):
        st.write(f"{class_names[i]}: {p:.3f}")
