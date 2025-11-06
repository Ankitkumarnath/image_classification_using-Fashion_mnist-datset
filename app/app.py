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

def pad_to_square(img, fill=255):
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    new_img = Image.new(img.mode, (size, size), color=fill)
    new_img.paste(img, ((size - w) // 2, (size - h) // 2))
    return new_img

def should_invert(img_gray):
    corners = [
        img_gray.getpixel((0,0)),
        img_gray.getpixel((img_gray.width-1,0)),
        img_gray.getpixel((0,img_gray.height-1)),
        img_gray.getpixel((img_gray.width-1,img_gray.height-1)),
    ]
    return (sum(corners) / 4.0) > 128

def preprocess(img: Image.Image):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = pad_to_square(img, fill=255)
    img = img.resize((28,28), Image.Resampling.LANCZOS)
    if should_invert(img):
        img = ImageOps.invert(img)
    arr = np.array(img, dtype="float32") / 255.0
    arr = arr[..., None][None, ...]
    return arr

st.title("Fashion-MNIST CNN")
st.write("Upload an image (grayscale).This app is trained on grayscale data of FASHIONMNIST  so it's capable of detecting only gray scale image so only provide grayscale image The app normalizes to Fashion-MNIST style for better accuracy.")

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
    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    st.subheader(f"Prediction: {class_names[idx]}")
    st.write("Probabilities:")
    for i, p in enumerate(probs):
        st.write(f"{class_names[i]}: {p:.3f}")
