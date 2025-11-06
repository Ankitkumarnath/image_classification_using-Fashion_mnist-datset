
import json
import numpy as np
import tensorflow as tf
from PIL import Image

def load_model_and_labels(model_path="models/model.h5", labels_path="models/class_names.json"):
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r") as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_image(img: Image.Image):
    img = img.convert("L").resize((28,28))  # Fashion-MNIST is 28x28 grayscale
    arr = np.array(img, dtype="float32")/255.0
    arr = arr[..., None]  # add channel
    arr = np.expand_dims(arr, 0)  # add batch
    return arr

def predict_image(path):
    model, class_names = load_model_and_labels()
    img = Image.open(path)
    x = preprocess_image(img)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs.tolist()

if __name__ == "__main__":
    import sys
    label, p, _ = predict_image(sys.argv[1])
    print(f"Predicted: {label} ({p:.3f})")
