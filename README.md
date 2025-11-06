Fashion-MNIST Image Classification

A CNN-based image classifier built using TensorFlow/Keras to recognize 10 types of fashion items from the Fashion-MNIST dataset. Deployed using Streamlit.

⚙️ Overview

The model classifies grayscale 28×28 images into these categories:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

🧩 Model Architecture

Conv2D → BatchNorm → MaxPooling (×3)

Flatten → Dense(128, ReLU) → Dropout(0.4) → Dense(10, Softmax)

Optimizer: Adam

Loss: Sparse Categorical Crossentropy

Accuracy: ~93%

📊 Results
Metric	Value
Train Accuracy	93.2%
Val Accuracy	93.0%
Test Accuracy	92.8%

Visuals:

Accuracy Curve →

<img width="586" height="455" alt="acc_curve" src="https://github.com/user-attachments/assets/a0cd38d7-201d-40a4-8457-3fba6e05dc5e" />


Loss Curve → 

<img width="569" height="455" alt="loss_curve" src="https://github.com/user-attachments/assets/fccff4a6-1c0e-41e9-ba28-7ef5a96aadb8" />


Confusion Matrix → 

<img width="2400" height="1800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/35bc51ad-1052-4ea1-b336-ef760a90863c" />


Sample Predictions →

<img width="511" height="472" alt="sample_predictions" src="https://github.com/user-attachments/assets/bcc1c65f-4235-409d-9b22-c0e1ff2435f2" />


Run the App
pip install -r requirements.txt
streamlit run streamlit_app.py

📁 Structure
├── models/model.h5
├── streamlit_app.py
├── classification_report.txt
├── acc_curve.png
├── loss_curve.png
├── confusion_matrix.png
├── sample_predictions.png
└── README.md



