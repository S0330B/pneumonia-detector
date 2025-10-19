import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("models/pneumonia_model.h5")

labels = ["PNEUMONIA","NORMAL"]
img_size = 150

st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image, and the model will predict if it shows pneumonia or not.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # convert to grayscale
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    
    img = np.array(image)
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = img.reshape(1, img_size, img_size, 1)
    
    prediction = model.predict(img)
    prob = float(prediction[0][0])
    pred_label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if pred_label == "PNEUMONIA" else 1 - prob

    st.write(f"Prediction: **{pred_label}** ({confidence*100:.2f}% confidence)")
