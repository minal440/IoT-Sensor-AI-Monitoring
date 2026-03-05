import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models
model = load_model("D:/IoT_Project/models/lstm_classifier.h5")
tokenizer = pickle.load(open("D:/IoT_Project/models/tokenizer.pkl","rb"))
encoder = pickle.load(open("D:/IoT_Project/models/encoder.pkl","rb"))

st.title("IoT Sensor Alert Classification")

st.write(
    "Deep Learning system that automatically classifies IoT sensor alerts."
)

msg = st.text_input("Enter Sensor Message")

if st.button("Predict"):

    seq = tokenizer.texts_to_sequences([msg])
    pad = pad_sequences(seq, maxlen=10)

    pred = model.predict(pad)

    label = encoder.inverse_transform([np.argmax(pred)])

    confidence = np.max(pred)

    st.success(f"Prediction: {label[0]}")
    st.write(f"Confidence: {confidence:.2f}")

st.markdown("---")

st.subheader("Example Inputs")

st.write("battery critically low")
st.write("motion detected at door")
st.write("temperature sensor overheating")