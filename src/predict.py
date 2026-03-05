import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("D:/IoT_Project/models/models/lstm_classifier.h5")

tokenizer = pickle.load(open("D:/IoT_Project/models/tokenizer.pkl","rb"))
encoder = pickle.load(open("D:/IoT_Project/models/encoder.pkl","rb"))

msg = input("Enter sensor message: ")

seq = tokenizer.texts_to_sequences([msg])

pad = pad_sequences(seq, maxlen=10)

pred = model.predict(pad)

label = encoder.inverse_transform([np.argmax(pred)])

print("Prediction:", label[0])