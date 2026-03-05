import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data = pd.read_csv("D:/IoT_Project/datasets/iot_message_dataset.csv")

texts = data["message"]
labels = data["category"]


encoder = LabelEncoder()
y = encoder.fit_transform(labels)

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

X = pad_sequences(sequences, maxlen=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()

model.add(Embedding(input_dim=2000, output_dim=128))

model.add(LSTM(128, return_sequences=True))

model.add(LSTM(64))

model.add(Dense(64, activation="relu"))

model.add(Dense(len(set(y)), activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=4,
    validation_data=(X_test, y_test)
)

predictions = model.predict(X_test)
pred_labels = predictions.argmax(axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, pred_labels))


cm = confusion_matrix(y_test, pred_labels)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("D:/IoT_Project/images/confusion_matrix.png")


plt.close()

plt.figure(figsize=(6,4))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.savefig("D:/IoT_Project/images/accuracy_plot.png")

plt.close()


model.save("../models/lstm_classifier.h5")

pickle.dump(tokenizer, open("D:/IoT_Project/models/tokenizer.pkl", "wb"))
pickle.dump(encoder, open("D:/IoT_Project/models/encoder.pkl", "wb"))

print("\nModel and tokenizer saved successfully.")