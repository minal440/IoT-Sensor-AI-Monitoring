import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest

df = pd.read_csv("D:\IoT_Project\datasets\iot_messages_dataset.csv")

X = df.drop("label", axis=1)

model = IsolationForest(contamination=0.2)

model.fit(X)

pickle.dump(model, open("D:\IoT_Project\models\anomaly_model.pkl","wb"))

print("Anomaly model trained")