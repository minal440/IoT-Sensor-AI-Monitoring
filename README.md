# IoT Sensor AI Monitoring System

This project implements a Deep Learning based system for automatically classifying IoT sensor alerts.

The system analyzes sensor messages and predicts the type of alert such as battery issues, security alerts, temperature warnings, or device faults.

## Features

- Deep Learning model using LSTM
- Text preprocessing with Tokenizer
- Interactive Streamlit dashboard
- Real-time alert classification
- Confidence score prediction

## Technologies

Python  
TensorFlow / Keras  
Scikit-learn  
Streamlit  
Pandas  

## Project Structure

datasets - dataset files  
src - training and prediction code  
models - trained models  
app - deployment dashboard  

## Applications

Industrial IoT monitoring  
Smart factory alerts  
Predictive maintenance  
Device health monitoring

## How to Run

Train model

python src/train_lstm.py

Run dashboard

streamlit run app.py