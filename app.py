# app.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model
model = joblib.load('crop_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Prepare input features
    features = pd.DataFrame([data], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Predict
    prediction = model.predict(features)

    return jsonify({'crop': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
