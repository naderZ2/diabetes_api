from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Input features
features = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_Male', 'gender_Other',
    'smoking_history_current', 'smoking_history_ever', 'smoking_history_former',
    'smoking_history_never', 'smoking_history_not current'
]

@app.route('/')
def home():
    return {"message": "API is running"}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        input_data = [data[feature] for feature in features]
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        return jsonify({'prediction': int(prediction)})
    except KeyError as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
