from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the trained models and features
try:
    # Quick model
    model_quick = joblib.load('diabetes_model_quick.pkl')
    numeric_features_quick = joblib.load('numeric_features_quick.pkl')
    
    # Full model
    model_full = joblib.load('diabetes_model_full.pkl')
    numeric_features_full = joblib.load('numeric_features_full.pkl')
    
    # Shared features
    categorical_features = joblib.load('categorical_features.pkl')
    binary_features = joblib.load('binary_features.pkl')
    
    print("✓ Both models and features loaded successfully!")
except FileNotFoundError as e:
    print(f"✗ Error loading model files: {e}")
    print("Please run train_better_model.py first to train the models.")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Determine which model to use
        model_type = data.get('model_type', 'quick')  # default to quick
        
        if model_type not in ['quick', 'full']:
            return jsonify({'error': 'Invalid model_type. Use "quick" or "full"'}), 400
        
        # Select appropriate model
        model = model_quick if model_type == 'quick' else model_full
        
        # Validate required fields
        required_fields = ['gender', 'age', 'bmi', 'hypertension', 'heart_disease', 'smoking_history']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # For full model, also require lab tests
        if model_type == 'full':
            if 'HbA1c_level' not in data or 'blood_glucose_level' not in data:
                return jsonify({'error': 'Full model requires HbA1c_level and blood_glucose_level'}), 400
        
        # Convert hypertension and heart_disease to integers
        try:
            hypertension = int(data['hypertension'])
            heart_disease = int(data['heart_disease'])
        except (ValueError, TypeError):
            return jsonify({'error': 'hypertension and heart_disease must be 0 or 1'}), 400
        
        # Scale values to match the dataset format:
        # - Age: multiply by 10 (user input in years → dataset format)
        # - BMI: multiply by 100 (user input as decimal → dataset format)
        # - HbA1c: multiply by 10 (if provided)
        # - Blood Glucose: no scaling needed
        age_scaled = data['age'] * 10
        bmi_scaled = data['bmi'] * 100
        
        # Create DataFrame with the exact features expected by the model
        if model_type == 'quick':
            input_data = pd.DataFrame({
                'gender': [data['gender']],
                'age': [age_scaled],
                'bmi': [bmi_scaled],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'smoking_history': [data['smoking_history']]
            })
        else:  # full model
            hba1c_scaled = data['HbA1c_level'] * 10
            blood_glucose = data['blood_glucose_level']
            
            input_data = pd.DataFrame({
                'gender': [data['gender']],
                'age': [age_scaled],
                'bmi': [bmi_scaled],
                'HbA1c_level': [hba1c_scaled],
                'blood_glucose_level': [blood_glucose],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'smoking_history': [data['smoking_history']]
            })
        
        # Make prediction using the selected model
        diabetes_probability = model.predict_proba(input_data)[0][1]
        prediction = int(model.predict(input_data)[0])
        
        return jsonify({
            'diabetes_probability': float(diabetes_probability),
            'prediction': prediction,
            'model_type': model_type,
            'input_data': data
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'message': 'Diabetes Risk Prediction API is running'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host='0.0.0.0', port=port)



