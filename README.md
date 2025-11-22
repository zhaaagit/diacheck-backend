# Diabetes Risk Prediction Website

A web application that predicts diabetes risk based on user health factors using machine learning.

## Features

- **Interactive Web Interface**: User-friendly form to input health factors
- **Real-time Predictions**: Get instant diabetes risk assessment
- **Machine Learning Model**: Trained on diabetes health dataset
- **Risk Categories**: Low, Medium, and High risk classifications
- **Responsive Design**: Works on desktop and mobile devices

## Input Parameters

Users need to provide the following information:
- **Gender**: Male or Female
- **Age**: In months (e.g., 360 = 30 years)
- **BMI**: Body Mass Index (e.g., 25.5)
- **Hypertension**: Yes or No
- **Heart Disease**: Yes or No
- **Smoking History**: Never, Current, Former, Ever, or Not Current

## Installation & Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Train the Model

Run the model training script to create the machine learning model:

```powershell
python train_model.py
```

This will:
- Load the diabetes dataset
- Train a Random Forest classifier
- Save the model and encoders for later use

Expected output:
```
Dataset loaded successfully!
Training accuracy: ~0.95
Test accuracy: ~0.94
Model saved successfully!
```

### 3. Start the Backend Server

In a terminal, run:

```powershell
python app.py
```

You should see:
```
==================================================
Diabetes Risk Prediction API
==================================================
Starting server on http://localhost:5000
```

### 4. Open the Website

Open `index.html` in your web browser or serve it via a local server:

```powershell
# Using Python's built-in server
python -m http.server 8000
```

Then visit: http://localhost:8000

## How It Works

1. **User Input**: User fills out the form with their health information
2. **Data Processing**: The frontend sends the data to the backend API
3. **Model Prediction**: The machine learning model processes the data
4. **Risk Assessment**: Returns a diabetes risk probability
5. **Result Display**: Shows the risk level with recommendations

## Risk Levels

- **🟢 Low Risk** (< 40%): Continue maintaining a healthy lifestyle
- **🟡 Medium Risk** (40-70%): Consider regular check-ups and lifestyle changes
- **🔴 High Risk** (≥ 70%): Consult with healthcare professionals

## Files Structure

```
c:\web\
├── index.html              # Frontend web interface
├── app.py                  # Flask backend API
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── Dataset_Rill.csv        # Original dataset
├── diabetes_model.pkl      # Trained model (generated)
├── encoder_gender.pkl      # Gender encoder (generated)
└── encoder_smoking.pkl     # Smoking encoder (generated)
```

## Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask, Python
- **Machine Learning**: scikit-learn, Random Forest Classifier
- **Data Processing**: pandas, numpy

## Important Disclaimer

⚠️ This tool is for **educational and informational purposes only**. It should **NOT** be used as a replacement for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for accurate medical assessment.

## Troubleshooting

### "Unable to connect to prediction server"
- Make sure the backend is running: `python app.py`
- Verify it's running on port 5000
- Check for any firewall issues

### "Module not found" errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Port 5000 already in use
- Kill the process using port 5000 or change the port in `app.py`

## Future Enhancements

- Add more health parameters (HbA1c, blood glucose levels)
- Implement user data persistence
- Add data visualization and analytics
- Deploy to cloud platform
- Add multiple language support

## License

This project is for educational purposes.
