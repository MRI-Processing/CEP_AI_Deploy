

import numpy as np

def predict_heart_disease(age, cholesterol, model):
    # Reshape input to match 2D array expected by scikit-learn
    input_data = np.array([[age, cholesterol]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Heart Disease Detected " if prediction == 1 else "No Heart Disease Detected"
    return f"{result} (Confidence: {probability * 100:.2f}%)"