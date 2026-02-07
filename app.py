from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model & scaler with error handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    heart_model = joblib.load(os.path.join(BASE_DIR, "models", "heart_model.pkl"))
    heart_scaler = joblib.load(os.path.join(BASE_DIR, "scalers", "heart_scaler.pkl"))
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    heart_model = None
    heart_scaler = None


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    patient_data = {}
    prediction_text = ""
    risk_level = ""
    charts = {}

    if request.method == "POST":
        # Collect input with validation
        try:
            age = int(request.form["age"])
            sex = int(request.form["sex"])
            cp = int(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = int(request.form["fbs"])
            restecg = int(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = int(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = int(request.form["slope"])
            ca = int(request.form["ca"])
            thal = int(request.form["thal"])
            # Convert inputs

            # Validate ranges
            if not (10 <= age <= 120):
                return "Invalid input: Age must be between 10 and 120"
            if not (0 <= chol <= 600):
                return "Invalid input: Cholesterol must be between 0 and 600"
            if not (0 <= trestbps <= 250):
                return "Invalid input: Resting BP must be between 0 and 250"
                
        except ValueError:
            return "Invalid input: Please enter valid numeric values"
        except KeyError as e:
            return f"Invalid input: Missing field {e}"

        # Store raw patient data
        patient_data = {
            "Age": age,
            "Sex": "Male" if sex == 1 else "Female",
            "Chest Pain Type": cp,
            "Resting BP": trestbps,
            "Cholesterol": chol,
            "Fasting BS": fbs,
            "Rest ECG": restecg,
            "Max Heart Rate": thalach,
            "Exercise Angina": exang,
            "Oldpeak": oldpeak,
            "Slope": slope,
            "CA": ca,
            "Thal": thal
        }

        # Prepare input for model
        input_data = np.array([age, sex, cp, trestbps, chol, fbs,
                               restecg, thalach, exang, oldpeak,
                               slope, ca, thal]).reshape(1, -1)
        input_scaled = heart_scaler.transform(input_data)

        # Prediction
        pred = heart_model.predict(input_scaled)[0]
        prob = heart_model.predict_proba(input_scaled)[0][1]

        prediction_text = "Heart Disease Detected!" if pred == 1 else "No Heart Disease"

        if prob > 0.7:
            risk_level = "HIGH 🔴"
        elif prob > 0.4:
            risk_level = "MEDIUM 🟡"
        else:
            risk_level = "LOW 🟢"

        # Charts data
        charts = {
            "risk_pie": {
                "labels": ["No Disease", "Disease"],
                "data": [round((1-prob)*100,2), round(prob*100,2)],
                "colors": ["#4CAF50", "#FF4136"]
            },
            "chol_line": {
                "labels": ["Cholesterol", "Resting BP", "Max Heart Rate", "Oldpeak"],
                "data": [chol, trestbps, thalach, oldpeak],
                "colors": ["#3498db"]
            }
        }

        return render_template("result.html",
                               prediction_text=prediction_text,
                               risk_level=risk_level,
                               patient_data=patient_data,
                               charts=charts)

    return render_template("heart.html")

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", False))
