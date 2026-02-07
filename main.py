import os
import joblib
from train import model, scaler

os.makedirs("models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)

joblib.dump(model, "models/heart_model.pkl")
joblib.dump(scaler, "scalers/heart_scaler.pkl")

print(" Model and scaler saved successfully")