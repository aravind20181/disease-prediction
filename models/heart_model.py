import pandas as pd
import os


# Step 1: Load Dataset

file_path = "./data/heart.csv"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    print(f"Current working directory: {os.getcwd()}")
    exit()
else:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded dataset with shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df["target"].value_counts())
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Train the Model

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)


#Evaluate

from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))





# Show probability of disease for test samples
probs = model.predict_proba(X_test)




import joblib

# Create directories if they don't exist
os.makedirs("../models", exist_ok=True)
os.makedirs("../scalers", exist_ok=True)

# Save model and scaler
joblib.dump(model, "../models/heart_model.pkl")
joblib.dump(scaler, "../scalers/heart_scaler.pkl")

print("Heart disease model and scaler saved successfully") 

