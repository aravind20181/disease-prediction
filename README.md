# Heart Disease Risk Predictor 🫀

A **machine learning web application** that predicts the risk of heart disease based on patient health metrics. The app takes inputs such as age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression (oldpeak), slope of the peak exercise ST segment, number of major vessels, and thalassemia type. It outputs a **risk level: LOW 🟢, MEDIUM 🟡, or HIGH 🔴**.

This project demonstrates an **end-to-end machine learning workflow**, including data preprocessing, feature scaling, model training, evaluation, and deployment using Flask.  

> ⚠️ **Important Note:** This project is for **educational purposes only**. It is **not suitable for real-world medical diagnosis** 
---

## 📊 Features

- Predicts **heart disease risk levels** based on patient vitals and test results.  
- Real-time predictions via a **user-friendly Flask web interface**.  
- Handles **data preprocessing**, including duplicate removal and numeric feature scaling.  
- Displays **probability-based risk scores** to make predictions intuitive.  
- Saves and loads trained model and scaler using `joblib`.

---

## ⚙️ Technologies Used

- **Python** – core programming  
- **Pandas & NumPy** – data manipulation  
- **scikit-learn** – machine learning modeling  
- **Flask** – web deployment  
- **joblib** – model serialization  

---
## ⚠️ Limitations

- This model is trained on a **small, publicly available dataset**, which may not represent real-world patient populations accurately.
- Some medical features are **simplified or encoded numerically**, which may reduce clinical realism.
- The model does **not account for age-adjusted or condition-specific heart rate ranges**, which can affect prediction accuracy.
- Logistic Regression assumes **linear relationships** between features and outcomes, which may not capture complex medical patterns.
- Risk categories (LOW / MEDIUM / HIGH) are based on **probability thresholds**, not clinical guidelines.
- The application does **not perform medical validation**, lab verification, or professional review.
- Predictions should **not be interpreted as medical advice** or diagnosis.

> This project is intended **only for learning and demonstration purposes** and is **not suitable for real-world medical use**.

---


