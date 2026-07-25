# 🧠 Stress Detection System using Machine Learning

A full-stack machine learning application for detecting human stress levels from physiological sensor data. The system automatically extracts statistical features from uploaded biosignal data, predicts the stress state using a trained Random Forest model, and presents the results through an interactive Streamlit dashboard.

---

## 🚀 Features

- Upload physiological sensor CSV files
- Automatic feature extraction from raw signals
- Interactive visualization of ECG, EDA, Respiration and Temperature signals
- Machine Learning based stress prediction
- Probability scores for every stress class
- REST API built with FastAPI
- Interactive web interface using Streamlit
- Cloud deployment using Render

---

## 📊 Dataset

The model is trained using the **WESAD (Wearable Stress and Affect Detection)** dataset.

The application expects CSV files containing the following physiological signals:

- ECG (Electrocardiogram)
- EDA (Electrodermal Activity)
- Respiration
- Temperature

For every uploaded sample the application automatically computes statistical features before prediction.

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- Feature extraction
- Feature scaling using StandardScaler
- Class balancing using SMOTE

### Feature Extraction

Five statistical features are extracted from every physiological signal.

For every signal:

- Mean
- Standard Deviation
- Minimum
- Maximum
- Median

Total Features:

4 signals × 5 statistics = **20 Features**

---

## 🤖 Model

Random Forest Classifier

Hyperparameter tuning performed using **GridSearchCV**

Best Parameters

```text
max_depth = 20
max_features = sqrt
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 100
```

Best Cross Validation Accuracy

```text
97.20%
```

---

## 📈 Prediction Classes

The model predicts one of the following states:

| Class | Meaning |
|--------|---------|
| Baseline | Relaxed State |
| Stress | Stress Detected |
| Amusement | Positive Emotional State |

The application also displays prediction probabilities for every class.

---

## 🏗️ Project Architecture

```
User
 │
 ▼
Streamlit Frontend
 │
 ▼
FastAPI REST API
 │
 ▼
Feature Scaling
 │
 ▼
Random Forest Model
 │
 ▼
Prediction
 │
 ▼
Results returned to Streamlit
```

---

## 📁 Project Structure

```
stress-detection-api/

│── app.py                 # Streamlit frontend
│── main.py                # FastAPI backend
│── requirements.txt
│── rf_smote_model.pkl     # Trained Random Forest model
│── scaler.pkl             # StandardScaler
│── render.yaml
│── runtime.txt
│── start.sh
│── test_api.py
│── stress_api/
│── .gitignore
```

---

## 🛠 Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- FastAPI
- Streamlit
- Uvicorn
- Requests
- Pydantic
- Render

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/stress-detection-api.git

cd stress-detection-api
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running FastAPI

```bash
uvicorn main:app --reload
```

API Documentation

```
http://127.0.0.1:8000/docs
```

---

## ▶ Running Streamlit

```bash
streamlit run app.py
```

---

## 🌐 REST API

### Health Check

```
GET /
```

Returns API status.

---

### Model Information

```
GET /info
```

Returns model details including number of features and supported classes.

---

### Predict Stress

```
POST /predict
```

Input

```json
{
  "features":[
    ...
    20 feature values
  ]
}
```

Response

```json
{
  "predicted_class":2,
  "label":"Stress",
  "probabilities":{
      "Baseline":0.03,
      "Stress":0.95,
      "Amusement":0.02
  }
}
```

---

## 📊 Application Workflow

1. Upload physiological signal CSV.
2. Extract statistical features.
3. Scale the extracted features.
4. Send features to FastAPI.
5. Random Forest predicts stress level.
6. Display prediction probabilities.
7. Visualize uploaded physiological signals.

---

## 💻 Deployment

Backend deployed using **Render**

Frontend developed using **Streamlit**

Application served using **Uvicorn ASGI Server**

---

## 🔮 Future Improvements

- Deep Learning based stress detection
- Real-time wearable sensor integration
- Explainable AI using SHAP
- Authentication system
- Docker containerization
- Continuous deployment pipeline

---

## 👨‍💻 Author

**Hemanth Kumar**

B.Tech Electronics and Communication Engineering

National Institute of Technology Warangal

---

## 📄 License

This project is intended for educational and research purposes.
