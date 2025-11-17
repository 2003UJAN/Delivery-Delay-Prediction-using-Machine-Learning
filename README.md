# 📦 Delivery Delay Prediction using Machine Learning  
### An End-to-End Data Science Project for Logistics (FedEx / DHL / UPS)

This project predicts whether a shipment will be **delayed** based on logistics factors such as distance, weather severity, traffic, package weight, and dispatch time.  
It demonstrates skills relevant for **Data Scientist / Decision Scientist / Data Analyst / AI–ML Intern** roles in logistics and supply-chain companies.

---

## 🚀 Tech Stack

- **Python**, **Pandas**, **NumPy**
- **Scikit-Learn** (Random Forest, pipelines, preprocessing)
- **Streamlit** (deployment)
- **Google Colab** (training environment)
- **GitHub** (version control)
- **Synthetic dataset** (no real company data used)

---

## ⭐ Project Features

✔ Synthetic logistics dataset (5000 shipments)  
✔ ML pipeline with preprocessing + Random Forest  
✔ Binary classification: *Delayed* vs. *On-Time*  
✔ Exported model (`delay_predictor.pkl`)  
✔ Streamlit Web App for real-time prediction  
✔ Clean, reproducible code & notebook  

---

## 📁 Dataset Description

The dataset is generated synthetically to simulate real logistics operations.

| Feature | Description |
|--------|-------------|
| origin | Shipment starting city |
| destination | Destination city |
| distance_km | Total distance to be covered |
| package_weight_kg | Weight of package |
| weather_severity | 0 = clear, 4 = storm |
| traffic_level | 1 = low, 4 = heavy |
| dispatch_hour | Hour of dispatch (0–23) |
| is_delayed | Target variable |

---

## 🧠 ML Model Overview

A **Random Forest Classifier** is trained using a scikit-learn preprocessing pipeline:

- One-hot encoding for categorical features
- Numerical features passed through directly
- Train-test split (80/20)
- Evaluation: accuracy + classification report

---

## 📊 Results

The model typically achieves:

- **Accuracy:** 80–88%  
- **High precision** on delayed shipments  
- **Stable performance** due to synthetic noise

(Results may vary slightly per run.)

---

## ▶️ Running the Streamlit App

### **1. Install dependencies**
```bash
pip install -r requirements.txt
---
### **2. Start the app**
```bash
streamlit run app.py
---
### **3. Upload or enter shipment details**

 The app predicts:

    🟢 On-Time Shipment

    🔴 Delayed Shipment
---
### **🖥 Folder Structure**

delivery-delay-prediction/
│
├── app.py
├── delay_predictor.pkl
├── synthetic_logistics_data.csv
├── README.md
└── notebooks/
       └── model_training.ipynb
---
### **📈 Future Improvements**

    Add route-based features (lat-long + geospatial)

    Incorporate weather API for real-time predictions

    Use XGBoost or LightGBM for improved accuracy

    Add SHAP explanations for model interpretability

    Deploy the Streamlit app on Streamlit Cloud
---
### **👨‍💻 Author**

Ujan Pradhan
AI/ML & Data Science Projects
Google Colab • Streamlit • Machine Learning • Optimization • Analytics
