# 📦 Delivery Delay Prediction using Machine Learning  
### End-to-End Logistics ML Project (FedEx / DHL / UPS)

This project predicts whether a shipment will be **delayed** based on logistics features such as distance, weather severity, traffic, package weight, and dispatch time.  
It demonstrates skills relevant for **Data Scientist / Decision Scientist / Data Analyst / AI–ML Intern** roles in supply chain & logistics.

---

## 🚀 Tech Stack

- **Python**, **Pandas**, **NumPy**
- **Scikit-Learn** (Random Forest, pipelines, preprocessing)
- **Streamlit** (deployment)
- **Google Colab** (model training)
- **GitHub** (version control)
- **Synthetic dataset** (no real data used)

---

## ⭐ Project Features

✔ Synthetic logistics dataset (5000 shipments)  
✔ ML pipeline (preprocessing + Random Forest)  
✔ Binary classification: *Delayed* vs *On-Time*  
✔ Exported model (`delay_predictor.pkl`)  
✔ Streamlit app for real-time prediction  
✔ Clean notebook + organized repo  

---

## 📁 Dataset Description

| Feature | Description |
|--------|-------------|
| origin | Starting city |
| destination | Destination city |
| distance_km | Distance in KM |
| package_weight_kg | Weight of shipment |
| weather_severity | 0 = clear, 4 = storm |
| traffic_level | 1 = low, 4 = heavy |
| dispatch_hour | Hour of dispatch |
| is_delayed | Target |

---

## 🧠 Model Overview

- Random Forest Classifier  
- One-hot encoding for categorical features  
- Train-test split (80/20)  
- Evaluation metrics: accuracy & classification report  

Typical performance:

- **Accuracy:** 97%  
- Good recall on delayed shipments  

---

## ▶️ Running the Project

### **1. Install dependencies**
```bash
pip install -r requirements.txt
