import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")
st.title("📦 Delivery Delay Prediction App")

MODEL_PATH = "delay_predictor.pkl"


# -------------------------
# TRAIN MODEL IF NOT EXISTS
# -------------------------
def train_and_save_model():
    st.info("⏳ Training model for the first time...")

    np.random.seed(42)
    n = 5000

    origins = ["Delhi", "Mumbai", "Chennai", "Bangalore", "Kolkata"]
    destinations = ["New York", "London", "Dubai", "Singapore", "Frankfurt"]

    data = {
        "origin": np.random.choice(origins, n),
        "destination": np.random.choice(destinations, n),
        "distance_km": np.random.randint(500, 9000, n),
        "package_weight_kg": np.round(np.random.uniform(0.1, 25.0, n), 2),
        "weather_severity": np.random.randint(0, 5, n),
        "traffic_level": np.random.randint(1, 4, n),
        "dispatch_hour": np.random.randint(0, 23, n),
    }

    df = pd.DataFrame(data)

    df["delay_score"] = (
        0.02 * df["distance_km"] +
        1.5 * df["weather_severity"] +
        2.0 * df["traffic_level"] +
        np.random.normal(0, 5, n)
    )

    df["is_delayed"] = (df["delay_score"] > df["delay_score"].median()).astype(int)

    X = df.drop(["is_delayed", "delay_score"], axis=1)
    y = df["is_delayed"]

    categorical_cols = ["origin", "destination"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    st.success("✅ Model trained and saved successfully!")


# -------------------------
# LOAD OR TRAIN MODEL
# -------------------------
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

model = joblib.load(MODEL_PATH)


# -------------------------
# UI INPUT FORM
# -------------------------
st.sidebar.header("Enter Shipment Details")

origin = st.sidebar.selectbox("Origin", ["Delhi", "Mumbai", "Chennai", "Bangalore", "Kolkata"])
destination = st.sidebar.selectbox("Destination", ["New York", "London", "Dubai", "Singapore", "Frankfurt"])

distance_km = st.sidebar.number_input("Distance (km)", 100, 10000, 1500)
package_weight_kg = st.sidebar.number_input("Package Weight (kg)", 0.1, 50.0, 5.0)

weather_severity = st.sidebar.slider("Weather Severity (0 = clear, 4 = storm)", 0, 4, 1)
traffic_level = st.sidebar.slider("Traffic Level (1 = low, 4 = heavy)", 1, 4, 2)

dispatch_hour = st.sidebar.slider("Dispatch Hour (0–23)", 0, 23, 10)

input_data = pd.DataFrame({
    "origin": [origin],
    "destination": [destination],
    "distance_km": [distance_km],
    "package_weight_kg": [package_weight_kg],
    "weather_severity": [weather_severity],
    "traffic_level": [traffic_level],
    "dispatch_hour": [dispatch_hour]
})

st.subheader("Shipment Details")
st.dataframe(input_data)

if st.button("Predict Delay"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.error("🚨 Shipment likely to be DELAYED")
    else:
        st.success("✅ Shipment likely to be ON-TIME")
