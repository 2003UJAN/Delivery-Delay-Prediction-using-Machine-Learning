import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Delivery Delay Predictor")

st.title("📦 Delivery Delay Prediction App")

# Safe model loading
model_path = "delay_predictor.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Upload `delay_predictor.pkl` to the app directory.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error("❌ Failed to load model. This usually happens due to Python version mismatch.")
    st.code(str(e))
    st.info("""
    **Fix:** Retrain your model and save it using the SAME Python version used by this Streamlit environment.
    """)
    st.stop()

# UI form
st.sidebar.header("Enter Shipment Details")

origin = st.sidebar.selectbox("Origin", ["Delhi", "Mumbai", "Chennai", "Bangalore", "Kolkata"])
destination = st.sidebar.selectbox("Destination", ["New York", "London", "Dubai", "Singapore", "Frankfurt"])

distance_km = st.sidebar.number_input("Distance (km)", 100, 10000, 1500)
package_weight_kg = st.sidebar.number_input("Package Weight (kg)", 0.1, 50.0, 5.0)

weather_severity = st.sidebar.slider("Weather Severity (0–4)", 0, 4, 1)
traffic_level = st.sidebar.slider("Traffic Level (1–4)", 1, 4, 2)
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
