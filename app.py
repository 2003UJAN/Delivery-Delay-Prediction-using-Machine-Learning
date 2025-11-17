import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("delay_predictor.pkl")

st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

st.title("📦 Delivery Delay Prediction App")
st.write("Upload shipment details or fill the form to predict delivery delays.")

# --- Sidebar for form input ---
st.sidebar.header("Enter Shipment Details")

origin = st.sidebar.selectbox("Origin", ["Delhi", "Mumbai", "Chennai", "Bangalore", "Kolkata"])
destination = st.sidebar.selectbox("Destination", ["New York", "London", "Dubai", "Singapore", "Frankfurt"])

distance_km = st.sidebar.number_input("Distance (km)", min_value=100, max_value=10000, value=1500)
package_weight_kg = st.sidebar.number_input("Package Weight (kg)", min_value=0.1, max_value=50.0, value=5.0)

weather_severity = st.sidebar.slider("Weather Severity (0 = clear, 4 = storm)", 0, 4, 1)
traffic_level = st.sidebar.slider("Traffic Level (1 = low, 4 = heavy)", 1, 4, 2)

dispatch_hour = st.sidebar.slider("Dispatch Hour (0–23)", 0, 23, 10)

# Convert to dataframe for model
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

# Prediction button
if st.button("Predict Delay"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("🚨 **Prediction: Shipment is likely to be DELAYED.**")
    else:
        st.success("✅ **Prediction: Shipment will likely be ON TIME.**")


