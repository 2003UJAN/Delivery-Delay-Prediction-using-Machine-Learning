import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📦 Delivery Delay Prediction (Auto-Train Model)")

# -------------------------------------------
# STEP 1 — SYNTHETIC DATA GENERATION
# -------------------------------------------
@st.cache_data
def generate_dataset(n=5000):
    np.random.seed(42)

    data = pd.DataFrame({
        "distance_km": np.random.randint(5, 3000, n),
        "package_weight": np.random.uniform(0.1, 30, n),
        "weather": np.random.randint(0, 3, n),     # 0=Clear,1=Moderate,2=Severe
        "traffic": np.random.randint(0, 3, n),     # 0=Low,1=Medium,2=High
        "priority": np.random.randint(0, 2, n),    # 0=Normal,1=Express
    })

    # Define delay based on logic
    data["delayed"] = (
        (data["distance_km"] > 1500)
        | (data["weather"] == 2)
        | (data["traffic"] == 2)
        | ((data["priority"] == 0) & (data["distance_km"] > 800))
    ).astype(int)

    return data


# -------------------------------------------
# STEP 2 — MODEL TRAINING
# -------------------------------------------
@st.cache_resource
def train_model():
    df = generate_dataset()

    X = df.drop("delayed", axis=1)
    y = df["delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc


model, accuracy = train_model()

st.success(f"Model trained successfully! 🎯 Accuracy: **{accuracy*100:.2f}%**")


# -------------------------------------------
# STEP 3 — USER INPUT FOR PREDICTION
# -------------------------------------------
st.header("🔮 Predict Delivery Delay")

distance = st.number_input("Distance (km)", min_value=1, max_value=3000, value=500)
weight = st.number_input("Package Weight (kg)", min_value=0.1, max_value=30.0, value=2.5)

weather = st.selectbox(
    "Weather",
    ["Clear", "Moderate", "Severe"]
)

traffic = st.selectbox(
    "Traffic",
    ["Low", "Medium", "High"]
)

priority = st.selectbox(
    "Priority",
    ["Normal", "Express"]
)

weather_map = {"Clear": 0, "Moderate": 1, "Severe": 2}
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
priority_map = {"Normal": 0, "Express": 1}

input_df = pd.DataFrame([{
    "distance_km": distance,
    "package_weight": weight,
    "weather": weather_map[weather],
    "traffic": traffic_map[traffic],
    "priority": priority_map[priority],
}])


# -------------------------------------------
# STEP 4 — PREDICT
# -------------------------------------------
if st.button("Predict Delay"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("❌ Shipment is likely to be **DELAYED**.")
    else:
        st.success("✅ Shipment is likely to be **ON TIME**.")
