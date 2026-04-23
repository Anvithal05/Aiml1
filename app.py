import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data import generate_data
from model import train_models, train_fraud_model

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Insurance AI System", layout="wide")

st.title("🚀 Insurance Claim Prediction ")

# =========================
# LOAD DATA
# =========================
df = generate_data()

# Train models
model, results = train_models(df)
fraud_model = train_fraud_model(df)

# =========================
# INPUT UI
# =========================
st.subheader("🔮 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    vehicle_age = st.number_input("Vehicle Age", 0, 15, 0)
    premium = st.number_input("Premium", 5000, 100000, 50000)

with col2:
    accidents = st.number_input("Accidents", 0, 10, 0)
    vehicle_type = st.selectbox("Vehicle Type", ["car", "bike", "truck"])
    region = st.selectbox("Region", ["urban", "rural"])

# Encoding
vehicle_map = {"car": 0, "bike": 1, "truck": 2}
region_map = {"urban": 0, "rural": 1}

input_df = pd.DataFrame([{
    "age": age,
    "vehicle_age": vehicle_age,
    "premium": premium,
    "accidents": accidents,
    "vehicle_type": vehicle_map[vehicle_type],
    "region": region_map[region]
}])

# =========================
# PREDICTION
# =========================
if st.button("Predict Claim Amount"):

    ml_pred = model.predict(input_df)[0]

    # Business logic
    base = 2 * premium
    reduction = 1 - (vehicle_age * 0.04) - (accidents * 0.08)
    reduction = max(0.3, min(1, reduction))

    final_pred = base * reduction
    prediction = (0.7 * final_pred) + (0.3 * ml_pred)
    prediction = max(prediction, 5000)

    st.success(f"💰 Predicted Claim Amount: ₹ {int(prediction)}")

    # =========================
    # FRAUD DETECTION
    # =========================
    fraud_input = input_df[["age", "vehicle_age", "premium", "accidents"]]
    fraud_pred = fraud_model.predict(fraud_input)[0]

    if fraud_pred == -1:
        st.error("🚨 Fraud Alert: Suspicious Claim Detected!")
    else:
        st.info("✅ Claim looks normal")

# =========================
# VISUALIZATIONS
# =========================
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

# Graph 1: Premium vs Claim
with col1:
    fig, ax = plt.subplots()
    ax.scatter(df["premium"], df["claim_amount"])
    ax.set_xlabel("Premium")
    ax.set_ylabel("Claim Amount")
    ax.set_title("Premium vs Claim Amount")
    st.pyplot(fig)

# Graph 2: Accidents vs Claim
with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["accidents"], df["claim_amount"])
    ax2.set_xlabel("Accidents")
    ax2.set_ylabel("Claim Amount")
    ax2.set_title("Accidents Impact on Claim")
    st.pyplot(fig2)