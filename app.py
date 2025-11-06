# =====================================
# Finlytix - Credit Risk Prediction Web App (Final Version)
# =====================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# File Paths
# -------------------------------
MODEL_FILE = "final_model.pkl"
PIPELINE_FILE = "final_pipeline.pkl"
TEST_FILE = "testing-1.csv"

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Finlytix - Credit Risk Prediction", layout="centered")
st.title("💰 Finlytix: AI-Powered Credit Risk Prediction")
st.caption("Predicting financial risk with explainable AI | EY Techathon 6.0")

# -------------------------------
# Load Model, Pipeline, and Dataset
# -------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    test_data = pd.read_csv(TEST_FILE)
    return model, pipeline, test_data

model, pipeline, test_data = load_resources()

# Display basic info
st.sidebar.header("📂 Data Overview")
st.sidebar.write(f"Total Customers in Database: **{len(test_data)}**")
st.sidebar.write(f"Columns: {', '.join(test_data.columns)}")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("🔍 Check Customer Credit Risk")
customer_id = st.text_input("Enter Customer ID (as per testing-1.csv):")

if customer_id:
    try:
        customer_id = int(customer_id)
        if customer_id not in test_data["id"].values:
            st.error("❌ Invalid ID. Please enter a valid Customer ID from the dataset.")
        else:
            # Extract record
            customer = test_data.loc[test_data["id"] == customer_id].drop("id", axis=1)

            # Preprocess features using saved pipeline
            customer_prepared = pipeline.transform(customer)

            # Predict probabilities
            prob_default = model.predict_proba(customer_prepared)[0][1]
            prob_repay = 1 - prob_default

            # Display results
            st.success(f"✅ Prediction for Customer ID: {customer_id}")
            st.metric("Probability of Default", f"{prob_default:.2%}")
            st.metric("Probability of Repayment", f"{prob_repay:.2%}")

            # Risk interpretation
            if prob_default < 0.4:
                st.markdown("🟢 **Low Risk:** Customer is likely to repay debt.")
            elif prob_default < 0.7:
                st.markdown("🟠 **Medium Risk:** Customer may need monitoring.")
            else:
                st.markdown("🔴 **High Risk:** Customer is likely to default.")

            # Optional feature details
            with st.expander("View Customer Data"):
                st.write(customer)

    except ValueError:
        st.error("⚠️ Please enter a valid numeric Customer ID.")

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("Developed by Team Finlytix | EY Techathon 6.0 | Powered by XGBoost & Streamlit")