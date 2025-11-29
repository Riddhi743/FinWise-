import streamlit as st
import numpy as np
import joblib

def show_fraud_detection_tab():

    st.header("🚨BE AWARE OF FRAUD")
    
    #select box
    fraud_type = st.selectbox(
        "Select Type of Fraud to Check:",
        [
            "Select here",
            "Transaction Fraud",
            "Profile Manipulation Fraud"
        ]
    )
    # ----transacation fraud-----
    if fraud_type == "Transaction Fraud":

        # Load trained model + encoders
        model = joblib.load("fraud_detection_model.pkl")
        encoders = joblib.load("fraud_label_encoders.pkl")

        st.subheader("💳 Transaction Fraud Detection")

        # --- User Inputs ---
        age = st.number_input("Customer Age:", min_value=18, max_value=90)
        gender = st.selectbox("Gender:", ("M", "F"))
        amount = st.number_input("Transaction Amount (₹):", min_value=1)
        location = st.selectbox("Location Match:", ["same", "different"])

        # --- Predict Button ---
        if st.button("Check Fraud"):
            # Encode categorical inputs
            gender_enc = int(encoders["gender"].transform([gender])[0])
            location_enc = int(encoders["location"].transform([location])[0])

            # Prepare input for model
            input_data = np.array([[int(age), gender_enc, float(amount), location_enc]])

            # Predict
            prediction = model.predict(input_data)[0]
            fraud_prob = model.predict_proba(input_data)[0][1] * 100

            # Show result
            if prediction == 1:
                st.error(f"⚠️ Possible Fraud Detected! | Confidence: {fraud_prob:.2f}%")
            else:
                st.success(f"✔ Transaction appears safe | Fraud Chance: {fraud_prob:.2f}%")

    # fake account
    elif fraud_type == "Profile Manipulation Fraud":
        st.subheader("🧩 Profile Fraud Detection")
        old_income = st.number_input("Previous Income Value (₹):", min_value=0)
        new_income = st.number_input("Updated Income Value (₹):", min_value=0)
        multiple_profiles = st.selectbox("Is this info used in multiple profiles?", ["No", "Yes"])

        if st.button("Check Profile Fraud"):
            if abs(new_income - old_income) > 0.8 * old_income or multiple_profiles == "Yes":
                st.error("⚠️ Profile Data Change Looks Suspicious!")
                if abs(new_income - old_income) > 0.8 * old_income:
                    st.markdown("- Sudden large change in income value detected.")
                if multiple_profiles == "Yes":
                    st.markdown("- Profile details appear duplicated across accounts.")
            else:
                st.success("✅ Profile looks consistent.")
                st.markdown("- Profile data changes appear reasonable.")


