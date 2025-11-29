import streamlit as st
import joblib
import numpy as np
import math
import pandas as pd
import json

#path
PROFILE_PATH = r"C:\Users\riddh\OneDrive\Desktop\AI project\profile.json"
FINANCE_PATH = r"C:\Users\riddh\OneDrive\Desktop\AI project\finance_data.csv"
model_path = "model_loan_prediction1.pkl"

def show_loan_tab():
    st.title("🏦 Loan Approval & EMI Analysis")

    # --- Load Trained Model ---
    try:
        model = joblib.load(model_path)
        feature_names = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                         'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                         'credit_score', 'person_education_Bachelor', 'person_education_Doctorate',
                         'person_education_High School', 'person_education_Master',
                         'person_home_ownership_OTHER', 'person_home_ownership_OWN',
                         'person_home_ownership_RENT', 'loan_intent_EDUCATION',
                         'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                         'loan_intent_PERSONAL', 'loan_intent_VENTURE',
                         'previous_loan_defaults_on_file_Yes'

                         ]
    except:
        st.error("⚠️ Loan model not found! Please train and save 'model_loan_prediction1.pkl'.")
        st.stop()
    

    # --- Load Profile Data Safely ---
    try:
        with open(PROFILE_PATH, "r") as f:
            profile = json.load(f)

        # Extract values safely
        income = float(profile.get("income", 0))
        savings = float(profile.get("savings_goal", 0))

    except Exception as e:
        st.warning(f"⚠️ Could not load profile.json properly: {e}")
        st.info("Using default values (income=0, savings=0). Please check your profile file.")

     
    #Display existing financial info
    st.write(f"**Monthly Income:** ₹{income:,.2f}")
    st.write(f"**Current Savings:** ₹{savings:,.2f}")

    st.markdown("---")


    # ---  taking user inputs ---
    st.subheader("📋 Enter Loan & Financial Details")

    col1, col2 = st.columns(2)
    with col1:
        loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, step=5000, value=200000)
        interest_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=25.0, value=10.0)
        tenure_years = st.slider("Loan Tenure (years)", 1, 30, 5)
        existing_loans = st.number_input("Number of Existing Loans", min_value=0, max_value=10, value=1)
        age = st.slider("Age", 18, 65, 30)
        region = st.selectbox("Region", ["Urban", "Semi-Urban", "Rural"], key="region")
        financial_personality = st.selectbox(
            "Financial Personality",
            ["Saver", "Spender", "Balanced"],key="financial_personality"
        )
    with col2:
        credit_score = st.slider("Credit Score", 300, 900, 700)
        employment_status = st.selectbox("Employment Status", ["Unemployed", "Self-employed", "Employed"])
        loan_type = st.selectbox("Loan Type", ["Personal", "Home", "Car", "Education", "Business"],key="loan_type")
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        education_level = st.selectbox("Education Level", ["High School", "Graduate", "Post Graduate", "PhD"],key="education_level")
        gender = st.radio("Gender", ["Male", "Female"])
        marital_status = st.radio("Marital Status", ["Single", "Married"])
        
 

    # --- EMI Calculation ---
    r = (interest_rate / 12) / 100
    n = tenure_years * 12
    if r > 0:
        emi = loan_amount * r * ((1 + r)**n) / ((1 + r)**n - 1)
    else:
        emi = loan_amount / n

    total_payment = emi * n
    total_interest = total_payment - loan_amount

    
    # --- Prediction Preparation ---
    if st.button("🔮 Predict Loan Approval Likelihood"):
        try:
            # --- Map categorical fields ---
            emp_map = {"Unemployed": 0, "Self-employed": 1, "Employed": 2}
            edu_map = {"High School": 0, "Graduate": 1, "Post Graduate": 2, "PhD": 3}
            mar_map = {"Single": 0, "Married": 1}
            gen_map = {"Male": 1, "Female": 0}
            reg_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
            loan_map = {"Personal": 0, "Home": 1, "Car": 2, "Education": 3, "Business": 4}
            fin_map = {"Saver": 1, "Spender": 2, "Balanced": 3}
            
            # --- Build input feature array (must match model columns) ---
            input_features = np.array([[
                age,                    
                income,                 
                5,                      # person_emp_exp (dummy experience)
                loan_amount,            
                interest_rate,           
                loan_amount / income,    # loan_percent_income
                5,                       # cb_person_cred_hist_length (dummy)
                credit_score,            
                1 if education_level == "Graduate" else 0,        # person_education_Bachelor
                1 if education_level == "PhD" else 0,             # person_education_Doctorate
                1 if education_level == "High School" else 0,     # person_education_High School
                1 if education_level == "Post Graduate" else 0,   # person_education_Master
                0,                                                # person_home_ownership_OTHER
                1,                                                # person_home_ownership_OWN
                0,                                                # person_home_ownership_RENT
                1 if loan_type == "Education" else 0,             # loan_intent_EDUCATION
                0,                                                # loan_intent_HOMEIMPROVEMENT
                0,                                                # loan_intent_MEDICAL
                1 if loan_type == "Personal" else 0,              # loan_intent_PERSONAL
                0,                                                # loan_intent_VENTURE
                1 if existing_loans > 0 else 0                    # previous_loan_defaults_on_file_Yes
            ]])
            

            # ✅ Model prediction
            X_input = pd.DataFrame(input_features, columns=feature_names)
            prob_approve = model.predict_proba(X_input)[0][0] * 100

            #--outputs---
            st.success(f"✅ Loan Approval Likelihood: {prob_approve:.2f}%")
            st.write(f"**EMI per month:** ₹{emi:,.2f}")
            st.write(f"**Total Interest Payable:** ₹{total_interest:,.2f}")
            st.write(f"**Total Repayment:** ₹{total_payment:,.2f}")

            if emi > 0.4 * income:
                st.warning("⚠️ Your EMI exceeds 40% of your income. Loan approval chances may reduce.")
    

            # --- Suggestions ---
            st.markdown("### 💡 Suggestions to Improve Chances:")
            tips = []
            if credit_score < 650:
                tips.append("Improve your credit score above 700.")
            if emi > 0.4 * income:
                tips.append("Reduce the loan amount or increase tenure to lower EMI.")
            if savings < 0.1 * loan_amount:
                tips.append("Increase your savings to at least 10% of the loan amount.")
            if existing_loans>1:
                tips.append("Try closing one of your existing loans first.")
            
            if tips:
                for t in tips:
                    st.write(f"- {t}")
            else:
                st.info("Your profile looks strong! 👍")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

