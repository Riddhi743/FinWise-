import streamlit as st
import pandas as pd
import joblib
import os
import json

def show_predict_saving_tab():
    st.title("💰 Predict Your Next Month Savings")

    # load trained model
    model_path = r"C:\Users\riddh\OneDrive\Desktop\AI project\next_month_savings_model.pkl"
    if os.path.exists(model_path):
        rf_model = joblib.load(model_path)
    else:
        st.warning("Model not found. Please train the model first.")
        st.stop()

    # load profile JSON file -- for income,saving,loan data
    profile_path = "profile.json"
    if os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            profile = json.load(f)
    else:
        st.warning("Profile not found")
        st.stop()
    
    # load finance CSV file - for expense data
    try:
        expenses_df = pd.read_csv("finance_data.csv") 
    except:
        st.warning("Finance CSV not found.")
        st.stop()

    # --------getting data from profile ----------
    income = profile["income"]
    savings = profile["savings_goal"]
    loans = profile["loans"]

    # calculating monthly expenses from finance file
    last_30_rows = expenses_df.tail(30)
    monthly_expense = last_30_rows["Amount Spent"].sum()

    # ----showing data-----
    st.text(f"Income: ₹{profile['income']:,}")
    st.text(f"Savings Goal: ₹{profile['savings_goal']:,}")
    st.text(f"Loans: ₹{profile['loans']:,}")
    st.text(f"Monthly Expenses (last 30 entries): ₹{monthly_expense:,}")
    
    # dropdown for personality 
    personality = st.selectbox("Financial Personality", ["Saver", "Spender", "Balanced"])

    # map personality to dummy columns
    personal_balanced = 1 if personality == "Balanced" else 0
    personal_spender = 1 if personality == "Spender" else 0
    personal_saver = 1 if personality == "Saver" else 0

    # Input dataframe
    input_data = pd.DataFrame({
        "income": [income],
        "expenses": [monthly_expense],
        "savings": [savings],
        "loans": [loans],
        "financial_personality_Balanced": [personal_balanced],
        "financial_personality_Spender": [personal_spender],
        "financial_personality_Saver": [personal_saver]
    })
    
    # Ensure column order matches training
    input_data = input_data[rf_model.feature_names_in_]

    # ---------- final prediction ----------
    if st.button("Predict Next Month Savings"):
        predicted_savings = rf_model.predict(input_data)[0]
        st.success(f"💡 Predicted Savings: ₹{predicted_savings:,.0f}")
        additional_savings = max(predicted_savings - savings, 0)
        st.info(f"Suggested additional savings to reach predicted amount: ₹{additional_savings:,.0f}")
