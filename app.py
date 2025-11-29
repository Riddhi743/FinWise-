import streamlit as st
import pandas as pd
import os
import json
from datetime import date, timedelta

# linking other files
from stock_predictor import stock_market_simulator
from predict_savings import show_predict_saving_tab
from loan_approval import show_loan_tab
from fraud_detection import show_fraud_detection_tab

# --- File paths ---
PROFILE_FILE = "profile.json"
DATA_FILE = "finance_data.csv"


# ---------- Profile Handling ----------
def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_profile(profile):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=4)

# ---------- Expense Data Handling ----------
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        # Create empty structure if missing
        return pd.DataFrame(columns=["Date", "Category", "Amount Spent", "Note", "Budget", "Status"])

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def init_data():
    """Ensure CSV file exists"""
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=["Date", "Category", "Amount Spent", "Note", "Budget", "Status"])
        df.to_csv(DATA_FILE, index=False)
    else:
        df = pd.read_csv(DATA_FILE)
    return df

def add_expense(category, amount, note, profile):
    df = load_data()
    # Ensure column names exist
    required_cols = ["Date", "Category", "Amount Spent", "Note", "Budget", "Status"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    today_str = str(date.today())
    daily_budget = (profile["income"] - profile["loans"] - profile["savings_goal"]) / 30

def add_expense(category, amount, note, profile):
    df = load_data()
    today_str = str(date.today())
    daily_budget = (profile["income"] - profile["loans"] - profile["savings_goal"]) / 30

    # Compute total spent today (before adding new one)
    total_today = df[df["Date"] == today_str]["Amount Spent"].sum() + amount

    # Determine status based on *new total*
    status = "Under" if total_today <= daily_budget else "Over"

    # Add new expense row
    new_row = pd.DataFrame(
        [[today_str, category, amount, note, daily_budget, status]],
        columns=df.columns
    )
    df = pd.concat([df, new_row], ignore_index=True)

    save_data(df)
    return df

# ---- recomedation----
def get_ai_recommendation(df):
    last_30 = df.tail(30)
    overspend_days = len(last_30[last_30["Status"] == "Over"])
    underspend_days = len(last_30[last_30["Status"] == "Under"])

    if overspend_days > 10:
        return "⚠️ You are overspending frequently. Review lifestyle and food expenses."
    elif underspend_days > 20:
        return "🎉 Excellent! You're saving well — maybe invest or raise your savings goal."
    else:
        return "💡 Spending is fairly balanced. Keep monitoring and optimizing."

# ---------- Streamlit  ----------
st.set_page_config(page_title="AI Personal Finance Tracker", page_icon="💰", layout="wide")
st.title("💰 AI Personal Finance Tracker")

profile = load_profile()


# -------- Profile Section --------
if not profile:
    st.subheader("🧩 Set up your profile")
    income = st.number_input("Monthly Salary (₹):", min_value=0)
    loans = st.number_input("Monthly Loan Payments (₹):", min_value=0)
    savings_goal = st.number_input("Monthly Savings Goal (₹):", min_value=0)
    if st.button("Save Profile"):
        if income > 0 :
            profile = {
                "income": income,
                "loans": loans,
                "savings_goal": savings_goal,
                    }
            save_profile(profile)
            st.success("Profile saved! Please reload the app.")
        else:
            st.error("Please enter valid values.")
# grey wala part
else:
    # Sidebar Profile
    st.sidebar.header("👤 Profile Info")
    st.sidebar.write(f"**Salary:** ₹{profile['income']}")
    st.sidebar.write(f"**Loans:** ₹{profile['loans']}")
    st.sidebar.write(f"**Savings Goal:** ₹{profile['savings_goal']}")

    # Load data
    df = init_data()

    # -------- Add Expense Section --------
    st.subheader("➕ Add Today's Expense")
    category = st.selectbox("Category", ["Food", "Lifestyle", "Health", "Education", "Bank", "Others"])
    amount = st.number_input("Amount Spent (₹):", min_value=0)
    note = st.text_input("Notes (optional):")

    #button for add expense
    if st.button("Add Expense"):
        if amount > 0:
            df = add_expense(category, amount, note, profile)
            st.success("Expense added to database!")
            st.rerun()  # reload data + refresh UI
        else:
            st.error("Enter an amount greater than 0.")



    # -------- Calendar --------

    st.subheader("📅 Calendar View: Spending Pattern (Last 30 Days)")
    df = load_data()

    if not df.empty:

        df["Date"] = pd.to_datetime(df["Date"]).dt.date 

    # Add emojis to status
    def add_status_emoji(status):
        if isinstance(status, str):
            if status.lower() == "under":
                return "🟢 Under Budget"
            elif status.lower() == "over":
                return "🔴 Over Budget"
        return "⚪ Unknown"

    df["Status"] = df["Status"].apply(add_status_emoji)

    # Display selected columns in calender
    display_cols = ["Date", "Category", "Amount Spent", "Note", "Status"]
    df_last30 = df.sort_values("Date", ascending=False).head(30)[display_cols]

    # Color formatting for under n over
    def color_status(val):
        if "Under" in val:
            return "color: green; font-weight: bold"
        elif "Over" in val:
            return "color: red"
        else:
            return "color: gray"

    st.dataframe(df_last30.style.applymap(color_status, subset=["Status"]))


    # -------- tabs --------
    
    st.markdown("---")
    st.subheader("🤖 AI Assistant & Smart Insights")
    st.info(get_ai_recommendation(df))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([

        "💡 Lifestyle & Food Tips",
        "💰 Wealth Builder", 
        "📊 Deposits", 
        "🏦 Loan Tracking", 
        "🚨 Risk-Radar"
        
    ])
    #tab with tips and interaction
    with tab1:
        tips = [
        "🚶 Walk or cycle instead of using cabs → Save ₹200–₹500/week",
        "🥗 Cook at home instead of ordering food → Save ₹500–₹1000/week",
        "🛍️ Buy second-hand or discounted items → Save up to 30%",
        "☕ Make coffee at home → Save ₹300/week"
        ]
        st.markdown("💡 Lifestyle")

        for tip in tips:
            st.write("-", tip)

        # Mini Challenges & Gamification
        
        st.markdown("### 🎯 Weekly Challenges")
        challenge1 = st.checkbox("💰 Save ₹500 this week")
        challenge2 = st.checkbox("🚫 No online shopping for 3 days")

        st.markdown("### 🏅 Your Rewards & Badges")
        badges = []
        weekly_savings = 2000  # Example value
        if weekly_savings >= 2000:
            badges.append("💎 Savings Guru")
        if challenge1:
            badges.append("🎯 Challenge Completed: ₹500 saved")
        if challenge2:
            badges.append("🚫 Online-Free Streak: 3 days")

        if badges:
            for b in badges:
                st.success(b)
        else:
            st.info("Complete challenges to earn badges!")


    # --- TAB 2: stock portfolio ---
    with tab2:
        stock_market_simulator()

   # --- TAB 3: Savings prediction---
    with tab3:
        if not df.empty:
            avg_daily_spend = df["Amount Spent"].mean()
            monthly_budget = profile["income"] - profile["loans"] - profile["savings_goal"]
            projected_savings = monthly_budget - (avg_daily_spend * 30)
            st.write(f"Projected Savings for this Month: ₹{projected_savings:.2f}")
            if projected_savings > 0:
                st.success("Great job! You're on track to meet your savings goals.")
            else:
                    st.warning("You might exceed your spending target. Try adjusting your daily limits.")
        else:
            st.info("Not enough spending data yet for a forecast.")
        show_predict_saving_tab()

    # --- TAB 4: Loan & EMI Tracker ---
    with tab4:
        show_loan_tab()

    # --- TAB 5: Risk Monitoring & Alerts ---
    with tab5:
        show_fraud_detection_tab()
    
    
    # -------- Reset Button --------
    st.markdown("---")
    if st.button("🗑️ Reset All Data"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        if os.path.exists(PROFILE_FILE):
            os.remove(PROFILE_FILE)
        st.success("All data cleared. Please reload the app.")
