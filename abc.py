import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load synthetic dataset
df = pd.read_csv("synthetic_finance_data.csv")  # Your CSV path
print("✅ Dataset loaded:", df.shape)

# One-hot encode personality
df = pd.get_dummies(df, columns=["financial_personality"], drop_first=True)

# Features and target
X = df.drop("next_month_savings", axis=1)
y = df["next_month_savings"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)