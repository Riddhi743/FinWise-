# =============================
# Loan Approval Model - XGBoost Only
# =============================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ------------------------------
# 1️⃣ Load dataset
# ------------------------------
df = pd.read_csv(r"c:\Users\riddh\OneDrive\Desktop\AI project\loan_data_.csv")

# Drop missing values
df.dropna(inplace=True)

# ------------------------------
# 2️⃣ Define target variable
# ------------------------------
target = 'loan_status'  # 0 = Not approved, 1 = Approved

# Convert Y/N to numeric if necessary
df[target] = df[target].replace({'Y': 1, 'N': 0}).astype(int)

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# ------------------------------
# 3️⃣ Encode categorical variables
# ------------------------------
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ------------------------------
# 4️⃣ Split and scale
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# 5️⃣ Handle class imbalance
# ------------------------------
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# ------------------------------
# 6️⃣ Train XGBoost model
# ------------------------------
print("\n🚀 Training XGBoost model...\n")

xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train, y_train)
preds = xgb_model.predict(X_test)

# ------------------------------
# 7️⃣ Evaluate performance
# ------------------------------
acc = accuracy_score(y_test, preds)
print(f"✅ XGBoost Accuracy: {acc:.4f}\n")

print("Classification Report:\n", classification_report(y_test, preds))

model_path = r"c:\Users\riddh\OneDrive\Desktop\AI project\model_loan_prediction1.pkl"
joblib.dump(xgb_model, model_path)
print(f"✅ XGBoost model saved as: {model_path}")
