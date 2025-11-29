import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------
# 1️⃣ LOAD CLEAN DATASET
# -------------------------
df = pd.read_csv("transaction_dataset.csv")

# -------------------------
# 2️⃣ DROP UNUSED COLUMNS
# -------------------------
df = df.drop(columns=["customer", "merchant"])

# -------------------------
# 3️⃣ ENCODE CATEGORICAL COLUMNS
# -------------------------
cat_cols = ["gender", "location"]
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -------------------------
# 4️⃣ SELECT FEATURES & TARGET
# -------------------------
X = df[["age", "gender", "amount", "location"]]
y = df["fraud"]

# -------------------------
# 5️⃣ TRAIN–TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -------------------------
# 6️⃣ TRAIN FRAUD MODEL
# -------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# 7️⃣ EVALUATE MODEL
# -------------------------
y_pred = model.predict(X_test)

print("✔ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------
# 8️⃣ SAVE MODEL & ENCODERS
# -------------------------
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(encoders, "fraud_label_encoders.pkl")

print("\n🎯 Fraud Detection Model Created Successfully!")
