import pandas as pd

df = pd.read_csv("transaction_dataset_.csv")

# Remove ALL single and double quotes from the entire dataset
df = df.replace("'", "", regex=True)
df = df.replace('"', "", regex=True)

# Convert numeric columns
df["age"] = pd.to_numeric(df["age"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["fraud"] = pd.to_numeric(df["fraud"], errors="coerce")

# Drop any bad rows
df = df.dropna()

# Save cleaned dataset
df.to_csv("transaction_dataset.csv", index=False)

print("✔ Dataset cleaned successfully!")
