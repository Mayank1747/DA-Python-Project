# Step 2: Data Cleaning & Transformation

import pandas as pd

# Correct file path (CSV is inside "Data" folder)
file_path = r"C:\Users\Admin\Desktop\DA Python\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)

print("\n--- Before Cleaning ---")
print(df.dtypes)

# 1. Convert TotalCharges to numeric (it's object right now)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 2. Check again for missing values after conversion
print("\nMissing values after converting TotalCharges:")
print(df.isnull().sum())

# 3. Drop rows where TotalCharges is missing (very few cases)
df = df.dropna(subset=["TotalCharges"])

# 4. Reset index after dropping
df.reset_index(drop=True, inplace=True)

# 5. Encode SeniorCitizen (0 = No, 1 = Yes)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

# 6. Confirm transformations
print("\n--- After Cleaning ---")
print(df.dtypes)
print("\nPreview of cleaned dataset:")
print(df.head())