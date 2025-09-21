# Step 1: Load and Explore Dataset

import pandas as pd

# Load dataset
file_path = r"C:\Users\Admin\Desktop\DA Python\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)


# Show first 5 rows
print("Preview of dataset:")
print(df.head())

# Show basic info
print("\nDataset Info:")
print(df.info())

# Show missing values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Show basic statistics
print("\nStatistical Summary:")
print(df.describe())

















