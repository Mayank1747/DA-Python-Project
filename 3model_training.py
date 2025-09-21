# Step 3: Upgraded EDA & Visualization

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn", palette="Set2")
plt.title("Churn Distribution")
plt.show()

# 2. Correlation Heatmap for numeric columns
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
plt.figure(figsize=(6,5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# 3. Boxplots for numeric features by Churn
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Churn", y=col, data=df, palette="Set3")
    plt.title(f"{col} Distribution by Churn")
    plt.show()

# 4. Countplots for all categorical columns vs Churn
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove("customerID")  # Skip ID column

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=col, hue="Churn", palette="Set1")
    plt.title(f"{col} vs Churn")
    plt.xticks(rotation=45)
    plt.show()