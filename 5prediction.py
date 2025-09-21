# Step 5: Interactive Dashboard (Fixed)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load cleaned dataset
file_path = r"C:\Users\Admin\Desktop\DA Python\Data\WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(file_path)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df["SeniorCitizen"] = df["SeniorCitizen"].map({0:"No",1:"Yes"})

# Encode categorical variables
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove("customerID")
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train Logistic Regression model
X = df.drop(["customerID","Churn"], axis=1)
y = df["Churn"]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Streamlit App
st.title("📊 Telecom Customer Churn Dashboard")

# Sidebar for selecting feature to visualize
feature = st.sidebar.selectbox("Select Feature to Visualize", df.columns[1:])  # skip customerID

# Plot selected feature
st.write(f"### Distribution of {feature}")
plt.figure(figsize=(8,4))
sns.countplot(x=df[feature])
st.pyplot(plt.gcf())
plt.clf()

# Predict Churn for a new customer
st.write("### Predict Churn for a New Customer")

# Create input sliders and dropdowns
tenure = st.slider("Tenure (months)", int(df["tenure"].min()), int(df["tenure"].max()), 12)
monthly = st.slider("Monthly Charges", int(df["MonthlyCharges"].min()), int(df["MonthlyCharges"].max()), 50)
total = st.slider("Total Charges", int(df["TotalCharges"].min()), int(df["TotalCharges"].max()), 500)

# Use LabelEncoder classes_ to populate dropdowns (safe for transform)
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
senior = st.selectbox("Senior Citizen", label_encoders["SeniorCitizen"].classes_)
contract = st.selectbox("Contract Type", label_encoders["Contract"].classes_)  # Use classes_ from encoder


# Encode inputs using label_encoders
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "gender": label_encoders["gender"].transform([gender])[0],
    "SeniorCitizen": label_encoders["SeniorCitizen"].transform([senior])[0],
    "Contract": label_encoders["Contract"].transform([contract])[0]

}

input_df = pd.DataFrame([input_dict])

# Create a full row for prediction using user input and defaults
input_dict_full = {}

for col in X.columns:  # all columns used during training
    if col in input_dict:  # user provided
        input_dict_full[col] = input_dict[col]
    else:  # use default value (most frequent)
        input_dict_full[col] = df[col].mode()[0]

input_df_full = pd.DataFrame([input_dict_full])

# Predict using full feature set
pred = model.predict(input_df_full)[0]
pred_prob = model.predict_proba(input_df_full)[0][1]

st.write(f"Predicted Churn: **{label_encoders['Churn'].inverse_transform([pred])[0]}**")
st.write(f"Probability of Churn: **{pred_prob*100:.2f}%**")