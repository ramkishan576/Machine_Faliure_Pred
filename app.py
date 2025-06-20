import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Set page title and layout
st.set_page_config(page_title="Machine Failure Prediction", layout="wide")

# Title
st.title("Machine Failure Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"machine_failure_cleaned.csv")
    return df

df = load_data()

# Display dataset information
st.header("Dataset Overview")
st.subheader("First 20 Rows")
st.dataframe(df.head(20))

st.subheader("Dataset Info")
buffer = pd.DataFrame(df.dtypes, columns=['Dtype'])
buffer['Null Count'] = df.isnull().sum()
st.dataframe(buffer)

st.subheader("Dataset Summary")
st.dataframe(df.describe())

# Visualize distribution of Machine Failure
st.header("Distribution of Machine Failure")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Machine failure', data=df, ax=ax)
plt.title("Distribution of Machine Failure")
st.pyplot(fig)

# Prepare data
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Display class distribution after SMOTE
st.header("Class Distribution After SMOTE")
st.write(pd.Series(y_train_resampled).value_counts())

# Train and evaluate RandomForestClassifier
st.header("Random Forest Classifier Results")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)

# Display confusion matrix and classification report
st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred_rf))
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred_rf))

# Clean column names
X_train.columns = X_train.columns.str.replace(r"[\[\]<>]", "", regex=True)
X_test.columns = X_test.columns.str.replace(r"[\[\]<>]", "", regex=True)

# Calculate class imbalance ratio
class_imbalance_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

# Train and evaluate XGBoostClassifier
st.header("XGBoost Classifier Results (Custom Threshold 0.6)")
xgb_model = XGBClassifier(scale_pos_weight=class_imbalance_ratio, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_proba > 0.6).astype(int)

# Display confusion matrix and classification report
st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred_custom))
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred_custom))

# Optional: Allow user to input data for prediction
st.header("Predict Machine Failure")
st.write("Enter feature values to predict machine failure using the XGBoost model:")
input_data = {}
for col in X_train.columns:
    input_data[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    proba = xgb_model.predict_proba(input_df)[:, 1]
    prediction = (proba > 0.6).astype(int)
    result = "Failure" if prediction[0] == 1 else "No Failure"
    st.write(f"Prediction: {result} (Probability of Failure: {proba[0]:.2f})")