import streamlit as st
import pandas as pd
import joblib
import os
import requests

st.title("Customer Churn Prediction App")
st.write("Upload CSV to predict churn. Model will be loaded automatically.")

# --- Local file paths ---
MODEL_LOCAL = "churn_model.pkl"
FEATURES_LOCAL = "feature_columns.pkl"

# --- Google Drive file IDs ---
MODEL_ID = "1PKiIfXEUTGrk27R5g-C3ynfrbmAcxMSQ"
FEATURES_ID = "14v6PdSRDuxP-R5oisbMWvchRNfrjWTSn"

# --- Function to download from Google Drive ---
def download_file(file_id, local_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)
    st.success(f"Downloaded {local_path}!")

# --- Function to load joblib file ---
def load_joblib(local_path, file_id=None):
    if os.path.exists(local_path):
        return joblib.load(local_path)
    elif file_id:
        st.info(f"Downloading {local_path} from Google Drive...")
        download_file(file_id, local_path)
        return joblib.load(local_path)
    else:
        st.error(f"{local_path} not found!")
        st.stop()

# --- Load model and feature columns ---
model = load_joblib(MODEL_LOCAL, MODEL_ID)
feature_columns = load_joblib(FEATURES_LOCAL, FEATURES_ID)

# --- Upload CSV for prediction ---
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Encode categorical columns
    df_encoded = pd.get_dummies(df)
    
    # Align with training columns
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Make predictions
    predictions = model.predict(df_encoded)
    prediction_prob = model.predict_proba(df_encoded)[:, 1]
    
    # Add results to dataframe
    df['Churn Prediction'] = predictions
    df['Churn Probability'] = prediction_prob
    
    st.subheader("Predictions")
    st.dataframe(df)
    
    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='churn_predictions.csv',
        mime='text/csv',
    )
    
    # Highlight high-risk customers
    st.subheader("High-Risk Customers")
    high_risk = df[df['Churn Probability'] > 0.5]
    st.dataframe(high_risk.style.background_gradient(cmap='Reds'))
