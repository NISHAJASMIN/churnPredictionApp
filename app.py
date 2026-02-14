import streamlit as st
import pandas as pd
import joblib
import os
import requests
from io import BytesIO

st.title("Customer Churn Prediction App")
st.write("Upload CSV to predict churn. Model will be loaded automatically.")

# --- Model file paths ---
MODEL_LOCAL = "churn_model.pkl"
FEATURES_LOCAL = "feature_columns.pkl"

# --- Google Drive direct download URLs ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=1PKiIfXEUTGrk27R5g-C3ynfrbmAcxMSQ"
FEATURES_URL = "https://drive.google.com/uc?export=download&id=14v6PdSRDuxP-R5oisbMWvchRNfrjWTSn"

# --- Function to load file locally or download ---
def load_file(local_path, url=None):
    if os.path.exists(local_path):
        return joblib.load(local_path)
    elif url:
        st.info(f"Downloading {local_path} from cloud...")
        response = requests.get(url)
        return joblib.load(BytesIO(response.content))
    else:
        st.error(f"{local_path} not found!")
        st.stop()

# --- Load model and feature columns ---
model = load_file(MODEL_LOCAL, MODEL_URL)
feature_columns = load_file(FEATURES_LOCAL, FEATURES_URL)

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
    prediction_prob = model.predict_proba(df_encoded)[:,1]
    
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
    
    # Optional: highlight high-risk customers
    st.subheader("High-Risk Customers")
    high_risk = df[df['Churn Probability'] > 0.5]
    st.dataframe(high_risk.style.background_gradient(cmap='Reds'))
