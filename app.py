import streamlit as st
import pandas as pd
import joblib
model = joblib.load('churn_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
#app title
st.title('Customer Churn Prediction App')
st.write('Upload customer data csv to predict churn')
#upload csv
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Encode
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Make predictions
    predictions = model.predict(df_encoded)
    prediction_prob = model.predict_proba(df_encoded)[:, 1]

    # Add results
    df['Churn_Prediction'] = predictions
    df['Churn_Probability'] = prediction_prob

    st.subheader("Predictions")
    st.dataframe(df)

    # -------- Feature Importance --------
    st.subheader("Top 10 Feature Importances")

    feature_importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(10)

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8,5))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title("Feature Importance (Random Forest)")
    st.pyplot(plt)

    # -------- Download Button --------
    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name='churn_predictions.csv',
        mime='text/csv',
    )
