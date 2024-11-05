import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

# Load the saved model and other necessary components
model_data = load('artifacts/model_data.joblib')
model = model_data['model']
features = model_data['features']
scaler = model_data['scaler']
pca = model_data['cols_so_scale']

# Define default values for the input fields
default_values = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

# Streamlit app title
st.title("Telco Customer Churn Prediction")

# Explanation of data labels
st.header("Feature Descriptions")
st.write("""
- **Gender**: The customer's gender. This feature may have a minor impact on churn prediction.
- **SeniorCitizen**: Indicates if the customer is a senior citizen (1) or not (0). Senior citizens might have different churn patterns. **Range**: 0 (not a senior) or 1 (senior).
- **Partner**: Whether the customer has a partner. Customers with partners might be less likely to churn.
- **Dependents**: Whether the customer has dependents. Having dependents might influence the customer's decision to stay.
- **Tenure**: The number of months the customer has stayed with the company. Longer tenure often indicates lower churn risk. **Range**: Typically from 0 to 72 months.
- **PhoneService**: Whether the customer has phone service. This can affect the overall service satisfaction.
- **MultipleLines**: Whether the customer has multiple lines. More lines might indicate higher engagement.
- **InternetService**: Type of internet service. Different services might have different churn rates.
- **OnlineSecurity**: Whether the customer has online security add-on. Security services might reduce churn.
- **OnlineBackup**: Whether the customer has online backup add-on. Backup services might reduce churn.
- **DeviceProtection**: Whether the customer has device protection add-on. Protection services might reduce churn.
- **TechSupport**: Whether the customer has tech support add-on. Support services might reduce churn.
- **StreamingTV**: Whether the customer has streaming TV service. Entertainment services might reduce churn.
- **StreamingMovies**: Whether the customer has streaming movies service. Entertainment services might reduce churn.
- **Contract**: The contract term of the customer. Longer contracts usually indicate lower churn risk.
- **PaperlessBilling**: Whether the customer uses paperless billing. This might affect customer satisfaction.
- **PaymentMethod**: The customer's payment method. Automatic payments might correlate with lower churn.
- **MonthlyCharges**: The amount charged to the customer monthly. Higher charges might increase churn risk. **Range**: Typically from 18 to 120.
- **TotalCharges**: The total amount charged to the customer. This is a cumulative measure of customer value. **Range**: Varies widely depending on tenure and monthly charges.
""")

# Create input fields for user to enter data in a 4-row by 5-column layout
st.header("Customer Information")
input_data = {}
rows = 4
cols = 5

# Create a grid layout with spacing
for row in range(rows):
    cols = st.columns(5)
    for col_index, (feature, default) in enumerate(list(default_values.items())[row*5:(row+1)*5]):
        with cols[col_index]:
            if feature == 'gender':
                input_data[feature] = st.selectbox(feature, options=['Female', 'Male'], index=['Female', 'Male'].index(default))
            elif feature == 'Partner':
                input_data[feature] = st.selectbox(feature, options=['Yes', 'No'], index=['Yes', 'No'].index(default))
            elif feature == 'Dependents':
                input_data[feature] = st.selectbox(feature, options=['No', 'Yes'], index=['No', 'Yes'].index(default))
            elif feature == 'PhoneService':
                input_data[feature] = st.selectbox(feature, options=['No', 'Yes'], index=['No', 'Yes'].index(default))
            elif feature == 'MultipleLines':
                input_data[feature] = st.selectbox(feature, options=['No phone service', 'No', 'Yes'], index=['No phone service', 'No', 'Yes'].index(default))
            elif feature == 'InternetService':
                input_data[feature] = st.selectbox(feature, options=['DSL', 'Fiber optic', 'No'], index=['DSL', 'Fiber optic', 'No'].index(default))
            elif feature == 'OnlineSecurity':
                input_data[feature] = st.selectbox(feature, options=['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(default))
            elif feature == 'OnlineBackup':
                input_data[feature] = st.selectbox(feature, options=['Yes', 'No', 'No internet service'], index=['Yes', 'No', 'No internet service'].index(default))
            elif feature == 'DeviceProtection':
                input_data[feature] = st.selectbox(feature, options=['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(default))
            elif feature == 'TechSupport':
                input_data[feature] = st.selectbox(feature, options=['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(default))
            elif feature == 'StreamingTV':
                input_data[feature] = st.selectbox(feature, options=['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(default))
            elif feature == 'StreamingMovies':
                input_data[feature] = st.selectbox(feature, options=['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(default))
            elif feature == 'Contract':
                input_data[feature] = st.selectbox(feature, options=['Month-to-month', 'One year', 'Two year'], index=['Month-to-month', 'One year', 'Two year'].index(default))
            elif feature == 'PaperlessBilling':
                input_data[feature] = st.selectbox(feature, options=['Yes', 'No'], index=['Yes', 'No'].index(default))
            elif feature == 'PaymentMethod':
                input_data[feature] = st.selectbox(feature, options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(default))
            else:
                input_data[feature] = st.number_input(feature, value=default)

# Predict button
if st.button("Predict"):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess the input data
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    input_df['total_services'] = (input_df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == "Yes").sum(axis=1)

    # Check for unique values before applying qcut
    if input_df['MonthlyCharges'].nunique() > 1:
        input_df['monthly_charges_binned'] = pd.qcut(input_df['MonthlyCharges'], q=4, labels=False)
    else:
        input_df['monthly_charges_binned'] = 0

    input_df['tenure_binned'] = pd.cut(input_df['tenure'], bins=[0, 12, 24, 100], labels=False)
    input_df['avg_monthly_charges'] = input_df.apply(lambda x: x['TotalCharges'] / x['tenure'] if x['tenure'] != 0 else 0, axis=1)
    input_df['senior_with_dependents'] = input_df.apply(lambda x: 1 if x['SeniorCitizen'] == 1 and x['Dependents'] == 'Yes' else 0, axis=1)
    input_df['multiple_services'] = input_df.apply(lambda x: 1 if x['total_services'] > 1 else 0, axis=1)
    input_df['tenure_MonthlyCharges'] = input_df['tenure'] * input_df['MonthlyCharges']
    input_df['tenure_TotalCharges'] = input_df['tenure'] * input_df['TotalCharges']
    input_df['MonthlyCharges_TotalCharges'] = input_df['MonthlyCharges'] * input_df['TotalCharges']

    # Polynomial features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(input_df[numerical_cols])
    poly_df = pd.DataFrame(poly_features, columns=[f'poly_{col}' for col in poly.get_feature_names_out(numerical_cols)])
    input_df = pd.concat([input_df, poly_df], axis=1)

    # Scale numerical features
    numerical_cols = input_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # One-hot encode categorical features
    categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Ensure all necessary columns are present
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[features]

    # Apply PCA transformation
    input_df_pca = pca.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df_pca)
    prediction_proba = model.predict_proba(input_df_pca)[:, 1]

    # Display the prediction
    st.subheader("Prediction")
    if prediction_proba[0] < 0.25:
        st.write("The customer is very unlikely to churn.")
    elif prediction_proba[0] < 0.50:
        st.write("The customer is unlikely to churn.")
    elif prediction_proba[0] < 0.75:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is very likely to churn.")
    st.write(f"Probability of churn: {prediction_proba[0]:.2f}")

 