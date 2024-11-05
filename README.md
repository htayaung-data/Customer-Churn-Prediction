# Telco Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn for a telecom company. Customer churn, the loss of clients, can significantly impact a company's revenue and growth. By accurately identifying customers at risk of leaving, the company can implement targeted retention strategies to enhance customer loyalty and lifetime value.

## Business Context
Customer retention is critical for telecom companies due to the high costs associated with acquiring new customers. Understanding the factors that contribute to churn allows businesses to proactively address customer concerns and improve service offerings. The primary goal of this project is to achieve a high recall score, ensuring that most potential churners are identified even if it increases the number of false positives.

## Data Overview
The dataset comprises various customer attributes, including demographic information, service usage, billing details, and interaction with additional services. Key features include:

- **Demographics**: Gender, SeniorCitizen status, Partner, Dependents
- **Service Usage**: Tenure, PhoneService, InternetService, and other add-ons like OnlineSecurity and StreamingTV
- **Billing Information**: MonthlyCharges, TotalCharges
- **Behavioral Indicators**: Contract type, PaymentMethod, PaperlessBilling

## Methodology

### Data Loading & Cleaning
- **Data Import**: Loaded the dataset using Pandas.
- **Data Cleaning**: Converted `TotalCharges` to numeric, handling missing values by imputing with the median.
- **Duplicate Removal**: Ensured data integrity by removing duplicate entries.
- **Feature Dropping**: Excluded the `customerID` column as it does not contribute to churn prediction.

### Exploratory Data Analysis (EDA)
- **Target Variable Analysis**: Visualized the distribution of the churn variable to identify class imbalance.
- **Categorical Feature Visualization**: Analyzed the relationship between categorical features and churn using count plots.
- **Numerical Feature Analysis**: Assessed the distribution and outliers in numerical features like tenure, MonthlyCharges, and TotalCharges using box plots and histograms.

### Feature Engineering
- **Derived Features**: Created new features such as `total_services`, `monthly_charges_binned`, `tenure_binned`, `avg_monthly_charges`, `senior_with_dependents`, and `multiple_services` to capture complex relationships within the data.
- **Interaction Features**: Generated interaction terms like `tenure_MonthlyCharges`, `tenure_TotalCharges`, and `MonthlyCharges_TotalCharges`.
- **Polynomial Features**: Created polynomial features to model non-linear relationships using `PolynomialFeatures` from Scikit-learn.
- **Scaling & Encoding**: Applied standard scaling to numerical features and one-hot encoding to categorical variables to prepare the data for modeling.

### Model Selection & Training

#### Handling Class Imbalance
- **Sampling Techniques**: Addressed class imbalance using various resampling methods, including SMOTE, Random Over-Sampling, ADASYN, and SMOTEENN. After evaluation, **SMOTEENN** was selected for its balanced performance in improving recall while maintaining a reasonable number of false positives.

#### Feature Selection
- **Principal Component Analysis (PCA)**: Reduced dimensionality while retaining 95% of the variance to eliminate multicollinearity and enhance model performance.
- **Mutual Information & Recursive Feature Elimination (RFE)**: Applied feature selection methods to identify the most impactful features on churn prediction.

#### Ensemble Learning
- **Voting Classifier**: Combined multiple classifiers—**Support Vector Classifier (SVC)**, **Gradient Boosting Classifier**, and **Logistic Regression**—to make collective predictions, leveraging the strengths of each model for improved accuracy and robustness.
- **Stacking Classifier**: Implemented a stacking ensemble that trains a meta-model (**Logistic Regression**) to learn how to best combine the predictions of base models (**Support Vector Classifier**, **Gradient Boosting Classifier**, and **Logistic Regression**), resulting in enhanced predictive performance.

### Evaluation & Interpretation
- **Performance Metrics**: Assessed models using precision, recall, F1-score, accuracy, ROC-AUC, and Gini coefficient.
- **Confusion Matrix**: Analyzed true positives, false positives, true negatives, and false negatives to understand model effectiveness.
- **ROC Curve**: Plotted ROC curves to evaluate the trade-off between true positive rate and false positive rate.
- **Probability Distribution**: Visualized the distribution of predicted probabilities to gauge model confidence.

### Deployment
- **Streamlit Application**: Developed an interactive web application using Streamlit to allow users to input customer data and receive churn predictions in real-time.
- **Model Serialization**: Saved the trained model and preprocessing artifacts using Joblib for seamless integration into the deployment pipeline.

## Technologies Used
- **Programming Languages**: Python
- **Libraries & Frameworks**:
  - Data Manipulation: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, XGBoost, Gradient Boosting, CatBoost, Random Forest, Logistic Regression, Support Vector Classifier, K-Nearest Neighbors, Gaussian Naive Bayes
  - Handling Imbalanced Data: imbalanced-learn (SMOTE, SMOTEENN, ADASYN, RandomOverSampler, RandomUnderSampler)
  - Hyperparameter Tuning: Optuna
  - Ensemble Methods: VotingClassifier, StackingClassifier
  - Deployment: Streamlit
  - Model Persistence: Joblib

## Usage
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/telco-churn-prediction.git
   cd telco-churn-prediction
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

4. **Interact with the App**
   - Input customer details through the provided interface.
   - Receive churn predictions along with the probability score.

## Results
The final model demonstrates strong performance in predicting customer churn with high recall, ensuring that most potential churners are identified. Key metrics include:

- **Precision**: Balances the accuracy of positive predictions.
- **Recall**: Ensures that a significant proportion of actual churners are captured.
- **F1 Score**: Provides a harmonic mean between precision and recall.
- **ROC-AUC**: Indicates the model's ability to distinguish between classes effectively.
- **Gini Coefficient**: Reflects the model's discriminatory power.

The ensemble methods, particularly the **Stacking Classifier**, contributed to improved recall and overall model robustness by leveraging the strengths of individual classifiers such as **Support Vector Classifier**, **Gradient Boosting Classifier**, and **Logistic Regression**.

## Conclusion
This project successfully developed a predictive model for customer churn in the telecom industry, employing comprehensive data preprocessing, feature engineering, and advanced machine learning techniques, including ensemble methods like **Voting** and **Stacking Classifiers**. The deployed Streamlit application offers a user-friendly interface for real-time churn prediction, facilitating actionable insights for the business. Future enhancements could include integrating more diverse data sources and exploring additional modeling techniques to further improve prediction accuracy and operational efficiency.

