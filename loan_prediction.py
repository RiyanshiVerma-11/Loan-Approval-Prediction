# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ---------------------
# Load dataset
# ---------------------
@st.cache_data
def load_data():
    df = pd.read_csv("loan_prediction.csv")
    return df

data = load_data()

st.title("Loan Prediction App")
st.write("Predict if a loan will be approved or not.")

# ---------------------
# Sidebar Navigation
# ---------------------
option = st.sidebar.selectbox("Choose Section", ["Data Analysis", "Model Prediction"])

if option == "Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")

    # Create Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dataset Preview", 
        "Summary Statistics", 
        "Missing Values", 
        "Categorical Value Counts", 
        "Correlation Heatmap", 
        "Interactive Plot"
    ])

    # Tab 1 - Dataset Preview
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

    # Tab 2 - Summary Statistics
    with tab2:
        st.subheader("Summary Statistics")
        st.write(data.describe())

    # Tab 3 - Missing Values
    with tab3:
        st.subheader("Missing Values")
        st.write(data.isnull().sum())


    # Tab 4 - Categorical Columns
    with tab4:
        st.subheader("Categorical Value Counts")

        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        selected_cat_col = st.selectbox("Select a Categorical Column", cat_cols)

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Value Counts for {selected_cat_col}:**")
            st.write(data[selected_cat_col].value_counts())

        with col2:
            fig, ax = plt.subplots()
            sns.countplot(
                x=data[selected_cat_col],
                order=data[selected_cat_col].value_counts().index,
                ax=ax
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)


    # Tab 5 - Correlation Heatmap
    with tab5:
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        numeric_df = data.select_dtypes(include=['int64', 'float64'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Tab 6 - Interactive Plot
    with tab6:
        st.subheader("Interactive Plot")
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        selected_col = st.selectbox("Select Column for Histogram", numeric_cols)
        bins = st.slider("Number of Bins", 5, 50, 20)
        plt.figure(figsize=(8, 5))
        sns.histplot(data[selected_col], bins=bins, kde=True)
        st.pyplot(plt)



elif option == "Model Prediction":
    # ---------------------
    # Data Preprocessing
    # ---------------------
    df = data.copy()
    
    # Fill missing numerical values with median
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    
    # Encode target
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # ---------------------
    # Train model
    # ---------------------
    X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test accuracy
    y_pred = model.predict(X_test)
    st.write(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    
    # ---------------------
    # Streamlit User Input
    # ---------------------
    st.sidebar.header("Enter Loan Details")
    
    def user_input():
        Gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
        Married = st.sidebar.selectbox("Married", ['Yes', 'No'])
        Dependents = st.sidebar.selectbox("Dependents", ['0', '1', '2', '3+'])
        Education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
        Self_Employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])
        ApplicantIncome = st.sidebar.number_input("Applicant Income", min_value=0, value=5000)
        CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", min_value=0, value=0)
        LoanAmount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0, value=100)
        Loan_Amount_Term = st.sidebar.number_input("Loan Amount Term (in months)", min_value=12, max_value=480, value=360)
        Credit_History = st.sidebar.selectbox("Credit History", [1.0, 0.0])
        Property_Area = st.sidebar.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])
        
        # Encode user input same as training
        input_data = {
            'Gender': le.fit_transform(['Male','Female'])[list(['Male','Female']).index(Gender)],
            'Married': le.fit_transform(['No','Yes'])[list(['No','Yes']).index(Married)],
            'Dependents': le.fit_transform(['0','1','2','3+'])[list(['0','1','2','3+']).index(Dependents)],
            'Education': le.fit_transform(['Graduate','Not Graduate'])[list(['Graduate','Not Graduate']).index(Education)],
            'Self_Employed': le.fit_transform(['No','Yes'])[list(['No','Yes']).index(Self_Employed)],
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': le.fit_transform(['Urban','Semiurban','Rural'])[list(['Urban','Semiurban','Rural']).index(Property_Area)]
        }
        
        features = pd.DataFrame(input_data, index=[0])
        return features
    
    input_df = user_input()
    
    # ---------------------
    # Make Prediction
    # ---------------------
    if st.button("Predict Loan Status"):
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][prediction]
        
        if prediction == 1:
            st.success(f"Loan Approved ✅ (Confidence: {prediction_proba*100:.2f}%)")
        else:
            st.error(f"Loan Not Approved ❌ (Confidence: {prediction_proba*100:.2f}%)")
