# 📊 Loan Prediction App

A machine learning–powered web app built with **Streamlit** that predicts whether a loan application will be **approved or rejected** based on applicant details.  
It also includes **Exploratory Data Analysis (EDA)** tools for better understanding of the dataset.

---

## 🚀 Features

- **Data Analysis Tabs**:
  - Dataset Preview
  - Summary Statistics
  - Missing Values
  - Correlation Heatmap
  - Interactive Histograms
  - Categorical Value Counts (with charts + tables side by side)

- **Machine Learning**:
  - Preprocessing of categorical & numerical features
  - Random Forest Classifier
  - Model accuracy displayed after training

- **Prediction Interface**:
  - Sidebar input form for applicant details
  - Predicts loan approval status ✅❌
  - Shows model confidence score

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Streamlit**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib & Seaborn**

## 📁 Repository Structure
Loan-Approval-Prediction/
├── loan_prediction.csv # Dataset
├── app.py # Main Streamlit app
├── README.md # Project README (this file)
├── requirements.txt # Python dependencies



---

## ⚙️ Features

- Interactive **Data Analysis** section:
  - Dataset preview
  - Summary statistics
  - Missing values overview
  - Categorical value counts (with chart next to table)
  - Correlation heatmap (for numeric columns)
  - Interactive histograms for numeric columns

- **Prediction Model**:
  - Preprocessing: handling missing values, encoding categorical features
  - Random Forest classifier
  - Displaying model accuracy

- **User Input Form**:
  - Sidebar form for entering applicant details
  - Output: Prediction (Approved / Rejected) with confidence score

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python     | Backend & ML logic |
| Streamlit  | Web UI & interactive interface |
| Pandas, Numpy | Data manipulation |
| Scikit-learn | Model training & evaluation |
| Seaborn, Matplotlib | Visualizations |

---

## 📊 Usage / How to Run Locally

Clone the repository:

   ```bash
   git clone https://github.com/RiyanshiVerma-11/Loan-Approval-Prediction.git
   cd Loan-Approval-Prediction

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac / Linux

pip install -r requirements.txt


streamlit run app.py


✅ Model Performance

Algorithm: Random Forest Classifier

Current Accuracy: ~78% (on test data) – can vary based on how data is split and preprocessed.

🔍 Possible Improvements

Hyperparameter tuning (e.g. using GridSearchCV) to improve model accuracy

Better encoding of categorical features (One-Hot Encoding vs Label Encoding)

Imputation strategies beyond median / mode

Handling class imbalance more explicitly (if applicable)

More feature engineering (e.g. combining incomes, deriving new features)

👥 Contributions

This is part of my internship project. Feel free to explore, suggest improvements, raise issues or pull requests.

📄 License

Specify the license here (e.g. MIT, GNU GPL, etc.), if you want to make it open.
If you haven’t chosen one yet, you can include a placeholder like:
Licensed under MIT License