# Automobile Insurance Fraud Detection System
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-green.svg)
![ML](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange.svg)

## 📌 Project Overview
The automotive insurance sector natively processes millions of claims annually. While the majority are legitimate, a statistically vital minority are staged, exaggerated, or fabricated. This intermediate-level Machine Learning project dynamically predicts the probability of a fraudulent automobile insurance claim based on customer demographics, policy parameters, and incident severity.

By mathematically prioritizing **"Recall"** (the ability to successfully catch Frauds), resolving severe **Class Imbalance (75:25)** via SMOTE, and utilizing an engineered feature (`claim_ratio`), this project effectively replaces hard-coded human auditing rules with an adaptive **Random Forest Meta-Estimator**.

---

## 🚀 Live Demo (Render)
*Coming soon! Follow the local deployment steps below in the meantime.*

---

## 🧠 Technical Architecture 
* **Data Preparation:** Handles structural NaN anomalies via median/mode statistical imputation.
* **Feature Engineering:** Dynamically generates `claim_ratio` (Claim Amount / Annual Premium) to measure incident severity vs baseline.
* **Encoding & Scaling:** Utilizes One-Hot Encoding (`drop_first=True`) and `StandardScaler` to remove geometric magnitude bias for algorithms like SVM and KNN.
* **SMOTE Resampling:** Applies Synthetic Minority Over-sampling Technique natively to the training data to balance the 75% Legitimate vs 25% Fraud skew.
* **Stratified Cross-Validation:** Operates on 5-Fold Stratified CV to prevent data leakage and guarantee metrics.
* **Algorithms Tested:** Logistic Regression, Naïve Bayes, Support Vector Machine (Linear), K-Nearest Neighbors, Decision Tree, Random Forest.
* **Deployment:** Best estimator serialized via `pickle` and served through a lightweight Flask REST API.

---

## 🛠️ Installation & Setup (Local Environment)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Virajpotdar10/Fraud_Detection_Automobile.git
cd Fraud_Detection_Automobile
```

### Step 2: Install Dependencies
It is highly recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model & Serialize
Generate the intermediate data, apply SMOTE, run GridSearchCV, and export the tuned `model.pkl`.
```bash
python train.py
```

### Step 4: Launch the Web Application
Start the Flask web server to interact with the model via a clean UI.
```bash
python app.py
```
Open a browser and navigate to: `http://127.0.0.1:5000/`

---

## 📊 Model Performance Evaluation
The models were evaluated under strict 5-Fold Stratified CV. The primary objective function was maximizing **Recall** to protect institutional capital from false negatives.

| Algorithm | Mean CV Accuracy | Mean CV Precision | Mean CV Recall | F1-Score | Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naïve Bayes** | 68.4% | 45.1% | 58.2% | 50.8% | Probabilistic |
| **KNN** | 71.2% | 46.3% | 40.5% | 43.1% | Euclidean Distance + Scaling |
| **Decision Tree** | 77.1% | 53.6% | 56.4% | 54.9% | Sequential Logic (`class_weight`) |
| **Logistic Regression**| 73.5% | 49.8% | 52.1% | 50.9% | Sigmoid Regression + Scaling |
| **SVM (Linear)** | 75.8% | 55.4% | 51.0% | 53.0% | Hyperplane Max-Margin |
| **Random Forest** | **83.6%** | **68.2%** | **65.3%** | **66.7%** | **Ensemble Bootstrapping** |

*Random Forest was deployed as the premier artifact following hyperparameter tuning.*

---

## 👨‍💻 Development Team
Developed by:
* **Snehal Tanaji Khot** (Workflow, EDA Visualizations, Scaling, Decision Tree/KNN)
* **Samruddhi Yadav** (Business Problem, Literature Survey, OHE Encoding, Random Forest)
* **Tanuja Talekar** (Business Requirements, Demonstrations, Model Comparison, Naïve Bayes)
* **Viraj Potdar** (Data Preparation, Feature Engineering, SMOTE, Deployment Pipeline)
