# Viraj Potdar - Assigned Python Code
# Topic: Data Preparation, Feature Engineering, SMOTE, LogReg, SVM, GridSearch

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

def intermediate_data_preparation(df):
    """
    EPIC 2: DATA COLLECTION & PREPARATION
    Activity 2: Loading, Handling Missing, Label Converting, Train-Test Split with Stratify
    """
    print("--- Starting Intermediate Data Preparation ---")
    
    # 1. Duplicates
    print(f"Removing {df.duplicated().sum()} duplicated rows...")
    df.drop_duplicates(inplace=True)
    
    # 2. Missing Value Imputation (Using Median for numeric, Mode for categorical)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 3. Label conversion (Y/N -> 1/0)
    if 'fraud_reported' in df.columns:
        df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
        # Handle cases where synthetic data might already be 1/0
        df['fraud_reported'].fillna(0, inplace=True) 
        df['fraud_reported'] = df['fraud_reported'].astype(int)

    # 4. Feature Engineering
    # creating claim_ratio: How massive is the claim compared to what they pay?
    if 'total_claim_amount' in df.columns and 'policy_annual_premium' in df.columns:
        # add a small epsilon 0.01 to prevent division by zero
        df['claim_ratio'] = df['total_claim_amount'] / (df['policy_annual_premium'] + 0.01)
        print("Feature Engineered: 'claim_ratio'")
    
    return df

def perform_stratified_split_and_smote(X, y):
    """
    EPIC 2: Stratified Split and SMOTE Resampling
    """
    print("\nExecuting Stratified 80-20 Train-Test Split...")
    # 'stratify=y' enforces the 75:25 class imbalance exactly in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Pre-SMOTE Training Fraud ratio: {sum(y_train==1)} Fraud / {sum(y_train==0)} Legitimate")
    
    print("Applying SMOTE (Synthetic Minority Over-sampling Technique) to Training Data ONLY...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Post-SMOTE Training Fraud ratio: {sum(y_train_smote==1)} Fraud / {sum(y_train_smote==0)} Legitimate")
    
    return X_train_smote, X_test, y_train_smote, y_test

def run_cross_validation(model, X, y, model_name="Model"):
    print(f"\nRunning 5-Fold Stratified CV for {model_name}...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring_metrics, n_jobs=-1)
    
    print(f"--- {model_name} CV Metrics ---")
    print(f"Mean Accuracy : {np.mean(cv_results['test_accuracy']):.4f}  (+/- {np.std(cv_results['test_accuracy']):.4f})")
    print(f"Mean Precision: {np.mean(cv_results['test_precision']):.4f}  (+/- {np.std(cv_results['test_precision']):.4f})")
    print(f"Mean Recall   : {np.mean(cv_results['test_recall']):.4f}  (+/- {np.std(cv_results['test_recall']):.4f})")
    print(f"Mean F1-Score : {np.mean(cv_results['test_f1']):.4f}  (+/- {np.std(cv_results['test_f1']):.4f})")

def train_logistic_regression(X_train_smote, y_train_smote, X_test, y_test, X_full, y_full):
    """
    EPIC 4: MODEL BUILDING (Logistic Regression with CV)
    """
    print("\nTraining Logistic Regression Pipeline...")
    # class_weight='balanced' provides extra security on imbalanced data boundaries
    log_reg = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    run_cross_validation(log_reg, X_full, y_full, "Logistic Regression")
    
    log_reg.fit(X_train_smote, y_train_smote)
    predictions = log_reg.predict(X_test)
    
    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    return log_reg

def train_svm(X_train_scaled_smote, y_train_smote, X_test_scaled, y_test, X_full_scaled, y_full):
    """
    EPIC 4: MODEL BUILDING (SVM Linear with CV)
    Note: Must receive StandardScaler scaled spatial data
    """
    print("\nTraining Support Vector Machine (Linear)...")
    svm_model = SVC(kernel='linear', class_weight='balanced', random_state=42)
    run_cross_validation(svm_model, X_full_scaled, y_full, "SVM (Linear)")
    
    svm_model.fit(X_train_scaled_smote, y_train_smote)
    predictions = svm_model.predict(X_test_scaled)
    
    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    return svm_model

def hyperparameter_tuning_rf(X_train_smote, y_train_smote):
    """
    EPIC 5: PERFORMANCE TESTING & HYPERPARAMETER TUNING
    GridSearchCV executing a massive tensor space search for Absolute Peak Performance.
    """
    print("\nStarting Intermediate GridSearchCV for Random Forest Optimization...")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],      # Amount of bootstrap trees
        'max_depth': [10, 20, None],         # Pruning depths preventing Overfitting
        'min_samples_split': [2, 5]          # Branch splitting criteria
    }
    
    # We optimize strictly for Recall 'scoring=recall' to stop False Negatives
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='recall', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_smote, y_train_smote)
    
    print("GridSearch Phase Complete.")
    print(f"Optimal Parameters Discovered: {grid_search.best_params_}")
    print(f"Peak Operational CV Recall Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def save_model(model, filename='model.pkl'):
    """
    EPIC 6: MODEL DEPLOYMENT
    Standard Python Serialized Artifact Export
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"\nModel pipeline strictly serialized & exported to [{filename}]!")
