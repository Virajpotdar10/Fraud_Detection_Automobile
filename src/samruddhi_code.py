# Samruddhi Yadav - Assigned Python Code
# Topics: Encoding Categorical Features, Random Forest Classifier with CV

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def encode_categorical_features(df):
    """
    EPIC 3: EXPLORATORY DATA ANALYSIS
    Encoding Categorical Features Using One-Hot Encoding
    """
    print("--- Applying One-Hot Encoding ---")
    print(f"Original Data Shape: {df.shape}")
    
    # Identify non-numeric columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'fraud_reported' in categorical_cols:
        categorical_cols.remove('fraud_reported')
        
    # Apply get_dummies with drop_first=True to avoid perfect multi-collinearity (Dummy Variable Trap)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"Encoded Data Shape: {df_encoded.shape}")
    return df_encoded

def run_cross_validation(model, X, y, model_name="Model"):
    """
    Intermediate 5-Fold Stratified Cross-Validation mechanism.
    Ensures that the 25:75 class imbalance ratio is maintained across all 5 folds.
    """
    print(f"\nRunning 5-Fold Stratified CV for {model_name}...")
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    cv_results = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring_metrics, n_jobs=-1)
    
    print(f"--- {model_name} CV Metrics ---")
    print(f"Mean Accuracy : {np.mean(cv_results['test_accuracy']):.4f}  (+/- {np.std(cv_results['test_accuracy']):.4f})")
    print(f"Mean Precision: {np.mean(cv_results['test_precision']):.4f}  (+/- {np.std(cv_results['test_precision']):.4f})")
    print(f"Mean Recall   : {np.mean(cv_results['test_recall']):.4f}  (+/- {np.std(cv_results['test_recall']):.4f})")
    print(f"Mean F1-Score : {np.mean(cv_results['test_f1']):.4f}  (+/- {np.std(cv_results['test_f1']):.4f})")

def train_random_forest(X_train, y_train, X_test, y_test, X_full, y_full):
    """
    EPIC 4: MODEL BUILDING
    Story: Random Forest with class_weight handling imbalance implicitly.
    """
    print("\nTraining Random Forest Ensemble Classifier...")
    
    # class_weight='balanced' automatically adjusts weights inversely proportional to class frequencies
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=12)
    
    # Validate the architecture using CV
    run_cross_validation(rf_model, X_full, y_full, "Random Forest")
    
    # Final Fit for inference/deployment
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    
    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"Test F1-Score: {f1_score(y_test, predictions):.4f}")
    
    return rf_model
