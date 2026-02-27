# Snehal Tanaji Khot - Assigned Python Code
# Topic: EDA Visualizations, Multivariate Analysis, Scaling, and Model Training (Decision Tree, KNN) with 5-Fold CV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def perform_eda_visualizations(df):
    """
    EPIC 3: EXPLORATORY DATA ANALYSIS
    Generates academic-level plots for structural interpretation.
    """
    print("--- Generating Intermediate-Level EDA Visualizations ---")
    sns.set_theme(style="whitegrid")
    
    # 1. Fraud distribution plot (Demonstrating class imbalance)
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='fraud_reported', palette='Set2')
    plt.title("Distribution of Target Variable (Fraud Reported)")
    ax.bar_label(ax.containers[0])
    plt.show()
    
    # 2. Claim amount comparison (Boxplot for median and outliers)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='fraud_reported', y='total_claim_amount', palette='pastel')
    plt.title("Total Claim Amount vs Fraud (Identifies Aggressive Claims)")
    plt.show()
    
    # 3. Age vs fraud (KDE for demographic relationships)
    if 'age' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df, x='age', hue='fraud_reported', fill=True, common_norm=False, palette='crest')
        plt.title("Age Density Distribution relative to Fraud")
        plt.show()
    
    # 4. Correlation heatmap (Multicollinearity check)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def perform_scaling(X_train, X_test):
    """
    EPIC 3: EXPLORATORY DATA ANALYSIS
    Story: Scaling. Essential for SVM, KNN and LogReg convergence.
    """
    print("--- Applying StandardScaler to eliminate magnitude bias ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def run_cross_validation(model, X, y, model_name="Model"):
    """
    EPIC 4: MODEL BUILDING
    Applies Stratified 5-Fold Cross Validation.
    """
    print(f"\nRunning 5-Fold Stratified CV for {model_name}...")
    
    # Stratified K-Fold ensures the class imbalance ratio is maintained in all 5 folds.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    print(f"--- {model_name} Cross-Validation Results ---")
    print(f"Mean CV Accuracy : {np.mean(scores['test_accuracy']):.4f}  (+/- {np.std(scores['test_accuracy']):.4f})")
    print(f"Mean CV Precision: {np.mean(scores['test_precision']):.4f}  (+/- {np.std(scores['test_precision']):.4f})")
    print(f"Mean CV Recall   : {np.mean(scores['test_recall']):.4f}  (+/- {np.std(scores['test_recall']):.4f})")
    print(f"Mean CV F1-Score : {np.mean(scores['test_f1']):.4f}  (+/- {np.std(scores['test_f1']):.4f})")

def train_decision_tree(X_train, y_train, X_test, y_test, X_full, y_full):
    """
    EPIC 4: MODEL BUILDING
    Story: Decision Tree with class_weight='balanced'
    """
    print("\nTraining Decision Tree with class weighting...")
    # Used 'balanced' to heuristically handle class imbalance natively
    dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=6)
    
    run_cross_validation(dt_model, X_full, y_full, "Decision Tree")
    
    dt_model.fit(X_train, y_train)
    predictions = dt_model.predict(X_test)
    
    print("Test Set Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    return dt_model

def train_knn(X_train_scaled, y_train, X_test_scaled, y_test, X_full_scaled, y_full):
    """
    EPIC 4: MODEL BUILDING
    Story: KNN
    """
    print("\nTraining K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    
    run_cross_validation(knn_model, X_full_scaled, y_full, "K-Nearest Neighbors")
    
    knn_model.fit(X_train_scaled, y_train)
    predictions = knn_model.predict(X_test_scaled)
    
    print("Test Set Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    return knn_model
