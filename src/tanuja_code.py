# Tanuja Talekar - Assigned Python Code
# Topic: Naïve Bayes Classifier with Intermediate Cross-Validation Integration

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def run_cross_validation(model, X, y, model_name="Model"):
    """
    Ensures that the 25:75 class imbalance ratio is heavily maintained
    via 5-Fold Stratified sampling across all training phases.
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

def train_naive_bayes(X_train, y_train, X_test, y_test, X_full, y_full):
    """
    EPIC 4: MODEL BUILDING
    Story: Naïve Bayes
    Note: Probabilistic model, utilizes Gaussian distribution calculation on continuous data.
    """
    print("\nTraining Gaussian Naïve Bayes Classifier...")
    nb_model = GaussianNB()
    
    # Run the full 5-Fold cross validation to observe pure unbiased statistical metrics
    run_cross_validation(nb_model, X_full, y_full, "Naïve Bayes")
    
    # Fit upon explicit training split for inference extraction
    nb_model.fit(X_train, y_train)
    predictions = nb_model.predict(X_test)
    
    print("\nTest Set Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"Test F1-Score: {f1_score(y_test, predictions):.4f}")
    
    return nb_model
