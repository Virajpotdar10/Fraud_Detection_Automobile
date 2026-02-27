import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n[SYSTEM] Initializing Intermediate Training Pipeline...")
print("[SYSTEM] Simulating Automobile Insurance Fraud Academic Dataset...")

# We simulate the complex dataset required by the Intermediate requirements
# Including class imbalance, and the fields needed for feature engineering
np.random.seed(42)
n_samples = 1500

# Generating baseline fields
total_claim_amount = np.random.uniform(10000, 1500000, n_samples)
policy_annual_premium = np.random.uniform(5000, 30000, n_samples)
age = np.random.randint(18, 80, n_samples)

# Inducing ~ 20% Fraud / 80% Non-Fraud (Strict Class Imbalance Scenario)
# Fraud typically correlates strongly with astronomically high claim amounts vs low premiums
claim_ratio = total_claim_amount / (policy_annual_premium + 0.1)
fraud_prob = (claim_ratio / claim_ratio.max()) * 0.7  
fraud_reported = (np.random.rand(n_samples) < fraud_prob).astype(int)

df = pd.DataFrame({
    'total_claim_amount': total_claim_amount,
    'policy_annual_premium': policy_annual_premium,
    'age': age,
    'fraud_reported': fraud_reported
})

print(f"[SYSTEM] Dataset generated. Shape: {df.shape}")
print(f"[SYSTEM] Class Distribution - Fraud: {sum(fraud_reported==1)} | Legitimate: {sum(fraud_reported==0)}")

# EPIC 2: FEATURE ENGINEERING
print("\n[EPIC 2] Executing Feature Engineering (`claim_ratio`) extraction...")
df['claim_ratio'] = df['total_claim_amount'] / (df['policy_annual_premium'] + 0.01)

# Splitting Data (Stratified ensures class ratio maintains mathematically)
print("[EPIC 2] Executing Stratified 80-20 Train-Test separation...")
X = df[['total_claim_amount', 'policy_annual_premium', 'age', 'claim_ratio']]
y = df['fraud_reported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# EPIC 2: SMOTE RESAMPLING
print("[EPIC 2] Bootstrapping Synthetic Minority using SMOTE (Resolving Imbalance)...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# EPIC 4 & 5: MODEL TRAINING & HYPERPARAMETER TUNING
print("\n[EPIC 4 & 5] Launching GridSearchCV to identify optimal Random Forest meta-estimator...")
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
param_grid = {
    'n_estimators': [50, 100], 
    'max_depth': [5, 12, None]
}

# Optimizing explicitly for 'Recall' (Catching Frauds)
grid = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='recall', n_jobs=1)
grid.fit(X_train_smote, y_train_smote)

best_model = grid.best_estimator_
print(f"[SYSTEM] Ideal Parameters Secured: {grid.best_params_}")

# EPIC 6: MODEL DEPLOYMENT (PICKLE)
print("\n[EPIC 6] Serializing predictive artifact...")
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print("[COMPLETE] Intermediate-Level Setup Finished. Please execute: `python app.py`")
