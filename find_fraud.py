import pandas as pd
import pickle
import numpy as np

# Load files
try:
    df = pd.read_csv('dataset/insurance_fraud.csv')
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    model = pickle.load(open('models/model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

features = [
    'age', 'months_as_customer', 'policy_annual_premium', 'total_claim_amount', 
    'number_of_vehicles_involved', 'witnesses', 'injury_claim', 'property_claim', 
    'vehicle_claim', 'incident_hour_of_the_day'
]

# Get model predictions for all rows
X = df[features]
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)[:, 1]

df['model_pred'] = preds
df['model_prob'] = probs

# Filter rows the model actually thinks are fraud
fraud_cases = df[df['model_pred'] == 1].sort_values(by='model_prob', ascending=False)

if not fraud_cases.empty:
    print("Found actual fraud cases in your current dataset!")
    for i in range(min(1, len(fraud_cases))):
        case = fraud_cases.iloc[i]
        print(f"\n--- Case {i+1} (Probability: {case['model_prob']*100:.2f}%) ---")
        for f in features:
            print(f"{f}: {case[f]}")
else:
    print("No cases predicted as fraud. Please train the model with `python notebooks/fraud_model.py` first.")
