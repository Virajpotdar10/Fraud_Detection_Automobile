import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

os.makedirs("models", exist_ok=True)
os.makedirs("eda_results", exist_ok=True)

print("--- Data Preprocessing ---")
df = pd.read_csv("dataset/insurance_fraud.csv")

df["claim_ratio"] = df["total_claim_amount"] / df["policy_annual_premium"]

df.fillna(method="ffill", inplace=True)

print("--- Generating EDA Plots ---")
plt.figure()
sns.countplot(x="fraud_reported", data=df)
plt.title("Fraud vs Genuine Claims Distribution")
plt.savefig("eda_results/fraud_distribution.png")
plt.close()

plt.figure()
sns.histplot(df["total_claim_amount"], kde=True, bins=30)
plt.title("Distribution of Total Claim Amount")
plt.savefig("eda_results/claim_amount_dist.png")
plt.close()

plt.figure()
sns.boxplot(x="fraud_reported", y="total_claim_amount", data=df)
plt.title("Total Claim Amount vs Fraud Reported")
plt.savefig("eda_results/claim_vs_fraud_boxplot.png")
plt.close()

plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("eda_results/correlation_heatmap.png")
plt.close()

features_for_model = [
    "age", "months_as_customer", "policy_annual_premium",
    "total_claim_amount", "number_of_vehicles_involved", 
    "witnesses", "injury_claim", "property_claim", 
    "vehicle_claim", "incident_hour_of_the_day"
]

X = df[features_for_model]
y = df["fraud_reported"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pickle.dump(scaler, open("models/scaler.pkl", "wb"))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("--- Model Building and Comparison ---")

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naïve Bayes": GaussianNB(),
    "SVM": SVC()
}

results = []
best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

print("\n--- Model Evaluation (Best Model) ---")
print(f"Best Model selected: {results_df.loc[results_df['F1 Score'].idxmax()]['Model']} (based on F1)")
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Best Model)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.savefig("eda_results/confusion_matrix.png")
plt.close()

print("\nConfusion Matrix Explanation:")
print(f"True Positive (Fraud pred as Fraud): {cm[1,1]}")
print(f"False Positive (Genuine pred as Fraud): {cm[0,1]}")
print(f"True Negative (Genuine pred as Genuine): {cm[0,0]}")
print(f"False Negative (Fraud pred as Genuine): {cm[1,0]}")
print("Recall is critical here because failing to catch a fraudulent claim (False Negative) costs the company significantly more than manually double-checking a genuine claim (False Positive).")

pickle.dump(best_model, open("models/model.pkl", "wb"))
print("\nBest model saved to `models/model.pkl` successfully!")
print("Scaler saved to `models/scaler.pkl` successfully!")
