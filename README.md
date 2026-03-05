# Automobile Insurance Fraud Detection System using Machine Learning

## 1️⃣ PROJECT OVERVIEW

### What is Insurance Fraud?
Insurance fraud occurs when individuals deceive an insurance company to collect money to which they are not entitled. This can include staging accidents, exaggerating claims, or submitting false information about the incident.

### Why Fraud Detection is Important?
Fraudulent claims cost insurance companies billions of dollars annually, which leads to increased premiums for honest customers. Detecting fraud early helps companies minimize financial losses, maintain affordable insurance rates, and ensure a fair system for everyone. 

### Objective of the Project
The primary objective of this project is to build a machine learning pipeline that can effectively predict whether an incoming automobile insurance claim is "Fraud" or "Genuine." This system will empower insurance agents to flag high-risk claims for further investigation.

### Expected Output
A web-based application where an insurance agent can input details of a claim. The system processes these inputs using a trained machine learning model and instantly predicts:
- **1 (Fraud)**
- **0 (Genuine)**

---

## 2️⃣ DATASET DESCRIPTION

The dataset contains automobile insurance claim details with various features that describe the customer, the policy, and the incident.

### Features
* `months_as_customer`: Number of months the person has been a customer.
* `age`: Age of the customer.
* `policy_state`: State where the policy was issued.
* `policy_deductable`: The amount the customer pays before insurance kicks in.
* `policy_annual_premium`: The yearly cost of the insurance policy.
* `umbrella_limit`: Extra liability insurance coverage.
* `incident_type`: Type of incident (e.g., Single Vehicle Collision, Multi-vehicle Collision).
* `collision_type`: Type of physical collision (e.g., Rear Collision, Front Collision).
* `incident_severity`: The severity of the damage (e.g., Major Damage, Minor Damage).
* `incident_state`: The state where the incident occurred.
* `incident_hour_of_the_day`: Hour when the incident happened (0-23).
* `number_of_vehicles_involved`: Total vehicles involved in the incident.
* `bodily_injuries`: Number of bodily injuries reported.
* `witnesses`: Number of witnesses present.
* `total_claim_amount`: Total amount claimed for the incident.
* `injury_claim`: Claim amount specifically for bodily injuries.
* `property_claim`: Claim amount for property damage.
* `vehicle_claim`: Claim amount for the vehicle damage.
* `auto_year`: The manufacturing year of the automobile.

### Target Variable
* `fraud_reported`: Indicates if the claim was found to be fraudulent.
  * **1 = Fraud**
  * **0 = Genuine**

---

## How to Run the Project
1. Install requirements: `pip install -r requirements.txt` (to be created)
2. Generate dataset: `python dataset_generator.py`
3. Train model & run EDA: `python notebooks/fraud_model.py`
4. Run web app: `python app.py`
5. Visit `http://127.0.0.1:5000/` in your browser.
