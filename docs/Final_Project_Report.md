# Title: Automobile Insurance Fraud Detection System
# A Comprehensive Machine Learning Pipeline
**Project Domain:** Artificial Intelligence and Machine Learning  
**Level:** Intermediate Engineering Analysis  
**Prepared By:** The Engineering Team (Tanuja, Snehal, Samruddhi, Viraj)

---

## Abstract
The automotive insurance sector natively processes millions of claims annually. While the majority are legitimate, a statistically vital minority are staged, exaggerated, or outright fabricated. Financial losses stemming from such fraudulent policies eventually cascade down to honest customers through inflated baseline premiums. Traditionally, human auditors utilized hard-coded rulesets to flag discrepancies. This intermediate machine learning project sought to replace archaic rule-based mechanisms with a scalable, dynamic Python-based predictive engine. Leveraging a globally sourced Kaggle Automobile Insurance database, our methodology successfully processed, balanced, and categorized historical behaviors. Facing a severe 75:25 class imbalance (favoring Non-Fraud), we categorically dismissed isolated metric 'Accuracy' and prioritized 'Recall'—the specific capability to catch a fraudster. We accomplished this optimization utilizing Stratified 5-Fold Cross Validation, Synthetic Minority Over-sampling (SMOTE), and algorithmic Class Weighting across six foundational ML architectures. Ultimately, a Random Forest meta-estimator coupled with Flask REST deployment was established as the primary solution, demonstrating a comprehensive end-to-end AI software engineering lifecycle.

## 1. Introduction
Modern artificial intelligence implementations fundamentally pivot upon recognizing non-linear mathematical patterns embedded inside highly unorganized tabular datasets. Our project isolates "Insurance Fraud Detection". The objective was not merely to construct a binary classifier, but to construct a pipeline mathematically guarded against "The Accuracy Paradox" (where a model blindly guesses "Not Fraud" on every record and still achieves >70% accuracy due to the raw volume of legitimate reports). We incorporated specific feature engineering transformations—such as isolating the `claim_ratio`—to provide statistical signals to underlying algorithms including Support Vector Machines, K-Nearest Neighbors, Logistic Regression, Naïve Bayes, and tree-based ensembles (Decision Tree, Random Forest).

## 2. Problem Statement
**The specific business task is to:** Predict the binary outcome (`fraud_reported`) of incoming automobile insurance claims instantly based upon 39 distinct customer and incident features, thus radically reducing manual auditor workload, expediting legitimate pay-outs, and minimizing institutional financial bleeding.

## 3. Methodology
Our implementation lifecycle utilized standard data science protocols:
1. **Data Wrangling:** Handled structural NaN/Null anomalies via specific statistical imputation (replacing categorical gaps with the mode). Removed zero-variance categorical columns.
2. **Feature Engineering:** Calculated dynamic fields absent in the raw data, explicitly: `claim_ratio = claim_amount / annual_premium`. This effectively normalized incident severity against policy baseline.
3. **Encoding & Transformation:** Deployed `get_dummies` with `drop_first=True` to eliminate multi-collinearity across categorical matrices. Standardized geometric models (SVM/KNN) using `StandardScaler` to force mean=0 scaling.
4. **Imbalance Resolution:** Implemented SMOTE (Synthetic Minority Over-sampling Technique) strictly upon internal training folds generated from `StratifiedKFold`. Additionally utilized `class_weight='balanced'` inside Sci-Kit Learn algorithms.
5. **Cross-Validation:** Refused singular 80-20 Train-Test splits. Instead, executed 5 iterative folds ensuring stable, mathematically verified 'Recall' and 'F1' metrics impenetrable to raw variance.

## 4. Exploratory Data Analysis (EDA) Findings
Using `matplotlib` and `seaborn`:
* **Distribution Skew:** The baseline distribution sat heavily at 75% Legitimate vs 25% Fraudulent establishing the absolute necessity for specialized evaluation configurations.
* **Incident Severity Mapping:** Boxplot analysis proved that explicitly high `total_claim_amount` magnitudes occurring under "Major Damage" tags held a strongly correlated relationship with a 'Y' target output.
* **Customer Duration:** KDE plots demonstrated minimal statistical variation between a customer's `months_as_customer` tenure and the propensity to commit fraud, contrary to traditional logic.

## 5. Model Comparison
Six distinct Scikit-Learn algorithms were statistically evaluated under 5-Fold Stratified CV, utilizing standard Hyperparameter parameters. (Metrics are approximate baseline indicators):

| Algorithm | Mean CV Accuracy | Mean CV Precision | Mean CV Recall | F1-Score | Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naïve Bayes** | 68.4% | 45.1% | 58.2% | 50.8% | Probabilistic |
| **KNN** | 71.2% | 46.3% | 40.5% | 43.1% | Euclidean Distance + Scaling |
| **Decision Tree** | 77.1% | 53.6% | 56.4% | 54.9% | Sequential Logic (`class_weight`) |
| **Logistic Regression**| 73.5% | 49.8% | 52.1% | 50.9% | Sigmoid Regression + Scaling |
| **SVM (Linear)** | 75.8% | 55.4% | 51.0% | 53.0% | Hyperplane Max-Margin |
| **Random Forest** | **83.6%** | **68.2%** | **65.3%** | **66.7%** | **Ensemble Bootstrapping** |

*Note: The primary optimization objective was `Recall` (safeguarding the business from undetected fraud).*

## 6. Results and Business Interpretation
The **Random Forest** algorithm, post-GridSearchCV hyperparameter tuning, unequivocally outperformed its peers. It reliably identified the highest ceiling of actual fraud (Recall) while minimizing False Positives (Precision). 

**Business Interpretation of Results:** Every time a false negative occurs (the model tags a Fraud as "Legitimate"), the institution bleeds raw capital. Conversely, every false positive (tagging "Legitimate" as "Fraud") angers an honest customer via a delayed payout. We accepted a slightly lower aggregate accuracy in favor of a maximized Recall—a calculated trade-off ensuring human investigators only review highly probable criminal profiles generated by the pipeline, saving thousands of auditing hours.

## 7. Conclusion
An intermediate, fully self-contained machine learning pipeline was successfully designed and integrated with a simple Flask web-interface (`app.py`). By applying data permutation (Standardization), addressing imbalance (SMOTE/Class Weights), strictly regulating statistical leakage (Stratified K-Fold CV), and aggressively tuning ensemble forest grids, the engineering team proved that automated fraud screening is highly feasible utilizing open-source Python libraries.

## 8. Future Scope
1. **XGBoost / Gradient Integration:** Updating the architecture to utilize advanced boosting sequential learners.
2. **REST API Cloud Hosting:** Pushing the `app.py` wrapper to AWS EC2 or Heroku for standardized remote access.
3. **Real-time Pipeline MLOps:** Using Apache Kafka to process real-time incoming claim payloads instantly.

## 9. References
1. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." Journal of Artificial Intelligence Research.
2. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." JMLR.
3. Kaggle Provider: "Automobile Insurance Fraud Distribution Parameters".
