# Project Tasks Assigned to Samruddhi Yadav

## EPIC 1: DEFINE PROBLEM / PROBLEM UNDERSTANDING
**Tasks:** Business Problem and Literature Survey

### Activity 1.1: Business Problem
* **Clear Business Problem Definition:** The automobile insurance sector is highly vulnerable to fraudulent claims—individuals deliberately staging accidents, exaggerating damages, or misrepresenting facts for financial gain. The business problem is that investigating all claims manually is incredibly slow, exceptionally costly, and human reviewers are prone to operational fatigue. The organization explicitly needs an automated predictive engine that can accurately tag high-risk claims instantly upon submission.
* **Real-world Fraud Statistics:** Globally, fraud accounts for approximately 10-15% of all property and casualty insurance claims, costing global insurers upwards of billions annually. 
* **Why Fraud Detection is Difficult:** Fraudsters constantly evolve their techniques. Because legitimate claims astronomically outnumber fraudulent ones (severe class imbalance), basic models often blindly guess "Not Fraud" for everything and achieve high "Accuracy" while entirely failing to catch the actual criminals. It is mathematically very nuanced to separate a strange genuine claim from a cleverly structured fake one.

### Activity 1.3: Literature Survey
**Summarized Research Approaches:**
1. *Rule-Based Expert Systems:* Early academic papers focused on static algorithms that flagged claims if they met specific physical parameters (e.g., claiming over ₹10 Lakhs at 3 AM). Highly interpretable but not adaptive.
2. *Supervised Machine Learning (Trees):* Literature utilizing Decision Trees and Random Forests proved that machine learning could identify the non-linear interaction rules between features (e.g., policy age vs incident severity) that human engineers couldn't spot.
3. *Distance & Probabilistic Approaches (SVM, NB):* Researches utilizing Support Vector Machines mapping data onto hyperplanes showed strong capability when standard statistical assumptions were met, although highly computationally intensive.
4. *Synthetic Resampling Methodologies:* Modern peer-reviewed studies (Chawla et al.) established that applying algorithmic SMOTE (Synthetic Minority Over-sampling Technique) to insurance data heavily increases the Recall rate (catching the fraud) by artificially generating balanced sample cases for the model to train exclusively on.

**Traditional vs ML-Based Detection Comparison:**
Traditional detection uses hard-coded rules ("IF claim_amount > limit THEN flag"). This leads to extraordinarily high false positive rates and immediate obsolescence when fraudsters alter their behavior. ML-based detection analyzes historical data vectors simultaneously to detect complex, invisible patterns and dynamically shifts its decision boundaries as new historical data is pipelined in.

**Gap in Existing Systems:**
Current industry solutions fail when confronted with highly imbalanced class distributions without strict mathematical intervention. They largely prioritize 'Accuracy' rather than penalizing False Negatives. Our project mitigates this gap utilizing advanced Feature Engineering and resampling techniques (SMOTE / Class Weighting).

---

## EPIC 2: DATA COLLECTION & PREPARATION
**Task:** Dataset Collection

### Activity 1: Dataset Collection
* **Dataset Source:** We sourced the benchmark Automobile Insurance Fraud dataset from Kaggle.
* **Feature List:** The dataset provides high-fidelity attributes divided into demographic data (Age, Gender, Education), policy data (Annual Premium, Umbrella Limit), and specific incident data (Incident Type, Collision Type, Vehicles Involved, Authorities Contacted).
* **Target Variable Explanation:** The dependent variable is `fraud_reported`. This dictates our problem as a Supervised Binary Classification task (1 = Fraud, 0 = Non-Fraud).
* **Class Imbalance Discussion:** A critical observation within our dataset is an approximate 75:25 ratio of non-fraud to fraud. If ignored, the ML algorithms converge to a local minimum by ignoring the minority class entirely. To address this, we integrated methodologies specifically tailored to address imbalance, including Stratified K-Fold splitting and algorithm-level class weights.

---

## EPIC 3: EXPLORATORY DATA ANALYSIS
**Task:** Encoding Categorical Features

*(Code provided in `samruddhi_code.py`)*

### Encoding Categorical Features
* **Label Encoding vs One-Hot Encoding Explanation:**
  * *Label Encoding* replaces categorical names with integers (e.g., City -> 0, Suburb -> 1, Rural -> 2). This intrinsically introduces a hierarchical mathematical rank (2 is greater than 0) where none actually exists in reality, confusing models. 
  * *One-Hot Encoding* creates a strict binary (1/0) vector column for every distinct category. "Is_City", "Is_Suburb", "Is_Rural". This assumes no ordinal hierarchy.
* **Justify Chosen Method:** For our intermediate-level implementation utilizing regression and distance architectures (SVM, LR), we categorically utilized One-Hot Encoding (via `pd.get_dummies`) dropping the first column to avoid the 'dummy variable trap' (perfect multi-collinearity), ensuring robust mathematical integrity across all tested models.

---

## EPIC 4: MODEL BUILDING
**Task:** Random Forest (Training & CV)

*(Code provided in `samruddhi_code.py`)*

For Random Forest, we apply 5-Fold Stratified Cross Validation. 
* **Short Interpretation:** Random Forest acts as a wisdom-of-the-crowd ensemble model. By using `class_weight='balanced'`, it mathematically penalizes the trees heavily every time they incorrectly misclassify a fraudulent claim. It natively handles non-linear relationships and interactions among our newly engineered features.

---

## EPIC 7: PROJECT DEMONSTRATION & DOCUMENTATION
**Task:** Project Documentation

### Activity 2: Project Report 
*(Please see the generated `Final_Project_Report.md` in the `docs/` folder for the fully compiled and structured final academic document containing the Abstract, Methodology, and complete metrics.)*
