# Project Tasks Assigned to Viraj Potdar

## EPIC 2: DATA COLLECTION & PREPARATION
**Task:** Data Preparation & Intermediate Feature Engineering

*Note: Code in `viraj_code.py`.*

### Data Preparation Steps (Intermediate ML Level)
1. **Loading Dataset:** Utilize Pandas for robust DataFrame ingestion from CSV format.
2. **Handling Missing Values & Duplicates:** Explicitly scan for `df.duplicated().sum()` and drop exact duplicates. Missing numeric entities are imputed using `df.median()` rather than `mode`, as means are violently skewed by outliers in finance data.
3. **Target Transformation:** Binaries ('Y', 'N') mathematically cast to integer matrices (1, 0).
4. **Intermediate Feature Engineering:** 
   * `claim_ratio = claim_amount / annual_premium`: A custom domain metric calculating if a user is trying to extract drastically more money than they actually pay for the policy, a highly volatile indicator for fraud.
   * `claim_frequency`: (Derived if past timeline data exists). 
5. **Stratified Splitting & Imbalance Control (SMOTE):** 
   Using simply `train_test_split` on a 75:25 imbalanced set guarantees terrible bias. We use `stratify=y` to lock the 75:25 ratio exactly in both Train and Test. Furthermore, heavily unbalanced training sets require synthetic adjustment using `SMOTE` specifically and ONLY on the training vector.

---

## EPIC 4: MODEL BUILDING
**Tasks:** Logistic Regression and SVM (with Stratified K-Fold CV)

*Note: Code in `viraj_code.py`.*

### Story: Logistic Regression
Logistic regression serves as the fundamental intermediate linear baseline. It utilizes a sigmoid activation function bounding raw linear summation between 0 and 1. We apply `class_weight='balanced'` to explicitly force the gradient descent logarithm to heavily penalize errors occurring on the minority 'Fraud' class.

### Story: SVM (Support Vector Machine)
SVM geometrically targets the maximum margin separating hyperplane between Fraud and Non-Fraud vectors in an N-Dimensional space. Because it calculates spatial distance vectors natively, providing it the `StandardScaled` features from EDA Epic 3 is structurally mandatory to ensure convergence and prevent extreme loss calculations.

---

## EPIC 5: PERFORMANCE TESTING & HYPERPARAMETER TUNING
**Task:** GridSearchCV

### Hyperparameter Tuning (Intermediate)
* **GridSearch Example for Random Forest:** Rather than arbitrarily guessing configuration integers, we deploy `GridSearchCV`. We establish a mathematical tensor dictionary (`param_grid`) featuring varied `n_estimators` (50, 100, 200) and `max_depth` structures (None, 10, 20). 
* **Tuning Impact:** The algorithm systematically tests every possible permutation using 3-Fold Cross-Validation, returning the absolute mathematical peak combination.
* **Before and After:** Before tuning, the RF model's trees might overfit (growing too deep) causing high variance and dropping blind recall to ~59%. Tuning restricts depth, standardizing the trees, which generally pushes blind F1-Score and Recall metrics upward toward ~65-70%.

---

## EPIC 6: MODEL DEPLOYMENT (Flask API)
**Tasks:** Code to save the best model, Flask structured app, steps to run.

*Note: See `app.py` and `train.py` for actual Python implementations.*

### Saving the Best Model
Once optimized via GridSearch, we serialize the best Python estimator object using the `pickle` library, generating an independent `model.pkl` binary artifact format.

### Clean Project Folder Structure
This framework separates execution contexts, standardizing deployment operations.
```text
Vehicle_Fraud/
├── src/                    # Internal Engineering modules (model training algorithms)
│   ├── viraj_code.py
│   ├── snehal_code.py
│   └── (etc)
├── docs/                   # Academic Reports, Workflows, Demonstration scripts
│   └── Final_Project_Report.md
├── templates/              # Jinja2 / HTML templates for the Web Interface
│   └── index.html
├── requirements.txt        # Virtual Environment dependencies
├── train.py                # Pipeline generator -> Feature Engineering -> SMOTE -> Model Dump
├── app.py                  # The Flask Application serving predictions
└── model.pkl               # Serialized Predictive Artifact
```

### Steps to Run the Project locally
**Step 1:** Open Terminal (`CMD` or `PowerShell`) inside `C:\Users\Viraj Potdar\Desktop\Vehicle_Fraud`.
**Step 2:** Ensure virtual environment active, run: `pip install -r requirements.txt`.
**Step 3:** Generate the artifact. Run: `python train.py`. *(This executes feature engineering, scales data, balances classes automatically, builds the model, and exports `model.pkl`.)*
**Step 4:** Launch the operational web gateway. Run: `python app.py`.
**Step 5:** Access the interface by navigating via Chrome/Edge to `http://127.0.0.1:5000/`.
