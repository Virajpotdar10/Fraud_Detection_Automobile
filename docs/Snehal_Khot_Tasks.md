# Project Tasks Assigned to Snehal Tanaji Khot

## EPIC: PREREQUISITES
**Task:** Software/Hardware Requirements and Project Setup

### Software Requirements
* **Operating System:** Windows 10/11, macOS, or Ubuntu Linux
* **Programming Environment:** Python 3.9+ installed and configured
* **IDE:** Jupyter Notebook for execution, VS Code for Flask editing
* **Web Framework:** Flask (for creating the local REST API and UI)

### Hardware Requirements
* **Processor:** Intel Core i5 / AMD Ryzen 5 or equivalent
* **RAM:** Minimum 8 GB (16 GB recommended for Cross-Validation and SMOTE processing)
* **Storage:** 20 GB of free SSD space

### Installation Steps
1. Download and install Python from `python.org`. Ensure "Add to PATH" is checked.
2. Open Terminal or Command Prompt in the desired project directory.
3. Create a clean virtual environment to prevent package conflicts:
   * `python -m venv venv`
4. Activate the virtual environment:
   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
5. Install the required libraries using pip:
   * `pip install -r requirements.txt`
6. Verify the installation by typing `python -c "import sklearn, imblearn; print('Installed successfully')"`

### Libraries List and Explanation
We use only foundational scientific computing libraries to keep our implementation academic but powerful:
1. **Pandas:** Used for reading the Kaggle CSV dataset and performing structured data manipulation (e.g., handling missing values, calculating custom features like claim ratios).
2. **NumPy:** Handles fast mathematical operations on multi-dimensional arrays, heavily relied upon internally by Scikit-Learn.
3. **Matplotlib & Seaborn:** Used for Exploratory Data Analysis (EDA). Matplotlib provides basic plotting, while Seaborn creates beautiful, statistical graphics like correlation heatmaps.
4. **Scikit-Learn:** The core machine learning engine for our project. It provides the models (Decision Tree, SVM, RF, etc.), Cross-Validation tools, Feature Scalers, and Evaluation Metrics.
5. **Imbalanced-learn (`imblearn`):** Essential for an intermediate-level fraud project. Used specifically for SMOTE (Synthetic Minority Over-sampling Technique) to balance out our target variable.
6. **Flask:** A lightweight web framework to serve our saved `.pkl` model as a functional web application.

### `requirements.txt` Content
```text
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
imbalanced-learn==0.11.0
Flask==2.3.2
```

---

## EPIC: PROJECT WORKFLOW
**Task:** Workflow Design and Architecture 

### Step-by-Step Workflow
1. **Data Acquisition:** Secure the raw Automobile Insurance Claim dataset from Kaggle.
2. **Data Wrangling:** Cleanse data by removing duplicates and imputing missing fields explicitly using statistical medians or domain logic.
3. **Feature Engineering:** Create new predictive variables (e.g., `claim_ratio`, `claim_frequency`) based on existing customer behaviors.
4. **EDA (Exploratory Data Analysis):** Run multivariate visualizations to interpret the structural relationships between independent variables and the fraud outcome.
5. **Preprocessing & Resampling:** Apply One-Hot Encoding and StandardScaler. Tackle class imbalance directly utilizing SMOTE to synthetically balance the training records.
6. **Model Formulation & Cross-Validation:** Train 6 different classification models securely using Stratified K-Fold Cross-Validation to ensure preventing data leakage and overfitting.
7. **Performance Tuning:** Optimize the highest-performing models using GridSearchCV focusing specifically on maximizing the Recall/F1-Score.
8. **Deployment:** Serialize the tuned model using Python's `pickle` library, and deploy it via a Flask web application.

### Text-Based Architecture Diagram
```text
[ Raw Data Source (Kaggle CSV) ]
        |
        v
[ Data Preparation Pipeline ]
  -> Duplicate Removal & Imputation
  -> Feature Engineering (claim_ratio)
        |
        v
[ Feature Transformation ]
  -> One-Hot Categorical Encoding
  -> StandardScaler (z-score normalization)
        |
        v
[ Target Resampling ]  <-- (CRITICAL FOR INTERMEDIATE AI)
  -> SMOTE (To fix class imbalance)
        |
        v
[ Model Training block ]
  -> Stratified 5-Fold Cross-Validation
  -> Algorithms: RF, SVM, LogReg, NB, KNN, DT
        |
        v
[ Model Evaluation ]
  -> Precision, Recall (Priority), F1-Score
  -> GridSearchCV Hyperparameter Tuning
        |
        v
[ Web Deployment (Flask) ]
  -> model.pkl integration
  -> Agent Input UI -> Prediction
```

### Explanation of Data Flow
The process begins with tabular data representing historical claim configurations. This unprocessed data flows into the preprocessing engine where missing inputs are statistically repaired. During Feature Engineering, raw numbers are converted into actionable business metrics (like the ratio of claim amount to annual premium). Next, categorical fields are converted to mathematical vectors (1s and 0s) and numerical fields are scaled. The most critical pivot is the Resampling loop, where SMOTE artificially balances the dataset so our models aren't biased towards the majority "Non-fraud" class. Finally, the data is pushed through 5-Fold Cross-Validation, ensuring the model's accuracy numbers are robust and reliable on unseen data.

---

## EPIC 3: EXPLORATORY DATA ANALYSIS
**Tasks:** Visual Analysis, Multivariate Analysis, and Scaling

*Note: All corresponding Python code is in `snehal_code.py`.*

### Story: Visual Analysis
1. **Fraud Distribution Plot:**
   * *Provide:* A Countplot showing `Fraud reported` vs `Non-Fraud`.
   * *Interpretation:* Fraud is inherently disproportionate. Usually, less than 25% of claims are fraudulent. This plot proves the existence of severe class imbalance, meaning we simply cannot rely on standard "Accuracy" as an evaluation metric.
2. **Claim Amount Comparison (Boxplot):**
   * *Provide:* A boxplot comparing Total Claim Amount based on the outcome.
   * *Interpretation:* The boxplot reveals whether the median claim amount is significantly higher for fraudulent cases. Outliers displayed here indicate overly aggressive claims, which are strong signals for model learning.
3. **Age vs Fraud:**
   * *Provide:* A KDE or stacked histogram showing the Age distribution for Fraud vs Non-Fraud.
   * *Interpretation:* We look for demographic patterns—e.g., if highly young drivers (inexperience) or older policyholders are statistically likelier to commit staging.
4. **Correlation Heatmap:**
   * *Provide:* A matrix of correlations for all numerical columns.
   * *Interpretation:* Shows multi-collinearity. If `vehicle_claim` and `total_claim_amount` have a correlation close to 1.0, they carry redundant information, which might negatively affect models like Logistic Regression.

### Story: Multivariate Analysis
* **Relationship between Multiple Features and Fraud:**
  * We use scatter plots mapped to 'fraud_reported' (e.g., Policy Annual Premium vs Total Claim Amount). 
  * *Observations:* We often observe clusters. When a user has a very low annual premium but an extraordinarily high sudden claim, the marker shifts heavily towards the "Fraud" category. This dynamic relationship directly motivates our feature engineering later.

### Story: Scaling
* **StandardScaler Example:** Data is scaled so it has a mean of 0 and standard deviation of 1.
* **Explain why scaling is important:** 
  Models like **SVM** and **KNN** are strictly distance-based algorithms; they measure the mathematical Euclidean distance between data points. If `claim_amount` is ₹5,00,000 and `age` is 30, the distance equation is completely dominated by the ₹5,00,000 figure, rendering `age` effectively useless to the algorithm. Scaling neutralizes this magnitude bias, ensuring the model respects the informational value of both parameters equally.

---

## EPIC 4: MODEL BUILDING
**Tasks:** Decision Tree, KNN (Training & Cross-Validation)

*(Full code and output for CV/metrics in `snehal_code.py`)*

For an intermediate-level approach, we do NOT train the model on a single 80-20 split once. We use **Stratified 5-Fold Cross Validation**. This cuts the data into 5 equal parts, tests on each part iteratively while training on the other 4, and averages the scores. This guarantees our reported metrics aren't just a "lucky" split. 

* **Short Interpretation for Decision Tree:** DT visually builds sequential rules natively separating fraud from non-fraud. The CV scores reveal it is slightly prone to overfitting but provides high transparency for the Business unit.
* **Short Interpretation for KNN:** By using distance measurement on our scaled data, KNN classifies a claim as fraud if its 5 mathematically closest historical cases were fraud. While conceptually elegant, its F1-Score generally trails behind tree-based models on complex tabular datasets.
