from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the Stratified, SMOTE-balanced, and GridSearch optimized ML Model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Intermediate 'model.pkl' artifact successfully loaded!")
except FileNotFoundError:
    print("CRITICAL EXCEPTION: 'model.pkl' not discovered. Ensure you execute 'python train.py' sequentially first.")
    model = None

@app.route('/')
def home():
    # Render the engineered intermediate UI
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model Initialization Failed: model.pkl missing.", 500

    if request.method == 'POST':
        try:
            # Reconstruct Features extracted from HTTP Form Payload
            total_claim_amount = float(request.form['total_claim_amount'])
            policy_annual_premium = float(request.form['policy_annual_premium'])
            age = int(request.form['age'])
            
            # [CRITICAL] Dynamically applying the Feature Engineering deployed in training
            # The model was structurally trained using 'claim_ratio'. The API must mirror this.
            claim_ratio = total_claim_amount / (policy_annual_premium + 0.01)
            
            # Rebuilding mathematical 2D matrix structure:
            # [total_claim_amount, policy_annual_premium, age, claim_ratio]
            input_vector = np.array([[total_claim_amount, policy_annual_premium, age, claim_ratio]])
            
            # Predict
            prediction = model.predict(input_vector)
            
            # Formatting Response
            if prediction[0] == 1:
                result_text = "RISK ALERT: Statistical Fraud Threshold Exceeded. Flagged for Audit."
                color_class = "danger"
            else:
                result_text = "STATUS: Standard Variance. Claim Probability Legitimate."
                color_class = "success"

            return render_template('index.html', prediction_text=result_text, color=color_class)
            
        except Exception as e:
            return f"Deployment Server Traceback Error: {str(e)}"

if __name__ == '__main__':
    # Flask Default Server Binding
    app.run(debug=True, port=5000)
