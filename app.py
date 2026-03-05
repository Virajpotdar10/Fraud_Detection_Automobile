from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = os.path.join('models', 'model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')

try:
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
except Exception as e:
    print("Warning: Model or scaler not found. Please train the model first.")
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    result_class = ""
    
    if request.method == 'POST':
        if not model or not scaler:
            return render_template('index.html', prediction_text="Error: Model not loaded.", result_class="error")
            
        try:
            age = float(request.form['age'])
            months_as_customer = float(request.form['months_as_customer'])
            annual_premium = float(request.form['annual_premium'])
            total_claim_amount = float(request.form['total_claim_amount'])
            vehicles = float(request.form['vehicles'])
            witnesses = float(request.form['witnesses'])
            injury_claim = float(request.form['injury_claim'])
            property_claim = float(request.form['property_claim'])
            vehicle_claim = float(request.form['vehicle_claim'])
            incident_hour = float(request.form['incident_hour'])
            
            input_features = np.array([[
                age, months_as_customer, annual_premium, total_claim_amount, 
                vehicles, witnesses, injury_claim, property_claim, vehicle_claim, incident_hour
            ]])
            
            input_scaled = scaler.transform(input_features)
            
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)[0]
            prob_fraud = round(probabilities[1] * 100, 2)
            prob_genuine = round(probabilities[0] * 100, 2)
            
            if prediction[0] == 1:
                prediction_text = f"FRAUDULENT CLAIM DETECTED! ({prob_fraud}% probability)"
                result_class = "fraud"
            else:
                prediction_text = f"GENUINE CLAIM. ({prob_genuine}% probability)"
                result_class = "genuine"
                
        except Exception as e:
            prediction_text = f"Error in processing inputs: {str(e)}"
            result_class = "error"
            
    return render_template('index.html', prediction_text=prediction_text, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)
