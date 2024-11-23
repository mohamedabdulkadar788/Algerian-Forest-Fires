import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained regression model and scaler
regressor_model = pickle.load(open('models/ridgecls.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Get form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale the input data
            new_data_scaled = standard_scaler.transform(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            )

            # Predict using the loaded model
            result = regressor_model.predict(new_data_scaled)

            # Render result
            return render_template('home.html', result=result[0])

        except Exception as e:
            # Handle errors gracefully
            return render_template('home.html', result=f"Error: {str(e)}")

    # Render the form page
    return render_template('home.html')

# Run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0")
