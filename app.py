from flask import Flask, render_template, request, flash
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Check if model exists, if not, train it
if not os.path.exists('model/model.pkl'):
    print("Model not found. Training new model...")
    exec(open('save_model.py').read())

# Load the trained model
try:
    with open('model/model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get values from the form
            age = int(request.form['age'])
            sex = 1 if request.form['sex'] == 'male' else 0
            bmi = float(request.form['bmi'])
            children = int(request.form['children'])
            smoker = 1 if request.form['smoker'] == 'yes' else 0
            region = request.form['region']
            
            # Create region encoding
            region_northeast = 1 if region == 'northeast' else 0
            region_northwest = 1 if region == 'northwest' else 0
            region_southeast = 1 if region == 'southeast' else 0
            region_southwest = 1 if region == 'southwest' else 0
            
            # Make prediction
            features = np.array([[age, sex, bmi, children, smoker, 
                                region_northeast, region_northwest, 
                                region_southeast, region_southwest]])
            
            if model is not None:
                prediction = model.predict(features)[0]
                return render_template('result.html', 
                                     prediction=f"${prediction:,.2f}",
                                     age=age,
                                     sex=request.form['sex'],
                                     bmi=bmi,
                                     children=children,
                                     smoker=request.form['smoker'],
                                     region=region)
            else:
                return "Error: Model not loaded properly", 500
                
        except Exception as e:
            return f"Error processing request: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True) 