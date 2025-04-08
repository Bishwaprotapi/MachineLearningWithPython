import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load your data (update path to match your insurance.csv location)
df = pd.read_csv('Sessions/Day12/insurance.csv')  # Updated path

# Prepare the data (same preprocessing as in your notebook)
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df = pd.get_dummies(df, columns=['region'])

X = df.drop('charges', axis=1)
y = df['charges']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file) 