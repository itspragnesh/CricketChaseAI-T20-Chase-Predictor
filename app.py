from flask import Flask, request, render_template
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained scaler and ANN model
scaler = pickle.load(open('scaler.pkl', 'rb'))
ann_model = tf.keras.models.load_model('ann_model.h5')

# Define feature columns (same as in the original code)
feature_cols = [
    'Runs From Ball', 'Innings Runs', 'Innings Wickets', 'Balls Remaining',
    'Target Score', 'Total Batter Runs', 'Total Non Striker Runs',
    'Batter Balls Faced', 'Non Striker Balls Faced'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = []
        for feature in feature_cols:
            value = float(request.form[feature.replace(' ', '_').lower()])
            input_data.append(value)
        
        # Convert to numpy array and reshape for scaling
        input_array = np.array(input_data).reshape(1, -1)
        
        # Scale the input data
        scaled_input = scaler.transform(input_array)
        
        # Make prediction
        prediction = ann_model.predict(scaled_input)
        result = 'Chase Successful' if prediction[0][0] > 0.5 else 'Chase Unsuccessful'
        probability = float(prediction[0][0]) * 100
        
        return render_template('result.html', prediction=result, probability=probability)
    
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)