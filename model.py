# pip install flask

from flask import Flask, render_template, request
import numpy as np
import joblib
import os
app = Flask(__name__)

# Load the pre-trained model
path=os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(path,'used_cars_prediction.pkl'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from user
    yr = int(request.form['year'])
    price = int(request.form['price']) / 100000
    kms = int(request.form['kms'])
    ft = int(request.form['ft'])
    st = int(request.form['st'])
    trans = int(request.form['trans'])
    owner = int(request.form['owner'])

    # Create input data array
    input_data = np.array([[yr, price, kms, ft, st, trans, owner]])

    # Predict the price using the pre-trained model
    predicted_price = model.predict(input_data)[0]

    # Render the result page with predicted price
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)

# Jinja
