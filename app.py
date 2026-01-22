from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open("model/house_price_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form inputs
        overallqual = float(request.form['overallqual'])
        grlivarea = float(request.form['grlivarea'])
        totalbsmtsf = float(request.form['totalbsmtsf'])
        garagecars = float(request.form['garagecars'])
        fullbath = float(request.form['fullbath'])
        yearbuilt = float(request.form['yearbuilt'])

        # Prepare data for model
        features = np.array([[overallqual, grlivarea, totalbsmtsf, garagecars, fullbath, yearbuilt]])
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)[0]
        price = round(prediction, 2)

        return render_template('index.html', prediction_text=f"Predicted House Price: ${price}")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
