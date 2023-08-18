from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for a home page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Location=request.form.get('Location'),
            Fuel_Type=request.form.get('Fuel_Type'),
            Transmission=request.form.get('Transmission'),
            Owner_Type=request.form.get('Owner_Type'),
            brand=request.form.get('brand'),
            Year=float(request.form.get('Year')),
            Kilometers_Driven=float(request.form.get('Kilometers_Driven')),
            Mileage=float(request.form.get('Mileage')),
            Engine=float(request.form.get('Engine')),
            Power=float(request.form.get('Power')),
            Seats=float(request.form.get('Seats'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=round(results[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
