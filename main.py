# flask, scikit-learn,pandas,pickle-mixin

import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

## Dipon De ##
app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    Bhk = sorted(data['bhk'].unique())
    Bath = sorted(data['bath'].unique())
    return render_template('index.html', locations=locations, Bhk=Bhk, Bath=Bath)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = request.form.get('total_sqft')

        if float(sqft) < 100:
            return " Square-fit is less than 100 not allowed!! "

        input_data = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_data)[0] * 1e5

        if prediction < 0:
            return "This configuration House Not Available "

        return "₹ " + str(np.round(prediction, 2))
    except:
        return "This configuration House Not Available"


if __name__ == "__main__":
    app.run()
