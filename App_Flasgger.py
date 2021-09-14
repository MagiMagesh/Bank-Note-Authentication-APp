## Can be used for POSTMAN

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)


pickle_in = open('model.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def Welcome():
    return "Welcome to the  Magesh's Bank Note Authentication App"

@app.route('/predict')
def predict_bank_Authentication():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return  'The prediction for your dataset is' + str(prediction)


@app.route('/predict_file',methods = ["POST"])
def predict_test_file():
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """

    test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(test)
    return 'The prediction for the test FIle is' + str(list(prediction))



if __name__=='__main__':
    app.run()