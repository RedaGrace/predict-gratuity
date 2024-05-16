from flask import *
import pandas as pd
import json, time
import pickle
import re
import os
from flask_cors import CORS

import warnings
warnings.filterwarnings('ignore')

model_name = 'hr_rf2'
def read_pickle(saved_model_name:str):
    '''
    In: 
        path:             path to folder where you want to read from
        saved_model_name: filename of pickled model you want to read in

    Out: 
        model: the pickled model 
    '''
    with open(saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

    return model
  
# Read in pickle
rf2 = read_pickle(model_name)

app = Flask(__name__)
CORS(app)

@app.route('/', methods = ['GET'])
def home_page():
    data_set = {'Page': 'Home', 'Message': "Let's get started and send me your inputs", 'Timestamp': time.time()}
    json_dump = json.dumps(data_set)
    return json_dump

@app.route('/predict/', methods=['GET'])
def request_page():
    inputs = request.args.get('inputs')                              # /predict/?inputs=inputs
    inputs= [float(n) for n in re.findall('[-+]?(?:\d*\.*\d+)', inputs)]
    

    output = int(rf2.predict([inputs])[0])
    proba = float(rf2.predict_proba([inputs])[0].max())
    proba *= 100
    proba = int(proba)

    #1- Positive Prediction with High Probability
    if output==1 and proba>=75:
        response = 'Based on our predictive analysis, \
        there is a high likelihood of {}% that the employee will continue with the company. \
        Their recent performance improvements and \
        engagement in team activities suggest a strong commitment to their role.'.format(proba)

    #2- Positive Prediction with Moderate Probability
    elif output==1 and proba<75:
        response = 'The prediction model indicates a {}% chance that the employee \
        will remain with the organization. While there are positive indicators of job satisfaction, \
        we recommend further engagement to solidify their retention.'.format(proba)

    #3- Negative Prediction with High Probability
    elif output==0 and proba>=75:
        response = 'Our analysis shows an {}% probability that the employee may leave the company.\
        Factors such as a lack of recent promotions and a high workload have contributed to this prediction. \
        Immediate action is advised to address these concerns.'.format(proba)

    #4- Negative Prediction with Moderate Probability
    elif output==0 and proba<75:
        response = 'There is a {}% chance that the employee is considering leaving. \
        This is primarily due to reported dissatisfaction with career progression opportunities. \
        We suggest a review of their development plan to mitigate this risk.'.format(proba)
    elif  proba < 55 and proba >=50:
        response = "The model presents a balanced view with a {}% probability of the employee \
        leaving or staying. It appears that the employee's decision may be influenced by \
        upcoming changes in their department. Close monitoring over the next quarter is recommended.".format(proba)
    
    
    

   
                     
    json_dump = json.dumps(response)
    return json_dump

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
