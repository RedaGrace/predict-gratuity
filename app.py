from flask import *
import pandas as pd
import numpy as np
import json, time
import pickle
import re
import os
from flask_cors import CORS

import warnings
warnings.filterwarnings('ignore')

model_name = 'rf-model'
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
rf = read_pickle(model_name)

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

    inputs = np.array(inputs).reshape(1, -1)
    

    output = int(rf.predict(inputs)[0])
    proba = float(rf.predict_proba(inputs)[0].max())
    proba *= 100
    proba = int(proba)

    if output==1 and proba>=70:
        response = "Based on our predictive analysis, \
        there is a high likelihood of {}% that this customer will be generous. \
        Generous customer means the customer will leave a tip more than or equal to 20% of the fare.".format(proba)
    
    elif output==1 and proba<70:
        response = "Based on our predictive analysis, \
        there is a likelihood of {}% that this customer will be generous. \
        Generous customer means the customer will leave a tip more than or equal to 20% of the fare.".format(proba)
    
    elif output==0 and proba>=70:
        response = "Based on our predictive analysis, \
        there is a high likelihood of {}% that this customer will not be generous. \
        Not generous customer means the customer will leave a tip less than 20% of the fare or they will not leave a tip.".format(proba)
    elif output==0 and proba<70:
        response = "Based on our predictive analysis, \
        there is a likelihood of {}% that this customer will not be generous. \
        Not generous customer means the customer will leave a tip less than 20% of the fare or they will not leave a tip.".format(proba)
   
                     
    json_dump = json.dumps(response)
    return json_dump

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
