# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function
import os
import json
import flask
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing import sequence

from awscoreml.resolve import paths
from awscoreml.train import preprocess_tweet


class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        """This class method just checks if the model path is available to us"""

        if os.path.exists(paths.model('model.h5')):
            cls.model = True
        else:
            cls.model = None

        return cls.model


app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():

    health = ScoringService.get_model() is not None

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """This method reads in the data (json object) sent with the request and returns a prediction
    as response """

    data = None

    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)

        model = load_model(paths.model('model.h5'))

        with open(paths.model('tokenizer.pickle'), 'rb') as handle:
            tk = pickle.load(handle)

        one_tweet = preprocess_tweet(data['data'])
        one_tweet = np.array([one_tweet])
        t = tk.texts_to_sequences(one_tweet)
        X_test = np.array(sequence.pad_sequences(t, maxlen=20, padding='post'))

        prediction = model.predict(X_test)
        result = {"prediction": str(prediction[0][0])}

    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')
    print(data)
    print(type(data))
    return flask.Response(response=json.dumps(result), status=200, mimetype='application/json')
