import requests
import json
from utils.create_plots import create_prediction_plot
from requests.exceptions import Timeout

GENRE_SERVER = 'https://top-n-server.uw.r.appspot.com'


def get_predictions(data, plots):
    # request genre predictions from server
    try:
        res = requests.post(
            GENRE_SERVER + '/genre',
            files={'audio': open(data['filename'], 'rb')},
            timeout=30
        )
        if res.status_code == 200:
            predict = create_prediction_plot(json.loads(res.content))
            plots.append(predict)
        else:
            print(res.text)
            data['error'] = res.text
    except Timeout:
        data['error'] = \
            'Genre Prediction Server was too slow so the request timed out'

    return data, plots
