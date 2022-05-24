from flask import Flask, request, render_template, redirect, url_for, jsonify
import uuid
import requests
import os
import json

from utils.fetch_audio import download_wav_file
from utils.create_plots import create_plots, create_prediction_plot

GENRE_SERVER = 'https://top-n-server.uw.r.appspot.com'

app = Flask(__name__)


# route to music upload page
@app.route('/')
def index():
    return render_template('index.html')


# route to upload file
@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return redirect(url_for('index'))


# route to fetch wav file from url
@app.route('/fetch_data', methods=['POST'])
def fetch_data():
    audio_url = request.get_json().get('audio_url')
    if audio_url != '':

        # create a unique id for the audio file
        unique = uuid.uuid4()
        data = download_wav_file(audio_url, str(unique))

        # create and save graph images for wav file
        plots = create_plots(data['filename'])
        data['plots'] = plots
        print('created track plots')

        # request genre predictions from server
        try:
            res = requests.post(
                GENRE_SERVER + '/genre',
                files={'audio': open(data['filename'], 'rb')},
                timeout=30
            )
            if res.status_code == 200:
                data['predictions'] = \
                    create_prediction_plot(json.loads(res.content))
            else:
                print(res.text)
        except Exception as err:
            print(err)

        # remove the audio file from storage
        try:
            os.remove(data['filename'])
        except Exception as err:
            print(err)

        return jsonify(data)


# route to about project page
@app.route('/about')
def about():
    return render_template('about.html')


# route to team information page
@app.route('/team')
def team():
    return render_template('team.html')


# route to 404 error
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html')


# route to server error
@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
