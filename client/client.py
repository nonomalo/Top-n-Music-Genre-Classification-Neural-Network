from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import uuid
import requests
import os
import json

from utils.fetch_audio import download_wav_file, STORED_AUDIO
from utils.create_plots import create_plots, create_prediction_plot
from utils.process_upload import process_upload

GENRE_SERVER = 'https://top-n-server.uw.r.appspot.com'

app = Flask(__name__)

ACCEPTED_EXTENSIONS = {'wav', 'mp3', 'm4a'}


def is_allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ACCEPTED_EXTENSIONS

# route to music upload page
@app.route('/')
def index():
    return render_template('index.html', data={})


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
    audio_url = request.form.get('text')
    uploaded_file = request.files['file']
    data = {}

    # create a unique id for the audio file
    unique = uuid.uuid4()

    if audio_url != '':

        data = download_wav_file(audio_url, str(unique))

    elif uploaded_file.filename != '':

        # get extension
        extension = uploaded_file.filename.rsplit('.', 1)[1].lower()
        print(extension)

        if is_allowed_file(uploaded_file.filename):
            # secure the file before saving it
            filename = secure_filename(uploaded_file.filename)

            # save file and create filename
            store_as = os.path.splitext(STORED_AUDIO)[0] + str(unique) + extension
            uploaded_file.save(store_as)

            data['filename'], error = process_upload(store_as, extension)
            data['title'] = 'User\'s Track'

            if error:
                return render_template('index.html', data={'error': error})

        else:
            data['error'] = \
                f"File type {extension} is not allowed"
            return render_template('index.html', data=data)

    # create and save graph images for wav file
    plots = create_plots(data['filename'])
    print('created track plots')

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
    except Exception as err:
        print(err)

    # remove the audio file from storage
    try:
        os.remove(data['filename'])
    except Exception as err:
        print(err)

    return render_template('dash.html',
                           data=data,
                           images=plots
                           )

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
