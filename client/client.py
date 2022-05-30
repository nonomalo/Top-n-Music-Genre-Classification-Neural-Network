from flask import Flask, request, render_template, redirect, url_for
import uuid
import os
import json

from utils.fetch_audio import download_wav_file
from utils.create_plots import create_plots
from utils.process_upload import save_uploaded_file
from utils.get_predictions import get_predictions

app = Flask(__name__)

# create directory to save files to
audio_dir = os.path.join(app.instance_path, 'audio')
os.makedirs(audio_dir, exist_ok=True)


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

    # create a unique id for the audio file
    unique = uuid.uuid4()

    if audio_url != '':
        data = download_wav_file(audio_url, str(unique), audio_dir)

    elif uploaded_file.filename != '':
        data = save_uploaded_file(uploaded_file, unique, audio_dir)

    else:
        return render_template(
            'index.html',
            data={'error': 'Please upload a file or submit a url.'})

    if 'error' in data:
        return render_template('index.html', data=data)

    # create and save graph images for wav file
    plots = create_plots(data['filename'])
    print('created track plots')

    # request predictions from server
    data, plots = get_predictions(data, plots)

    # remove the audio file from storage
    try:
        print(data['filename'])
        os.remove(data['filename'])
    except Exception as err:
        print(err)

    try:
        if data['error']:
            try:
                nested = json.loads(data['error'])
                if nested['error']:
                    data['error'] = nested['error']
            except Exception as err:
                print(err)
            return render_template('index.html', data=data)
    except KeyError:
        pass

    return render_template('dash.html',
                           data=data,
                           images=plots)


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
