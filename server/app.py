from flask import Flask, request, jsonify, render_template, redirect, url_for
import math
from process import download_wav_file, process_track
from process import STORED_AUDIO
from process import N_MFCC, SAMPLE_RATE, TRACK_DURATION, HOP_LENGTH

app = Flask(__name__)


@app.route('/genre', methods=['POST'])
def predict_genre():
    content = request.json
    expected = ['url']

    # check for url in content
    for name in content:
        if name not in expected:
            return jsonify({
                'error': f'Unexpected field: {name}'
            }), 400

    # check for extraneous json fields
    for name in expected:
        if name not in content:
            return jsonify({
                'error': f'Missing field: {name}'
            }), 400

    # get the wav file
    url = content['url']
    error = download_wav_file(url)

    if error:
        error = f'{error}'
        return jsonify({'error': error}), 400

    # get the data from processed wav file
    mfccs = process_track(STORED_AUDIO)
    if mfccs is None:
        return jsonify({
            'error': 'Unable to extract data from audio file'
        }), 400

    # check that data shape is correct
    expected_mfcc_length = math.ceil(
        (SAMPLE_RATE * TRACK_DURATION) / HOP_LENGTH
    )
    if mfccs.shape != (N_MFCC, expected_mfcc_length):
        return jsonify({
            'error': f'MFCC shape error: track from \
            {url} may not be long enough'
        }), 422

    # run the processed data through the prediction model
    return jsonify({
        'genres': 'THIS IS WHERE WE WOULD RUN THE MODEL'
    }), 200


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
