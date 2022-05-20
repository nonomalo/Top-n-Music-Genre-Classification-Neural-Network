from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import math
import tensorflow as tf
from utils.process import process_track
from utils.process import SAMPLE_RATE, TRACK_DURATION, HOP_LENGTH, MEL_BINS
from model.dataset import load_mappings, preprocess_inputs
from model.evaluate import evaluate_model

app = Flask(__name__)

# Create a directory in a known location to save files to.
audio_dir = os.path.join('audio')
if not os.path.isdir(audio_dir):
    os.mkdir(audio_dir)

ACCEPTED_EXTENSIONS = {'wav'}


def is_allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ACCEPTED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html', message='Server is running...')


@app.route('/genre', methods=['POST'])
def predict_genre():
    if 'audio' not in request.files:
        print(request.files)
        return jsonify({
            'error': 'Expecting audio file, but no file found'
        }), 406

    # get file and check for errors
    audio_file = request.files['audio']

    if not audio_file:
        return jsonify({
            'error': 'Missing audio file'
        }), 406

    if not is_allowed_file(audio_file.filename):
        ext = audio_file.filename.rsplit('.', 1)[1].lower()
        return jsonify({
            'error': f'File type {ext} not accepted'
        }), 400

    filename = secure_filename(audio_file.filename)
    audio_file.save(os.path.join(audio_dir, filename))

    # get the data from processed wav file
    mel_norm = process_track(audio_dir + '/' + filename)
    if mel_norm is None:
        return jsonify({
            'error': 'Unable to extract data from audio file'
        }), 400

    # check that data shape is correct
    expected_melspec_length = math.ceil(
        (SAMPLE_RATE * TRACK_DURATION) / HOP_LENGTH
    )
    if mel_norm.shape != (MEL_BINS, expected_melspec_length):
        return jsonify({
            'error': 'MFCC shape error: track  \
            may not be long enough'
        }), 422

    # run the processed data through the prediction model
    model = tf.keras.models.load_model('model/model.h5')
    inputs = preprocess_inputs([mel_norm])
    mappings = load_mappings()
    json_dict = evaluate_model(model, inputs, mappings, 1)

    return jsonify(json_dict), 200


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
