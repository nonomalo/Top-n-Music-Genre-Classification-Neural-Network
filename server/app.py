from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from utils.process import process_track
from model.dataset import load_mappings, preprocess_inputs
from model.evaluate import evaluate_model_max

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
    mel_array, error = process_track(audio_dir + '/' + filename)
    if error:
        return jsonify({'error': error}), 400
    if mel_array is None:
        return jsonify({
            'error': 'Unable to extract data from audio file'
        }), 400

    # run the processed data through the prediction model
    model = tf.keras.models.load_model('model/model_3.h5')
    inputs = preprocess_inputs(mel_array)
    mappings = load_mappings()
    json_dict = evaluate_model_max(model, inputs, mappings)

    return jsonify(json_dict), 200


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
