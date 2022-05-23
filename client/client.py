from flask import Flask, request, render_template, redirect, url_for, jsonify
import uuid

from utils.fetch_audio import download_wav_file

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
@app.route('/fetch_audio', methods=['POST'])
def fetch_audio():
    audio_url = request.get_json().get('audio_url')
    print(audio_url)
    if audio_url != '':
        unique = uuid.uuid4()
        data = download_wav_file(audio_url, str(unique))
        print(data)
        return jsonify(data)
    else:
        return jsonify({'error': 'Url was empty'})


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
