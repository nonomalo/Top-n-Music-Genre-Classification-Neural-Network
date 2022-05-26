import os
import subprocess

from werkzeug.utils import secure_filename
from utils.fetch_audio import STORED_AUDIO

ACCEPTED_EXTENSIONS = {'wav', 'mp3', 'm4a'}


def is_allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ACCEPTED_EXTENSIONS


def save_uploaded_file(uploaded_file, unique):

    data = {}
    extension = uploaded_file.filename.rsplit('.', 1)[1].lower()

    if is_allowed_file(uploaded_file.filename):
        # secure the file before saving it
        secure_filename(uploaded_file.filename)

        # create storage filename and store file
        store_as = os.path.splitext(STORED_AUDIO)[0] \
            + str(unique) + '.' + extension
        uploaded_file.save(store_as)

        data['filename'], error = process_upload(store_as, extension)
        data['title'] = 'User\'s Track'

        if error:
            return {'error': error}
    else:
        return {'error': f"File type {extension} is not allowed"}

    return data


def convert_and_clip_audio(filename, ext):
    error = None
    conversion_name = filename.rsplit('.', 1)[0].lower() + '.wav'
    command = f'ffmpeg -i {filename} -to 00:00:30 {conversion_name}'
    try:
        subprocess.call(command, shell=True)
        os.remove(filename)
    except Exception as err:
        error = err
    return conversion_name, error


def clip_audio(filename):
    error = None
    command = f'ffmpeg -i {filename} -to 00:00:30 {filename}'
    try:
        subprocess.call(command, shell=True)
    except Exception as err:
        error = err
    return filename, error


def process_upload(filename, ext):
    error = None
    if ext != 'wav':
        filename, error = convert_and_clip_audio(filename, ext)
    else:
        filename, error = clip_audio(filename)
    return filename, error
