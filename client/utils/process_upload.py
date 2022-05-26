import os
import subprocess

from werkzeug.utils import secure_filename
from utils.fetch_audio import STORED_FILENAME

ACCEPTED_EXTENSIONS = {'wav', 'mp3', 'm4a'}


def is_allowed_file(filename):
    # Returns True if file extension is accepted format
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ACCEPTED_EXTENSIONS


def save_uploaded_file(uploaded_file, unique, audio_dir):
    """Saves upload file in audio_dir with unique filename

    :param uploaded_file: user's uploaded file
    :param unique: unique id
    :param audio_dir: file storage directory
    :return: dict with filepath, metadata, and errors
    """
    data = {}
    extension = uploaded_file.filename.rsplit('.', 1)[1].lower()

    if is_allowed_file(uploaded_file.filename):
        # secure the file before saving it
        secure_filename(uploaded_file.filename)

        # create storage filename and store file
        store_as = os.path.join(
            audio_dir,
            STORED_FILENAME + str(unique) + '.' + extension)
        uploaded_file.save(store_as)

        data['filename'], error = process_upload(store_as, extension)
        data['title'] = 'User\'s Track'

        if error:
            return {'error': error}
    else:
        return {'error': f"File type {extension} is not allowed"}

    return data


def convert_and_clip_audio(filename, ext):
    """Clips audio to 30 seconds and saves in wav format

    :param filename: filepath to audio file
    :param ext: file extension
    :return: new filepath and any errors
    """
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
    """Clips wav file to 30 seconds

    :param filename: filepath to audio file
    :return: new filepath and any errors
    """
    error = None
    clipped_name = filename.rsplit('.', 1)[0].lower() + '_.wav'
    command = f'ffmpeg -i {filename} -to 00:00:30 {clipped_name}'
    try:
        subprocess.call(command, shell=True)
        os.remove(filename)
    except Exception as err:
        error = err
    return clipped_name, error


def process_upload(filename, ext):
    """Runs wav conversion and/or clipping for audio file

    :param filename: filepath to audio file
    :param ext: audio file extension
    :return: new filepath and any errors
    """
    error = None
    if ext != 'wav':
        filename, error = convert_and_clip_audio(filename, ext)
    else:
        filename, error = clip_audio(filename)
    return filename, error
