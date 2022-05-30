import os
import subprocess
from pathlib import Path

from werkzeug.utils import secure_filename
from utils.fetch_audio import STORED_FILENAME

ACCEPTED_EXTENSIONS = {'wav', 'mp3', 'm4a'}


def is_allowed_file(filename):
    # Returns True if file extension is accepted format
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ACCEPTED_EXTENSIONS


def save_uploaded_file(uploaded_file, unique, audio_dir, user_file=False):
    """Saves upload file in audio_dir with unique filename

    :param uploaded_file: user's uploaded file
    :param unique: unique id
    :param audio_dir: file storage directory
    :param user_file: True if file was uploaded directly
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

        # grab metadata from user's track
        if user_file:
            meta, error = extract_metadata(store_as)
            if error:
                return {'error': error}
            else:
                data = parse_metadata_to_dict(meta)

        data['filename'], error = process_upload(store_as, extension)

        if error:
            return {'error': error}
    else:
        return {'error': f"File type {extension} is not allowed"}
    return data


def parse_metadata_to_dict(filename):
    """Parses track metadata text into a dictionary

    :param filename: audio filepath
    :return: dictionary with artist, track, and album
    """
    # read the file text as list of lines
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    metadata_dict = {}

    # save track title, artist, and album name to dict
    for line in lines:
        parse_list = line.split('=', 1)
        if len(parse_list) == 2:
            if parse_list[0] in ['title']:
                metadata_dict['track'] = parse_list[1]
            if parse_list[0] in ['artist', 'album']:
                metadata_dict[parse_list[0]] = parse_list[1]
    if 'track' not in metadata_dict:
        metadata_dict['title'] = 'User\'s Track'

    try:
        os.remove(filename)
    except Exception as err:
        print(err)

    return metadata_dict


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

    # name in which to save clipped audio
    clipped_name = filename.rsplit('.', 1)[0].lower() + '_.wav'

    command = f'ffmpeg -i {filename} -to 00:00:30 {clipped_name}'
    try:
        subprocess.call(command, shell=True)
        os.remove(filename)
    except Exception as err:
        error = err
    return clipped_name, error


def extract_metadata(filename):
    """Writes audio metadata to a new file

    :param filename: audio filepath
    :return: new metadata filepath, os errors
    """
    error = None

    # filepath for metadata
    meta = filename.rsplit('.', 1)[0].lower() + '.txt'

    command = f'ffmpeg -i {filename} -f ffmetadata {meta}'
    try:
        subprocess.call(command, shell=True)
    except Exception as err:
        error = err

    # errors are not returned from FFmpeg,
    # but it will not create a metadata file from non-audio
    new_file = Path(meta)
    if not new_file.exists():
        error = 'Unable to extract audio data from file'

    return meta, error


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
