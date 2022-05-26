import io
import os
import subprocess

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