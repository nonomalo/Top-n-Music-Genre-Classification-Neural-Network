from __future__ import unicode_literals
import yt_dlp
import os
import librosa
import numpy as np
import re

# config information for processing:
STORED_AUDIO = 'audio/temp.wav'
SAMPLE_RATE = 22050
TRACK_DURATION = 30
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 1024


def process_track(file_path):
    """
    Creates an mfcc sequence for each audio file.
    :param file_path: file path to audio track (string)
    :param sample_info: dictionary of config settings for the processing,
    including: 'sample_rate', 'n_mfcc', 'n_fft', 'hop_length',
    and expected_mfcc_length
    :return: list of sequential mfccs for the audio file
    """
    expected_signal_length = SAMPLE_RATE * TRACK_DURATION

    try:

        # load audio file as floating point time series
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    except FileNotFoundError:

        print(f'File Not Found: {file_path}')
        return None

    # restrict signal length to maintain consistent shape along ndarrays
    if signal.shape[0] > expected_signal_length:
        signal = signal[:expected_signal_length]

    # apply short-term Fourier transform
    stft = np.abs(
        librosa.stft(
            signal,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH))

    # convert stft to spectrogram on mel-scale
    mel_spectrogram = librosa.feature.melspectrogram(
        S=stft ** 2,
        sr=SAMPLE_RATE)

    # apply discrete cosine transform
    db = librosa.power_to_db(mel_spectrogram)

    # generate mfcc for audio track (mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=N_MFCC)

    # delete the track from temporary storage
    os.remove(STORED_AUDIO)

    return mfcc


def download_wav_file(url):
    """
    Download and save the first 30 seconds of
    audio from the url in wav format
    :param url: music or music video url
    :return: error string if error, else None
    """

    # remove extension from STORED_AUDIO path
    store_as = os.path.splitext(STORED_AUDIO)[0]

    try:
        ydl_options = {
            'external_downloader': 'ffmpeg',
            'external_downloader_args': ['-to', '00:00:30'],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav'
            }],
            'outtmpl': store_as + '.%(ext)s'
        }

        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            ydl.download([url])
            song_info = ydl.extract_info(url, download=True)
            song_info.get('title', None)

    except yt_dlp.utils.DownloadError as e:
        # remove special string formatting:
        # see: https://stackoverflow.com/questions/30425105
        error = re.sub(r'\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))', '', str(e))
        return error

    return None
