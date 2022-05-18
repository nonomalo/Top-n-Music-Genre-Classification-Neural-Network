from __future__ import unicode_literals
import os
import librosa
import numpy as np

# config information for processing:
SAMPLE_RATE = 22050
TRACK_DURATION = 30
N_FFT = 2048
HOP_LENGTH = 1024
MEL_BINS = 128


def process_track(file_path):
    """Creates an mfcc sequence for each audio file.

    :param file_path: file path to audio track (string)
    :return: normalized melspectrogram for the audio file
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

    # normalize waveform
    signal_norm = librosa.util.normalize(signal)

    # apply short-term Fourier transform
    stft = np.abs(
        librosa.stft(
            signal_norm,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH))

    # convert to normalized mel spectrogram
    mel = librosa.feature.melspectrogram(S=stft ** 2, sr=sr)
    mel_log = np.log(mel + 1e-9)
    mel_norm = librosa.util.normalize(mel_log)

    # delete the track from temporary storage
    os.remove(file_path)

    return mel_norm
