from __future__ import unicode_literals
import os
import librosa
import numpy as np
import math

# config information for processing:
NUM_SEGMENTS = 5
SAMPLE_RATE = 22050
TRACK_DURATION = 30
N_FFT = 2048
HOP_LENGTH = 1024
MEL_BINS = 128
EXPECTED_SIGNAL_LENGTH = SAMPLE_RATE * TRACK_DURATION
EXPECTED_MELSPEC_LENGTH = \
    math.ceil(EXPECTED_SIGNAL_LENGTH / HOP_LENGTH)
EXPECTED_MELSPEC_SEGMENT_LENGTH = \
    math.ceil(EXPECTED_MELSPEC_LENGTH / NUM_SEGMENTS)


def process_track(file_path: str) -> np.ndarray:
    """Creates an mfcc sequence for each audio file.

    :param file_path: file path to audio track (string)
    :return: normalized melspectrogram for the audio file
    """
    signal, num_segments, error = extract_signal(file_path)
    if error:
        return None, error
    if not num_segments:
        num_segments = NUM_SEGMENTS

    # array to store segment data
    melspec_array = np.zeros((
        num_segments,
        MEL_BINS,
        EXPECTED_MELSPEC_SEGMENT_LENGTH
    ))

    segment_length = int(EXPECTED_SIGNAL_LENGTH / NUM_SEGMENTS)
    segment_count = 0

    for seg in range(num_segments):

        # calculate the start/end index for current segment
        start = segment_length * seg
        end = start + segment_length

        mel_norm = get_segment_melspec(signal, start, end)

        if mel_norm is not None:
            if mel_norm.shape == (
                    MEL_BINS,
                    EXPECTED_MELSPEC_SEGMENT_LENGTH
            ):
                melspec_array[segment_count] = mel_norm
                segment_count += 1

    # remove empty values
    melspec_array = melspec_array[:segment_count]

    # delete the track from temporary storage
    os.remove(file_path)

    if melspec_array.shape[0] == 0:
        return None, 'Unable to process data from audio_file'

    return melspec_array, None


def get_segment_melspec(signal, start, end):
    """Gets the normalized melspectrogram for an audio segment

    :param signal: librosa audio buffer
    :param start: (int) starting index
    :param end: (int) ending index
    :return: normalized melspectrogram
    """
    segment_length = int(EXPECTED_SIGNAL_LENGTH / NUM_SEGMENTS)

    # abort if array is empty
    if len(signal[start:end]) <= 0:
        return None

    # normalize segment
    signal_norm = librosa.util.normalize(signal[start:end])

    # restrict signal length to maintain consistent shape along ndarrays
    if signal_norm.shape[0] < segment_length:
        return None
    elif signal_norm.shape[0] > segment_length:
        signal_norm = signal_norm[:segment_length]

    # apply short-term Fourier transform
    stft = np.abs(
        librosa.stft(
            signal_norm,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH))

    # convert to normalized mel spectrogram
    mel = librosa.feature.melspectrogram(S=stft ** 2, sr=SAMPLE_RATE)
    mel_log = np.log(mel + 1e-9)
    mel_norm = librosa.util.normalize(mel_log)

    return mel_norm


def extract_signal(file_path):
    """Gets the librosa audio buffer for a given audio file
    includes the number of segments if the audio file duration
    is less than TRACK_DURATION

    :param file_path: path to audio file
    :return: audio buffer, number of segments, error string
    """
    num_segments = None
    try:
        duration = librosa.get_duration(filename=file_path)

        # restrict sample if audio file length is > TRACK_DURATION
        if duration > TRACK_DURATION:
            signal, sr = librosa.load(
                file_path,
                sr=SAMPLE_RATE,
                duration=TRACK_DURATION
            )
        else:
            # load audio file as floating point time series
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # change number of segments if file < TRACK_DURATION
        if duration < TRACK_DURATION:
            expected_segment_duration = int(TRACK_DURATION / NUM_SEGMENTS)
            num_segments = int(duration / expected_segment_duration)
            if num_segments == 0:
                os.remove(file_path)
                return None, None, \
                    'Track duration was not long enough to process'

    except FileNotFoundError:
        return None, None, f'File Not Found: {file_path}'

    return signal, num_segments, None
