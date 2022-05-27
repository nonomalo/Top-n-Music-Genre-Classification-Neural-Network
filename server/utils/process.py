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


def process_track(file_path: str) -> np.ndarray:
    """Creates an mfcc sequence for each audio file.

    :param file_path: file path to audio track (string)
    :return: normalized melspectrogram for the audio file
    """
    expected_signal_length = SAMPLE_RATE * TRACK_DURATION
    expected_melspec_length = \
        math.ceil(expected_signal_length / HOP_LENGTH)
    expected_melspec_segment_length = \
        math.ceil(expected_melspec_length / NUM_SEGMENTS)

    num_segments = NUM_SEGMENTS

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
                return None, 'Track duration was not long enough to process'

    except FileNotFoundError:

        return None, f'File Not Found: {file_path}'

    segment_length = int(expected_signal_length / NUM_SEGMENTS)

    # array to store segment data
    melspec_array = np.zeros((
        num_segments,
        MEL_BINS,
        expected_melspec_segment_length
    ))

    segment_count = 0

    for seg in range(num_segments):

        # calculate the start/end index for current segment
        start = segment_length * seg
        end = start + segment_length

        # abort if array is empty
        if len(signal[start:end]) <= 0:
            continue

        # normalize segment
        signal_norm = librosa.util.normalize(signal[start:end])

        # restrict signal length to maintain consistent shape along ndarrays
        if signal_norm.shape[0] < segment_length:
            continue
        elif signal_norm.shape[0] > segment_length:
            signal_norm = signal_norm[:segment_length]

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

        if mel_norm.shape == (
            MEL_BINS,
            expected_melspec_segment_length
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
