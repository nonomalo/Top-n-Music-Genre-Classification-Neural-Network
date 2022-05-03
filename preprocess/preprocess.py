"""
Preprocess audio files into mfccs from command line
and store as .json file

CL: python3 preprocess.py <csv-data-filepath> <audio-root-directory-path> <json-filepath>
"""

import argparse
import pandas as pd
import numpy as np
import librosa
import json
from sklearn.utils import shuffle
import math
import os


genre_num_dict = {
    2: 0, 3: 1, 4: 2, 5: 3, 8: 4,
    9: 5, 10: 6, 12: 7, 13: 8,
    14: 9, 15: 10, 17: 11, 20: 12,
    21: 13, 38: 14, 1235: 15
}

genre_dict = {
    'International': 0, 'Blues': 1, 'Jazz': 2,
    'Classical': 3, 'Old-Time / Historic': 4,
    'Country': 5, 'Pop': 6, 'Rock': 7,
    'Easy Listening': 8, 'Soul-RnB': 9,
    'Electronic': 10, 'Folk': 11, 'Spoken': 12,
    'Hip-Hop': 13, 'Experimental': 14, 'Instrumental': 15
}


def process_track(file_path, sample_info):
    """
    Splits an audio file into segments and creates an mfcc sequence for
    each segment.
    :param file_path: file path to audio track (string)
    :param sample_info: dictionary of config settings for the processing,
    including: 'sample_rate', 'number_of_segments', 'samples_per_segment',
    'samples_per_track', 'n_mfcc', 'n_fft', and 'hop_length'
    :param genre_index: genre index for track
    :return: list of sequential mfccs for the audio file
    """

    expected_signal_length = sample_info['sample_rate'] * \
        sample_info['track_duration']

    try:

        # load audio file as floating point time series
        signal, sr = librosa.load(file_path, sr=sample_info['sample_rate'])

    except FileNotFoundError:

        print('File Not Found: {}'.format(file_path))
        return None

    # restrict signal length to maintain consistent shape along ndarrays
    if signal.shape[0] > expected_signal_length:
        signal = signal[:expected_signal_length]

    # apply short-term Fourier transform
    stft = np.abs(
        librosa.stft(
            signal,
            n_fft=sample_info['n_fft'],
            hop_length=sample_info['hop_length']))

    # convert stft to spectrogram on mel-scale
    mel_spectrogram = librosa.feature.melspectrogram(S=stft ** 2, sr=sr)
    # apply discrete cosine transform
    db = librosa.power_to_db(mel_spectrogram)

    # generate mfcc for audio track (mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=sample_info['n_mfcc'])

    return mfcc


def process_track_list(dataset_path, audio_files_dir_path, json_path):

    # dictionary to store data
    data = {
        "mapping": ['International', 'Blues', 'Jazz', 'Classical', 'Old-Time / Historic',
                    'Country', 'Pop', 'Rock', 'Easy Listening', 'Soul-RnB', 'Electronic',
                    'Folk', 'Spoken', 'Hip-Hop', 'Experimental', 'Instrumental'],
        "mfcc": [],  # mfcc data arrays
        "labels": []  # segment labels by "mapping" index
    }

    # object to store config information for processing
    sample_info = {'sample_rate': 22050,
                   'track_duration': 30,
                   'n_mfcc': 20,
                   'n_fft': 2048,
                   'hop_length': 1024}

    sample_info['expected_mfcc_length'] = math.ceil(
        (sample_info['sample_rate'] *
         sample_info['track_duration']) /
        sample_info['hop_length'])

    print('Reading csv file to dataframe...')

    # create df from csv file and shuffle the data
    df = pd.read_csv(dataset_path)
    df = shuffle(df).reset_index(drop=True)

    # create genre array that maps to the track index
    top_genres_array = df['genre_top'].to_numpy()
    genre_labels = [genre_dict[genre_id] for genre_id in top_genres_array]

    count = 0

    # isolate indexed file list from dataframe
    file_list_array = df['path'].to_numpy()

    print('Processing audio files...')

    # for TESTING  # noqa
    # for index in range(0, 5): # noqa
    #     file = file_list_array[index] # noqa

    # loop through files in the dataframe
    for index, file in enumerate(file_list_array):

        # save mfccs and corresponding genre labels
        mfccs = process_track(file, sample_info)

        if mfccs is not None:

            # ensure the mfcc shape is consistent before saving
            if mfccs.shape != (
                    sample_info['n_mfcc'],
                    sample_info['expected_mfcc_length']):
                print('mfcc shape error: {}'.format(mfccs.shape))

            else:

                data['mfcc'].append(mfccs.tolist())
                data['labels'].append(genre_labels[index])

                # display count of processed files
                count += 1
                if count % 20 == 0:
                    print("files processed: {}".format(count))

    print('Total files processed: {}'.format(count))

    # save to json file
    with open(json_path, 'w') as jp:
        json.dump(data, jp, indent=4)


if __name__ == "__main__":

    # command line argument parsing
    parser = argparse.ArgumentParser(
        description='Preprocess audio files to mfccs')
    parser.add_argument('--tiny', '-t', action='store_true')
    parser.add_argument(
        'csv_file_path',
        type=str,
        help='csv file with audio track filepaths and track genres')
    parser.add_argument('audio_root_directory', type=str,
                        help='root directory of audio track files')
    parser.add_argument('json_file_path', type=str,
                        help='path to store resulting .json file')
    args = parser.parse_args()

    # CURRENTLY CONFIGURED TO PROCESS A SMALL SAMPLE FOR TESTING
    # create a dataframe with 100 samples of each category
    df = pd.read_csv(args.csv_file_path)
    df2 = df.groupby(['genre_top'])['genre_top'].count()
    n = 100 if not args.tiny else 2
    sample_df = df.groupby('genre_top').sample(n=n)
    sample_df.reset_index(drop=True, inplace=True)

    # save dataframe to csv file
    if not os.path.exists("csv"):
        os.mkdir("csv")
    sample_df.to_csv("csv/small_sample.csv")

    process_track_list(
        "csv/small_sample.csv",
        args.audio_root_directory,
        args.json_file_path)
