#!/usr/bin/python
# -*- coding: utf-8 -*-

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

genre_num_dict = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    8: 4,
    9: 5,
    10: 6,
    12: 7,
    13: 8,
    14: 9,
    15: 10,
    17: 11,
    20: 12,
    21: 13,
    38: 14,
    1235: 15,
    }

genre_dict = {
    'International': 0,
    'Blues': 1,
    'Jazz': 2,
    'Classical': 3,
    'Old-Time / Historic': 4,
    'Country': 5,
    'Pop': 6,
    'Rock': 7,
    'Easy Listening': 8,
    'Soul-RnB': 9,
    'Electronic': 10,
    'Folk': 11,
    'Spoken': 12,
    'Hip-Hop': 13,
    'Experimental': 14,
    'Instrumental': 15,
    }


def get_mfcc(
    signal,
    start,
    end,
    sr,
    n_mfcc=13,
    n_fft=20,
    hop_length=512,
    ):
    '''
    Returns the mfcc sequence for the segment of signal between start and end
    :param signal: audio as a floating point time series
    :param start: start index for the signal segment
    :param end: end index for the signal segment
    :param sr: sampling rate
    :param n_mfcc: number of mfccs to return
    :param n_fft: length of the windowed signal after padding with zeros
    :param hop_length: number of audio samples between adjacent STFT columns
    :return: mfcc sequence as numpy ndarray
    '''

    # apply short-term Fourier transform

    stft = np.abs(librosa.stft(signal[start:end], n_fft=n_fft,
                  hop_length=hop_length))

    # convert stft to spectrogram on mel-scale

    mel_spectrogram = librosa.feature.melspectrogram(S=stft ** 2, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # generate mfcc for audio track (mel-frequency cepstral coefficients)

    mfcc = librosa.feature.mfcc(S=mel_spectrogram, n_mfcc=n_mfcc)
    return mfcc


def process_track(file_path, sample_info, genre_index):
    '''
    Splits an audio file into segments and creates an mfcc sequence for
    each segment.
    :param file_path: file path to audio track (string)
    :param sample_info: dictionary of config settings for the processing,
    including: 'sample_rate', 'number_of_segments', 'samples_per_segment',
    'samples_per_track', 'n_mfcc', 'n_fft', and 'hop_length'
    :param genre_index: genre index for track
    :return: list of sequential mfccs for the audio file
    '''

    # load audio file as floating point time series

    (signal, sr) = librosa.load(file_path, sr=sample_info['sample_rate'
                                ])

    # to store mfcc segment data

    track_mfcc_list = []
    genre_list = []

    # break the signal up into segments

    for num in range(sample_info['number_of_segments']):

        # establish start and end indices for the segment

        start = sample_info['samples_per_segment'] * num
        end = start + sample_info['samples_per_track']

        mfcc = get_mfcc(
            signal,
            start=start,
            end=end,
            sr=sample_info['sample_rate'],
            n_mfcc=sample_info['n_mfcc'],
            n_fft=sample_info['n_fft'],
            hop_length=sample_info['hop_length'],
            )

        track_mfcc_list.append(mfcc.tolist())
        genre_list.append(genre_index)

    return (track_mfcc_list, genre_list)


def save_mfcc(dataset_path, audio_files_dir_path, json_path):

    # dictionary to store data

    data = {'mapping': [
        'International',
        'Blues',
        'Jazz',
        'Classical',
        'Old-Time / Historic',
        'Country',
        'Pop',
        'Rock',
        'Easy Listening',
        'Soul-RnB',
        'Electronic',
        'Folk',
        'Spoken',
        'Hip-Hop',
        'Experimental',
        'Instrumental',
        ], 'mfcc': [], 'labels': []}  # mfcc data arrays
                                      # segment labels by "mapping" index

    # object to store config information for processing

    sample_info = {
        'sample_rate': 22050,
        'track_duration': 30,
        'n_mfcc': 20,
        'n_fft': 2048,
        'hop_length': 512,
        'number_of_segments': 5,
        }

    # calculate additional information needed for processing

    sample_info['samples_per_track'] = sample_info['sample_rate'] \
        * sample_info['track_duration']
    sample_info['samples_per_segment'] = \
        int(sample_info['samples_per_track']
            / sample_info['number_of_segments'])

    # create df from csv file and shuffle the data

    df = pd.read_csv(dataset_path)
    df = shuffle(df).reset_index(drop=True)

    # create genre array that maps to the track index

    top_genres_array = df['top_genre_id'].to_numpy()
    genre_labels = [genre_num_dict[genre_id] for genre_id in
                    top_genres_array]

    count = 0

    # isolate indexed file list from dataframe

    file_list_array = df['filepath'].to_numpy()

    # # for TESTING
    # for index in range(0, 5):
        # file = file_list_array[index]

    # loop through files in the dataframe

    for (index, file) in enumerate(file_list_array):

        # combine directory path with file path

        file = audio_files_dir_path + file

        # save mfccs and corresponding genre labels

        (mfcc_list, genre_list) = process_track(file, sample_info,
                genre_labels[index])
        data['mfcc'].extend(mfcc_list)
        data['labels'].extend(genre_list)

        # display count as we process

        count += 1
        if count % 20 == 0:
            print 'files processed: {}'.format(count)

    # save to json file

    with open(json_path, 'w') as jp:
        json.dump(data, jp, indent=4)


if __name__ == '__main__':

    # command line argument parsing

    parser = \
        argparse.ArgumentParser(description='Preprocess audio files to mfccs'
                                )
    parser.add_argument('csv_file_path', type=str,
                        help='csv file with audio track filepaths and track genres'
                        )
    parser.add_argument('audio_root_directory', type=str,
                        help='root directory of audio track files')
    parser.add_argument('json_file_path', type=str,
                        help='path to store resulting .json file')
    args = parser.parse_args()

    # create a dataframe with 100 samples of each category

    df = pd.read_csv(args.csv_file_path)
    sample_df = df.groupby('top_genre_id').sample(n=100)
    sample_df.reset_index(drop=True, inplace=True)

    # save dataframe to csv file

    sample_df.to_csv('csv/small_sample.csv')

    save_mfcc('csv/small_sample.csv', args.audio_root_directory,
              args.json_file_path)
