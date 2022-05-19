import argparse
import pandas as pd
import numpy as np
import librosa
import json
from sklearn.utils import shuffle
import math
import os
from pydub import AudioSegment
from pydub.utils import make_chunks



# # MEDIUM DATASET
# genre_dict = {
#     'International': 0, 'Blues': 1, 'Jazz': 2,
#     'Classical': 3, 'Old-Time / Historic': 4,
#     'Country': 5, 'Pop': 6, 'Rock': 7,
#     'Easy Listening': 8, 'Soul-RnB': 9,
#     'Electronic': 10, 'Folk': 11, 'Spoken': 12,
#     'Hip-Hop': 13, 'Experimental': 14, 'Instrumental': 15
# }
# mapping = ['International', 'Blues', 'Jazz', 'Classical',
#            'Old-Time / Historic', 'Country', 'Pop', 'Rock',
#            'Easy Listening', 'Soul-RnB', 'Electronic',
#            'Folk', 'Spoken', 'Hip-Hop', 'Experimental',
#            'Instrumental']

# SMALL DATASET
genre_dict = {
    'Electronic': 0, 'Experimental': 1, 'Folk': 2,
    'Hip-Hop': 3, 'Instrumental': 4,
    'International': 5, 'Pop': 6, 'Rock': 7
}
mapping = ['Electronic', 'Experimental', 'Folk',
           'Hip-Hop', 'Instrumental',
           'International', 'Pop', 'Rock']


def main():
    """
    Preprocess audio files into mfccs from command line
    and store as .json file

    CL: python3 preprocess.py <csv-data-filepath> <json-filepath>

    Flags: -t to pre-process "tiny" file,
           -s <int> to pre-process sample size of <int> from each genre
           Example:
               python3 preprocess.py -s 50 <csv-data-filepath> <json-filepath>
               will process 50 audio files from each genre
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description='Preprocess audio files to mfccs')
    parser.add_argument('--tiny', '-t', action='store_true')
    parser.add_argument('--sample', '-s', type=int)
    parser.add_argument(
        'csv_file_path',
        type=str,
        help='csv file with audio track filepaths and track genres')
    parser.add_argument('json_file_path', type=str,
                        help='path to store resulting .json file')
    args = parser.parse_args()

    csv_file_path = args.csv_file_path

    # for tiny and sample, create a smaller dataset
    # and save to a local file
    if args.sample is not None or args.tiny:

        # create the smaller sample from the larger dataset
        df = pd.read_csv(args.csv_file_path)
        n = args.sample if args.sample is not None else 2
        sample_df = df.groupby('genre_top').sample(n=n)
        sample_df.reset_index(drop=True, inplace=True)

        # save dataframe to csv file
        if not os.path.exists("csv"):
            os.mkdir("csv")
        sample_df.to_csv("csv/small_sample.csv")
        csv_file_path = "csv/small_sample.csv"

    process_track_list(
        csv_file_path,
        args.json_file_path)


def process_track(file_path, sample_info):
    """
    Creates an mfcc sequence for each audio file.
    :param file_path: file path to audio track (string)
    :param sample_info: dictionary of config settings for the processing,
    including: 'sample_rate', 'n_mfcc', 'n_fft', 'hop_length',
    and expected_mfcc_length
    :return: list of sequential mfccs for the audio file
    """

    expected_signal_length = sample_info['sample_rate'] * \
        sample_info['track_duration']

    try:

        # load audio file as floating point time series
        signal, sr = librosa.load(file_path, sr=sample_info['sample_rate'])

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
            n_fft=sample_info['n_fft'],
            hop_length=sample_info['hop_length']))

    # convert stft to spectrogram on mel-scale
    mel_spectrogram = librosa.feature.melspectrogram(S=stft ** 2, sr=sr)
    # apply discrete cosine transform
    db = librosa.power_to_db(mel_spectrogram)

    # generate mfcc for audio track (mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=sample_info['n_mfcc'])

    return mfcc


def process_track_list(dataset_path, json_path):
    """
    Stores the mfcc data for each track in the dataset
    :param dataset_path: csv file from create_dataset_track_list.py
    :param json_path: path to json output file
    """

    # dictionary to store data
    data = {
        "mapping": mapping,
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
                print(f'mfcc shape error: {mfccs.shape}')

            else:

                data['mfcc'].append(mfccs.tolist())
                data['labels'].append(genre_labels[index])

                # display count of processed files
                count += 1
                if count % 20 == 0:
                    print(f'files processed: {count}')

    print(f'Total files processed: {count}')

    print('Saving file to json...')

    # save to json file
    with open(json_path, 'w') as jp:
        json.dump(data, jp, indent=4)

    with open(json_path) as json_file:
        data = json.load(json_file)

    # print shape of saved mfccs
    mfcc_array = np.array([np.array(n) for n in data['mfcc']])
    print(f'MFCCs shape: {mfcc_array.shape}')


def split_audio_files(wav_file_path):
    "Pass wav file path, split it into thirty second chunks and return a list of each chunk transformed into wav files"

    thirty_seconds_in_milliseconds = 30000
    original_audio_segment = AudioSegment.from_file(wav_file_path, "")
    split_chunks = make_chunks(original_audio_segment, thirty_seconds_in_milliseconds)

    for i, chunk in enumerate(split_chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print
        "exporting", chunk_name
        # Export each chunk to the same folder the function is called from for processing
        chunk.export(chunk_name, format="wav")


if __name__ == "__main__":
    main()
