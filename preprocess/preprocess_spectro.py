import argparse
import pandas as pd
import numpy as np
import librosa
import math
import os
from pydub import AudioSegment
from pydub.utils import make_chunks

# MEDIUM DATASET
genre_dict = {
    'International': 0, 'Blues': 1, 'Jazz': 2,
    'Classical': 3, 'Old-Time / Historic': 4,
    'Country': 5, 'Pop': 6, 'Rock': 7,
    'Easy Listening': 8, 'Soul-RnB': 9,
    'Electronic': 10, 'Folk': 11, 'Spoken': 12,
    'Hip-Hop': 13, 'Experimental': 14, 'Instrumental': 15
}
mapping = [
    'International', 'Blues', 'Jazz', 'Classical',
    'Old-Time / Historic', 'Country', 'Pop', 'Rock',
    'Easy Listening', 'Soul-RnB', 'Electronic',
    'Folk', 'Spoken', 'Hip-Hop', 'Experimental',
    'Instrumental'
]

# SMALL DATASET
# genre_dict = {
#     'Electronic': 0, 'Experimental': 1, 'Folk': 2,
#     'Hip-Hop': 3, 'Instrumental': 4,
#     'International': 5, 'Pop': 6, 'Rock': 7
# }
# mapping = [
#     'Electronic', 'Experimental', 'Folk',
#     'Hip-Hop', 'Instrumental',
#     'International', 'Pop', 'Rock'
# ]


def main():
    """Preprocess audio files into mel spectrograms from command line
    and store as .npy file.

    CL: python3 preprocess_spectro.py <csv-data-filepath> \
            <storage-dir> <batch-number>

    Flags: -t to pre-process "tiny" file,
           -s <int> to pre-process sample size of <int> from each genre
           Example:
               python3 preprocess.py -s 50 \
                    <csv-data-filepath> <storage-dir> <batch-number>
               will process 50 audio files from each genre
    """
    parser = argparse.ArgumentParser(
        description='Preprocess audio files to mel spectrograms')
    parser.add_argument('--tiny', '-t', action='store_true')
    parser.add_argument('--sample', '-s', type=int)
    parser.add_argument(
        'csv_file_path',
        type=str,
        help='csv file with audio track filepaths and track genres')
    parser.add_argument('storage_dir', type=str,
                        help='path to storage directory')
    parser.add_argument('batch_num', type=int,
                        help='integer number for batch')
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
        args.storage_dir,
        args.batch_num)


def process_track(file_path, sample_info):
    """Creates a mel spectrogram sequence for each audio file.
    :param file_path: file path to audio track (string)
    :param sample_info: dictionary of config settings for the processing,
    including: 'sample_rate', 'n_fft', 'hop_length',
    and expected_signal_length

    :return: list of sequential mel spectrograms for the audio file
    """

    try:

        # load audio file as floating point time series / waveform
        signal, sr = librosa.load(file_path, sr=sample_info['sample_rate'])

    except FileNotFoundError:

        print(f'File Not Found: {file_path}')
        return None

    # restrict signal length to maintain consistent shape along ndarrays
    if signal.shape[0] > sample_info['expected_signal_length']:
        signal = signal[:sample_info['expected_signal_length']]

    # normalize waveform
    signal_norm = librosa.util.normalize(signal)

    # apply short-term Fourier transform
    stft = np.abs(
        librosa.stft(
            signal_norm,
            n_fft=sample_info['n_fft'],
            hop_length=sample_info['hop_length']))

    # convert to normalized mel spectrogram
    mel = librosa.feature.melspectrogram(S=stft**2, sr=sr)
    mel_log = np.log(mel + 1e-9)
    mel_norm = librosa.util.normalize(mel_log)

    return mel_norm


def process_track_list(dataset_path, storage_dir, batch_num):
    """Stores the mel spectrogram data for each track in the dataset.

    :param dataset_path: csv file from create_dataset_track_list.py
    :param storage_dir: path to storage directory
    :param batch_num: integer batch file number
    """

    # object to store config information for processing
    sample_info = {'sample_rate': 22050,
                   'track_duration': 30,
                   'n_fft': 2048,
                   'hop_length': 1024,
                   'mel_bins': 128}

    sample_info['expected_melspec_length'] = math.ceil(
        (sample_info['sample_rate'] *
         sample_info['track_duration']) /
        sample_info['hop_length'])

    sample_info['expected_signal_length'] = \
        sample_info['sample_rate'] * sample_info['track_duration']

    print('Reading csv file to dataframe...')

    # create df from csv file
    df = pd.read_csv(dataset_path)

    # create genre array that maps to the track index
    top_genres_array = df['genre_top'].to_numpy()
    genre_labels = [genre_dict[genre_id] for genre_id in top_genres_array]

    count = 0

    # isolate indexed file list from dataframe
    file_list_array = df['path'].to_numpy()

    num_rows = len(file_list_array)

    # arrays to store data
    melspec_array = np.zeros((
        num_rows,
        sample_info['mel_bins'],
        sample_info['expected_melspec_length']
    ))
    labels_array = np.zeros((num_rows))

    print('Processing audio files...')
    num_bad_files = 0
    idx = 0

    # for TESTING  # noqa
    # for index in range(0, 5): # noqa
    #     file = file_list_array[index] # noqa

    # loop through files in the dataframe
    for index, file in enumerate(file_list_array):

        split_audio_files_path_list = split_audio_files(file)

        for audio_file_slice_name in split_audio_files_path_list:

            # save melspec and corresponding genre labels
            melspec = process_track(audio_file_slice_name, sample_info)

            if melspec is not None:

                # ensure melspec shape is consistent before saving
                # (128 default mel-bins, expected_length)
                if melspec.shape != (
                        sample_info['mel_bins'],
                        sample_info['expected_melspec_length']):

                    print(f'melspectrogram shape error: {melspec.shape}')

                    num_bad_files += 1

                else:
                    melspec_array[idx] = melspec
                    labels_array[idx] = genre_labels[index]
                    idx += 1

                # display count of processed files
                count += 1
                if count % 20 == 0:
                    print(f'files processed: {count}')

    # remove empty values
    melspec_array = melspec_array[:num_rows - num_bad_files]
    labels_array = labels_array[:num_rows - num_bad_files].astype(int)

    print(f'Total files processed: {count}')

    print(f'Data shape: {melspec_array.shape}')
    print(f'Label shape: {labels_array.shape}')

    print('Saving file to npy file...')

    melspec_path = storage_dir + '/melspec_data_' + str(batch_num)
    np.save(melspec_path, melspec_array, allow_pickle=False)

    labels_path = storage_dir + '/labels_data_' + str(batch_num)
    np.save(labels_path, labels_array, allow_pickle=False)


def split_audio_files(wav_file_path):
    six_seconds_in_milliseconds = 6000
    original_audio_segment = AudioSegment.from_file(wav_file_path, "")
    split_chunks = make_chunks(
        original_audio_segment,
        six_seconds_in_milliseconds)

    split_wav_file_list = []

    for i, chunk in enumerate(split_chunks):
        original_file_name = str(wav_file_path)
        new_file_name = original_file_name + "chunk{0}.wav"
        chunk_name = new_file_name.format(i)
        # Export each chunk to the same folder the function is called from for
        # processing
        (chunk.export(chunk_name, format="wav"))
        # Export each chunk to the same folder the function is called from for
        # processing
        split_wav_file_list.append(chunk_name)

    return split_wav_file_list


if __name__ == "__main__":
    main()
