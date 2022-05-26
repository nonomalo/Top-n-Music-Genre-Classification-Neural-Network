"""Creates npy dataset as mel-spectrograms and labels
with segmented tracks

Usage: python3 preprocess_spectro_slice.py \
    num_segments csv_file_path storage_dir batch_num
"""

import argparse
import pandas as pd
import numpy as np
import librosa
import math

SAMPLE_RATE = 22050
TRACK_DURATION = 30
N_FFT = 2048
HOP_LENGTH = 1024
MEL_BINS = 128
EXPECTED_MELSPEC_LENGTH = math.ceil(
    (SAMPLE_RATE * TRACK_DURATION) / HOP_LENGTH
)
EXPECTED_SIGNAL_LENGTH = SAMPLE_RATE * TRACK_DURATION

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


def main() -> None:
    """Preprocess audio files into mel spectrograms from command line
        and store as .npy files.
    """
    args = get_arguments()

    process_track_list(
        args.num_segments,
        args.csv_file_path,
        args.storage_dir,
        args.batch_num
    )


def process_track_list(
        num_segments: int,
        dataset_path: str,
        storage_dir: str,
        batch_num: int
) -> None:
    """Stores the mel spectrogram data for each track in the dataset.

    :param num_segments: int number of segments for each track
    :param dataset_path: csv file from create_dataset_track_list.py
    :param storage_dir: path to storage directory
    :param batch_num: integer batch file number
    """
    print('Reading csv file to dataframe...')

    # create df from csv file
    df = pd.read_csv(dataset_path)

    # create genre array that maps to the track index
    top_genres_array = df['genre_top'].to_numpy()
    genre_labels = [genre_dict[genre_id] for genre_id in top_genres_array]

    file_count = 0
    segment_count = 0

    # isolate indexed file list from dataframe
    file_list_array = df['path'].to_numpy()

    num_rows = len(file_list_array)

    expected_melspec_segment_length = \
        math.ceil(EXPECTED_MELSPEC_LENGTH / num_segments)

    # arrays to store data
    melspec_array = np.zeros((
        num_rows * num_segments,
        MEL_BINS,
        expected_melspec_segment_length
    ))
    labels_array = np.zeros((num_rows * num_segments))

    print('Processing audio files...')
    idx = 0

    segment_length = int(EXPECTED_SIGNAL_LENGTH / num_segments)

    # loop through files in the dataframe
    for index, file in enumerate(file_list_array):

        try:
            # load the audio file
            signal, sr = librosa.load(file, sr=SAMPLE_RATE)

            # normalize waveform
            signal_norm = librosa.util.normalize(signal)

        except FileNotFoundError:

            print(f'File Not Found: {file}')
            continue

        for seg in range(num_segments):

            # calculate the start/end index for current segment
            start = segment_length * seg
            end = start + segment_length

            # abort if array is empty
            if len(signal[start:end]) <= 0:
                continue

            # normalize segment
            signal_norm = librosa.util.normalize(signal[start:end])

            if signal_norm.shape[0] < segment_length:
                continue
            elif signal_norm.shape[0] > segment_length:
                signal_norm = signal_norm[:segment_length]

            # apply short-term Fourier transform
            stft = np.abs(
                librosa.stft(
                    signal_norm,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                ))

            # convert to normalized mel spectrogram
            mel = librosa.feature.melspectrogram(S=stft**2, sr=sr)
            mel_log = np.log(mel + 1e-9)
            mel_norm = librosa.util.normalize(mel_log)

            # check length of mel_norm
            if mel_norm.shape == (
                MEL_BINS,
                expected_melspec_segment_length
            ):
                melspec_array[idx] = mel_norm
                labels_array[idx] = genre_labels[index]
                idx += 1
                segment_count += 1

        file_count += 1
        if file_count % 20 == 0:
            print(f'files processed: {file_count}')

    # remove empty values
    melspec_array = melspec_array[:segment_count]
    labels_array = labels_array[:segment_count].astype(int)

    print(f'Total files processed: {file_count}')
    print(f'Total segments processed: {segment_count}')
    # print(f'Number of bad segments: {num_bad_segments}')

    print(f'Data shape: {melspec_array.shape}')
    print(f'Label shape: {labels_array.shape}')

    print('Saving file to npy file...')

    melspec_path = storage_dir + '/melspec_data_' + str(batch_num)
    np.save(melspec_path, melspec_array, allow_pickle=False)

    labels_path = storage_dir + '/labels_data_' + str(batch_num)
    np.save(labels_path, labels_array, allow_pickle=False)


def get_arguments() -> argparse.Namespace:
    """Parses command line arguments

    :return command line arguments object
    """
    parser = argparse.ArgumentParser(
        description='Preprocess audio files to mel spectrograms'
    )
    parser.add_argument(
        'num_segments',
        type=int,
        help='number of segments for each track'
    )
    parser.add_argument(
        'csv_file_path',
        type=str,
        help='csv file with audio track filepaths and track genres'
    )
    parser.add_argument(
        'storage_dir',
        type=str,
        help='path to storage directory'
    )
    parser.add_argument(
        'batch_num',
        type=int,
        help='integer number for batch'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
