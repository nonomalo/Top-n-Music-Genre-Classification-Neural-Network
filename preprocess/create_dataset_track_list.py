"""
Combine fma_ dataset track indexes with corresponding
genre and filepath to a csv file

To Run:
python3 create_dataset_track_list.py <dir-of-wav-files> <tracks.csv-path> <output-filepath>
"""
import os
import argparse
import pandas as pd


def get_track_list(audio_directory_name, track_csv_path, output_filepath, tiny=False):
    """
    Creates a csv file of track information including the track id,
    track filepath, and genre for each track nested in a particular directory
    :param audio_directory_name: path to directory to search
    :param track_csv_path: path to fma_metadata's tracks.csv file
    :param output_filepath: where to save the output csv file
    """

    track_dict = {
        'track_id': [],
        'track': [],
        'path': []
    }

    # save all .wav files in directory into dictionary
    for (dir_path, dir_names, filenames) in os.walk(audio_directory_name):
        for filename in filenames:
            if filename.endswith('.wav'):
                track_dict['track_id'].append(
                    int(os.path.splitext(filename)[0]))
                track_dict['track'].append(filename)
                track_dict['path'].append(os.sep.join([dir_path, filename]))

    # convert to pandas dataframe
    tracks_df = pd.DataFrame(track_dict)

    # convert tracks.csv to dataframe
    header = 1 if not tiny else 0
    meta_df = pd.read_csv(
        track_csv_path,
        header=header,
        low_memory=False)
    if not tiny:
        meta_df.columns.values[0] = 'track_id'
        meta_df = meta_df.iloc[1:]
    meta_df = meta_df[['track_id', 'genre_top']]
    meta_df = meta_df.astype({'track_id': int})

    # join tracks_df to meta_df on the track_id column
    df = pd.merge(
        tracks_df,
        meta_df,
        how='inner',
        on='track_id'
    )

    # save resulting dataframe to csv file
    df.to_csv(output_filepath)


if __name__ == '__main__':
    # command line argument parsing
    parser = argparse.ArgumentParser(
        description='Combine track, genre, and filepath to csv file')
    parser.add_argument('--tiny', '-t', action='store_true')
    parser.add_argument('audio_file_dir', type=str,
                        help='path directory containing .wav audio files')
    parser.add_argument('tracks_csv_file', type=str,
                        help='path to fma_metadata tracks.csv file')
    parser.add_argument('output_csv', type=str,
                        help='path to store resulting .csv file')
    args = parser.parse_args()

    get_track_list(args.audio_file_dir, args.tracks_csv_file, args.output_csv, args.tiny)
