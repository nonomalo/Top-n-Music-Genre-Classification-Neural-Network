import os
import yt_dlp
import re

STORED_AUDIO = 'audio/temp.wav'

def download_wav_file(url, unique_id):
    """
    Download and save the first 30 seconds of
    audio from the url in wav format
    :param url: music or music video url
    :return: error string if error, else None
    """
    data = {}

    # remove extension from STORED_AUDIO path
    store_as = os.path.splitext(STORED_AUDIO)[0] + unique_id

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

            # download the audio and return metadata
            metadata = ydl.extract_info(url, download=True)

            data = {
                'filename': store_as + '.wav',
                'title': metadata['title'],
                'track': metadata['track'],
                'artist': metadata['artist']
            }

    except yt_dlp.utils.DownloadError as e:
        # remove special string formatting:
        # see: https://stackoverflow.com/questions/30425105
        error = re.sub(r'\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))', '', str(e))
        data['error'] = error

    return data
