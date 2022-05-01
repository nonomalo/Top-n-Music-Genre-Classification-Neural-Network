"""
Stand alone script to load files into Google Cloud Storage bucket
Requires python-dotenv and json-key for authorization
Requires google-cloud-storage (pip install google-cloud-storage)

CL: python3 load_file_to_storage.py <local-filepath> <destination-filepath>
"""

import argparse
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

BUCKET_NAME = 'audio_philes_data'


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket: From cloud storage docs:
    https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-code-sample
    :param bucket_name: 'audio_philes_data'
    :param source_file_name: 'local/path/to/file'
    :param destination_blob_name: 'storage-object-name'
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    return blob.public_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload file to Google Cloud Storage')
    parser.add_argument('path_to_local', type=str, help='Local filepath to upload')
    parser.add_argument('bucket_path', type=str, help='Storage path: <bucket-dir/filename>')
    args = parser.parse_args()

    print(upload_blob(BUCKET_NAME, args.path_to_local, args.bucket_path))
