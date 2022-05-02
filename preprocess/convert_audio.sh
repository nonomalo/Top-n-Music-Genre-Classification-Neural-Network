#! /bin/bash
# first arg = source directory
# second arg = output directory

# convert all .mp3 files in given directory to .wav files


# guard to exit loop if no matching files
shopt -s nullglob

# grab command line arguments
SOURCE_DIR=$1
OUTPUT_DIR=$2


# set up files for the dump of .wav files
mkdir -p $OUTPUT_DIR


for file_path in $SOURCE_DIR/*; do

  #  pull subdirectory name from file path
  DIR_NAME=${file_path##*/}

  # create subdirectory in output directory
  if [ -d "$file_path" ]; then
    mkdir -p $OUTPUT_DIR/$DIR_NAME
  fi

  # loop through .mp3 files in the subdirectory
  for mp3_file in $file_path/*.mp3; do

    # extract the file name with .mp3 extension
    MP3_NAME=${mp3_file##*/}

    # convert file name to .wav extension
    WAV_NAME=${MP3_NAME%.mp3}.wav

    # convert the .mp3 to .wav file and save in output directory
    ffmpeg -i "${mp3_file}" $OUTPUT_DIR/$DIR_NAME/$WAV_NAME
  done;

done;

echo ".mp3 files converted to .wav files"