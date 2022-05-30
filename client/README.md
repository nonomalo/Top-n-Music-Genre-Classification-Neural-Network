## Top-n-Music-Genre-Classification-Neural-Network :headphones:

This Flask web app allows a user to analyze an audio file in one of two ways: 

* Input a url for a Youtube music video, Youtube Music track, or a SoundCloud track. 
* Upload their own file in one of the accepted audio file formats. 

The app produces data visualizations highlighting features of the audio sample and uses a
genre prediction service to predict the track's genre based on the first 30 seconds of audio. 

The genre prediction service uses a convolutional neural network model trained on data from the 
[Free Music Archive](https://github.com/mdeff/fma), a dataset of 25,000 tracks classified into 16 genres: 
* International
* Blues
* Jazz
* Classical
* Old-Time / Historic
* Country
* Pop
* Rock
* Easy Listening
* Soul-RnB
* Electronic
* Folk
* Spoken
* Hip-Hop
* Experimental
* Instrumental

## This App

https://top-n-client.uw.r.appspot.com/

## Genre Prediction API

Details can be found [here](https://github.com/nonomalo/Top-n-Music-Genre-Classification-Neural-Network/tree/main/server/README.md).

https://top-n-server.uw.r.appspot.com



### RESOURCES
* [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the neural network
* [Librosa](https://librosa.org/doc/latest/index.html) and [FFmpeg](https://ffmpeg.org/) for audio processing
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) for fetching audio and song metadata
* [NumPy](https://numpy.org/) for data processing
* [Pandas](https://pandas.pydata.org/) for labeling training data
* [Matplotlib](https://matplotlib.org/) for data visualization
* [Flask](https://flask.palletsprojects.com/en/2.1.x/) for web development
* [Google Cloud Platform](https://cloud.google.com/) for dataset storage & application deployment

