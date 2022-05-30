:musical_note:

## Top-n-Music-Genre-Classification-Neural-Network

### Developers:
* [Brooks Burns](https://github.com/Brrookss)
* [Noah Lopez](https://github.com/nonomalo)
* [Amin Malik](https://github.com/amin-malik)
* [Sydney Somerfield](https://github.com/somesyd)

### Goal
Train a neural network model to accurately predict musical genre for a previously unknown audio clip

### Project
To train the model we used the [Free Music Archive](https://github.com/mdeff/fma), a dataset of 25,000 tracks classified into 16 genres. 

Using Librosa, each 30 second track in the dataset was sliced into 5 segments and converted into mel-scaled spectrogram
(a visualization of the frequency spectrum over time). Convolutional neural networks are commonly used for image classification tasks, so 
we were able to build a convolutional neural network model with the audio spectrogram "images" using Keras and TensorFlow.

#### Genre Prediction API

To access the model, we created a Flask [web-service genre prediction API](https://github.com/nonomalo/Top-n-Music-Genre-Classification-Neural-Network/tree/main/server/README.md) that accepts a .wav file, converts it into segmented spectrogram data and runs the data 
against the model, returning a json array of confidence values for the 16 genres.

https://top-n-server.uw.r.appspot.com/

#### User-Friendly web service: "Hear We Go!"

To demonstrate the web-service and model, we developed a user-friendly Flask [web application](https://github.com/nonomalo/Top-n-Music-Genre-Classification-Neural-Network/tree/main/client/README.md) that accepts a Youtube music video url or a user-uploaded 
audio file and displays the prediction output based on the genre prediction API.

https://top-n-client.uw.r.appspot.com/


### RESOURCES
* [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the neural network
* [Librosa](https://librosa.org/doc/latest/index.html) and [FFmpeg](https://ffmpeg.org/) for audio processing
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) for fetching audio and song metadata
* [NumPy](https://numpy.org/) for data processing
* [Pandas](https://pandas.pydata.org/) for labeling training data
* [Matplotlib](https://matplotlib.org/) for data visualization
* [Flask](https://flask.palletsprojects.com/en/2.1.x/) for web development
* [Google Cloud Platform](https://cloud.google.com/) for dataset storage & application deployment
