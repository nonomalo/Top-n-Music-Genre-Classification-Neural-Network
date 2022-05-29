:musical_score:
## Top-n-Music-Genre-Classification-Neural-Network

This music genre prediction service applies a convolutional neural network
model trained on data from the [Free Music Archive](https://github.com/mdeff/fma) dataset. 
The Flask web service returns a list of categorical probabilities for 16 genres: 
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

## Genre Prediction API

https://top-n-server.uw.r.appspot.com/

### REQUEST

Get a list of genre predictions based on a .wav audio file

`POST` /genre

**curl**
```
curl -v -F audio=@"my-file.wav" https://top-n-server.uw.r.appspot.com/genre
```

**Python**
```
import requests
import json

res = requests.post(
    'https://top-n-server.uw.r.appspot.com/genre',
    files={'audio': open('filename', 'rb')},
)

predictions = json.loads(res.content)

```

### RESPONSE
```
200 OK
{
    "genres": [
        {
            "genre": "International",
            "prediction": 0.02285991981625557
        },
        {
            "genre": "Blues",
            "prediction": 0.005570589564740658
        },
        {
            "genre": "Jazz",
            "prediction": 0.019552502781152725
        },
        {
            "genre": "Classical",
            "prediction": 0.0024203481152653694
        },
        {
            "genre": "Old-Time / Historic",
            "prediction": 4.863575668423437e-06
        },
        {
            "genre": "Country",
            "prediction": 0.006214406806975603
        },
        {
            "genre": "Pop",
            "prediction": 0.18524646759033203
        },
        {
            "genre": "Rock",
            "prediction": 0.32703226804733276
        },
        {
            "genre": "Easy Listening",
            "prediction": 0.0001451242424082011
        },
        {
            "genre": "Soul-RnB",
            "prediction": 0.011797893792390823
        },
        {
            "genre": "Electronic",
            "prediction": 0.04490466043353081
        },
        {
            "genre": "Folk",
            "prediction": 0.2945161759853363
        },
        {
            "genre": "Spoken",
            "prediction": 0.0008763470686972141
        },
        {
            "genre": "Hip-Hop",
            "prediction": 0.00830247811973095
        },
        {
            "genre": "Experimental",
            "prediction": 0.031336601823568344
        },
        {
            "genre": "Instrumental",
            "prediction": 0.03921925276517868
        }
    ]
}
```
### ERRORS:

**Status:** 406 Not Acceptable

audio file is not included (or is not labeled "audio")
```
{
    "error": "Expecting audio file, but no file found"
}
```

**Status:** 400 Bad Request

* audio file length is under model threshold for processing
```
{
    "error": "Track duration was not long enough to process"
}
```
* audio file format not accepted or corrupted file
```
{
    "error": "File type png not accepted"
}
```
* audio file corrupted or not wav format
```
{
    "error": "Unable to extract audio data from this-audio.wav"
}
```

### API RESOURCES
* [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the neural network
* [Librosa](https://librosa.org/doc/latest/index.html) and [FFmpeg](https://ffmpeg.org/) for audio processing
* [NumPy](https://numpy.org/) for data processing
* [Pandas](https://pandas.pydata.org/) for labeling training data

