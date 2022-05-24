import librosa, librosa.display
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.figure import Figure
import numpy as np
import base64
from io import BytesIO

matplotlib.use('agg')

HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 13
COLOR = '#2C0955'
CMAP = 'magma'


def create_plots(filename):

    signal, sr = librosa.load(filename, sr=22050)
    wave_plot = create_waveplot(signal, sr, COLOR)
    log_spectrogram = create_log_spectrogram(signal, sr, HOP_LENGTH, N_FFT, CMAP)
    mfcc_plot = create_mfcc_plot(signal, sr, N_FFT, HOP_LENGTH, N_MFCC)

    return {
        'wave_plot': wave_plot,
        'spectrogram': log_spectrogram,
        'mfcc_plot': mfcc_plot
    }


def create_waveplot(signal, sr, color):
    librosa.display.waveshow(signal, sr=sr, color=color)
    plt.rcParams["figure.figsize"] = (6, 5)
    plt.title('Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close("all")
    return base64.b64encode(buffer.getbuffer()).decode('ascii')


def create_log_spectrogram(signal, sr, hop_length, n_fft, cmap):
    stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length, cmap=cmap)
    plt.rcParams["figure.figsize"] = (6, 5)
    plt.title('Log Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close("all")
    return base64.b64encode(buffer.getbuffer()).decode('ascii')


def create_mfcc_plot(signal, sr, n_fft, hop_length, n_mfcc):
    mfccs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    librosa.display.specshow(mfccs, sr=sr, hop_length=hop_length, cmap='magma')

    plt.rcParams["figure.figsize"] = (6, 5)
    plt.title('Mel Frequency Cepstral Coefficients')
    plt.xlabel('Time')
    plt.ylabel('MFCC')
    plt.colorbar()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close("all")
    return base64.b64encode(buffer.getbuffer()).decode('ascii')

def create_prediction_plot(predictions):
    prediction = []
    mappings = []

    for pred_obj in predictions['genres']:
        mappings.append(pred_obj['genre'])
        prediction.append(float(pred_obj['prediction']))

    fig = plt.figure(figsize=(10, 5))
    plt.title("Prediction")

    plt.bar(mappings, prediction, color='#7219B4')
    plt.xticks(rotation=90)
    plt.ylabel("Probability")
    fig.subplots_adjust(bottom=0.55)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close("all")
    return base64.b64encode(buffer.getbuffer()).decode('ascii')