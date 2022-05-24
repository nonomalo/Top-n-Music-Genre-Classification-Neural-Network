const form = document.getElementById("audio-file-form")
const dashboard = document.getElementById("dashboard")
const audioUrl = document.getElementById("audio-url")
const audioFile = document.getElementById("audio-file")
const title = document.getElementById("audio-title")

form.addEventListener("submit", (event) => {

    formData = new FormData()

    if (audioUrl.value !== "") {
        let url = audioUrl.value
        audioUrl.value = ""
        loadDashboard(url)
            .catch((error) => console.log(error))

    } else if (audioFile.files.length > 0) {
        formData.append('file', audioFile.files[0])
    }

    event.preventDefault();
})

async function loadDashboard(url) {
    const data = await runPredictions({audio_url: url});
    dashboard.style.display = "block"
    updateTitle(data)
    displayPlots(data['plots'])
    displayModelData()
    displayPredictions(data['predictions'])
}

async function runPredictions(data) {
    let request_options = {
        method: "POST",
        headers: {
            'content-type': 'application/json'
        },
        body: JSON.stringify(data)
    }
    const response = await fetch('/fetch_data', request_options)
    return response.json()
}

function displayPredictions(plot) {
    const predictions = document.getElementById("predictions")

    let pred = new Image()
    pred.src = 'data:image/png;base64,' + plot
    pred.className = "img-thumbnail mx-auto wide-plot"
    predictions.appendChild(pred)
}

function displayPlots(plots) {
    const waveform = document.getElementById("waveform")
    const spectrogram = document.getElementById("spectrogram")
    const mfccs = document.getElementById("mfccs")

    let wave = new Image()
    wave.src = 'data:image/png;base64,' + plots['wave_plot']
    wave.className = "img-thumbnail mx-auto small-plot"
    waveform.appendChild(wave)

    let spectro = new Image()
    spectro.src = 'data:image/png;base64,' + plots['spectrogram']
    spectro.className = "img-thumbnail mx-auto small-plot"
    spectrogram.appendChild(spectro)

    let mfcc = new Image()
    mfcc.src = 'data:image/png;base64,' + plots['mfcc_plot']
    mfcc.className = "img-thumbnail mx-auto small-plot"
    mfccs.appendChild(mfcc)
}

function displayModelData() {
    const loss = document.getElementById("model-loss")
    const accuracy = document.getElementById("model-accuracy")
    const confusion = document.getElementById("confusion-matrix")

    let mLoss = new Image()
    mLoss.src = "static/img/model-loss.png"
    mLoss.className = "img-thumbnail mx-auto small-plot"
    loss.appendChild(mLoss)

    let mAccuracy = new Image()
    mAccuracy.src = "static/img/model-accuracy.png"
    mAccuracy.className = "img-thumbnail mx-auto small-plot"
    accuracy.appendChild(mAccuracy)

    let mConfusion = new Image()
    mConfusion.src = "static/img/model-confusion-matrix.png"
    mConfusion.className = "img-thumbnail mx-auto small-plot"
    confusion.appendChild(mConfusion)
}

function updateTitle(data) {
    let message = ""
    if (data['artist']) {
        message = "Artist: " + data['artist'] + ' '
    }
    if (data['track']) {
        message = message + "Track: " + data['track']
    } else {
        message = data['title']
    }
    title.innerHTML = message
}

document.addEventListener('DOMContentLoaded', function () {
    dashboard.style.display = 'none'
})