const form = document.getElementById("audio-file-form")
const dashboard = document.getElementById("dashboard")
const audioUrl = document.getElementById("audio-url")
const audioFile = document.getElementById("audio-file")
const status = document.getElementById("status")
const title = document.getElementById("audio-title")
serverURL = "https://top-n-server.uw.r.appspot.com/genre"

form.addEventListener("submit", (event) => {

    dashboard.style.display = "block"
    updateStatus("Fetching audio data ...")
    formData = new FormData()

    if (audioUrl.value !== "") {
        loadDashboard(audioUrl.value)
            .catch((error) => console.log(error))

    } else if (audioFile.files.length > 0) {
        formData.append('file', audioFile.files[0])
    }

    event.preventDefault();
})

async function loadDashboard(url) {
    const data = await fetchWaveFileFromURL({audio_url: url});
    updateTitle(data)
    displayPlots(data['plots'])
    displayPredictions(data['predictions'])
}

function displayPredictions(plot) {
    const predictions = document.getElementById("predictions")

    let pred = new Image()
    pred.src = 'data:image/png;base64,' + plot
    pred.className = "img-thumbnail mx-auto wide-plot"
    predictions.appendChild(pred)
}

async function fetchPredictions(data) {

    let request_options = {
        method: "POST",
        headers: {
            'content-type': 'application/json'
        },
        body: JSON.stringify(data)
    }
    const response = await fetch('/fetch_predictions', request_options)
    return response.json()
}

async function fetchTrackPlots(data) {
    let request_options = {
        method: "POST",
        headers: {
            'content-type': 'application/json'
        },
        body: JSON.stringify(data)
    }
    const response = await fetch('/fetch_plots', request_options)
    return response.json()
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

function updateStatus(message) {
    status.innerHTML = message
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

async function fetchWaveFileFromURL(data) {
    let request_options = {
        method: "POST",
        headers: {
            'content-type': 'application/json'
        },
        body: JSON.stringify(data)
    }
    const response = await fetch('/fetch_audio', request_options)
    return response.json()
}


document.addEventListener('DOMContentLoaded', function() {
    dashboard.style.display = 'none'
})