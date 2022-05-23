const form = document.getElementById("audio-file-form")
const dashboard = document.getElementById("dashboard")
const audioUrl = document.getElementById("audio-url")
const audioFile = document.getElementById("audio-file")

form.addEventListener("submit", (event) => {

    formData = new FormData()

    if (audioUrl.value !== "") {

        // formData.append('audio_url', audioUrl.value)
        fetchWaveFileFromURL({audio_url: audioUrl.value})
            .then((data) => console.log(data))
            .catch((error) => console.log(error))

    } else if (audioFile.files.length > 0) {
        formData.append('file', audioFile.files[0])
    }

    event.preventDefault();
})

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