const form = document.getElementById("audio-file-form")

form.addEventListener("submit", function(event) {

    document.querySelectorAll('.spinner').forEach(function(el) {
        el.style.display = "block"
    })
    document.getElementById('error-message').innerHTML=""
})

document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.spinner').forEach(function(el) {
        el.style.display = "none"
    })
})