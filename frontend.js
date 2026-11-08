const startBtn = document.getElementById("startBtn");
const pauseBtn = document.getElementById("pauseBtn");
const restartBtn = document.getElementById("restartBtn");
const statusText = document.getElementById("status");
const volumeSlider = document.getElementById("volume");
const volumeLabel = document.getElementById("volumeLabel");

let isPlaying = false;

startBtn.addEventListener("click", () => {
    isPlaying = true;
    statusText.textContent = "Playing...";
    startBtn.disabled = true;
    pauseBtn.disabled = false;
    // Tone.js note: browsers require a click before sound can start.
    // When you add Tone.js, call:  await Tone.start();  here
    // (and put the word "async" before the arrow function).
});

pauseBtn.addEventListener("click", () => {
    isPlaying = false;
    statusText.textContent = "Paused";
    startBtn.disabled = false;
    pauseBtn.disabled = true;
});

restartBtn.addEventListener("click", () => {
    offset = 0;
    isPlaying = true;
    statusText.textContent = "Playing from the beginning...";
    startBtn.disabled = true;
    pauseBtn.disabled = false;
});

volumeSlider.addEventListener("input", () => {
    const value = volumeSlider.value;
    volumeLabel.textContent = value + "%";

    // Later, with Tone.js, convert the percent to decibels:
    // Tone.Destination.volume.value = value == 0 ? -60 : 20 * Math.log10(value / 100);
});