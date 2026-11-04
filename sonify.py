import numpy as np
import wfdb
from scipy.io import arff, wavfile


ECG_RECORD = "C:\\Users\\nikit\\OneDrive\\Documents\\GitHub\\BioBeats\\mitdb-1.0.0.zip"                 # no file extension!
EEG_FILE = "C:\\Users\\nikit\\OneDrive\\Documents\\GitHub\\BioBeats\\eeg+eye+state.zip"
DURATION = 30          # seconds of music to make
SR = 44100             # audio sample rate (CD quality)

# Symbols the MIT-BIH doctors used to mark heartbeats.
# "N" = normal beat. Others (like "V") = abnormal beats.
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# Brain-wave frequency bands (in Hz)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# Notes we are allowed to play (MIDI numbers, C minor pentatonic).
SCALE = [48, 51, 53, 55, 58, 60, 63, 65, 67, 70]


def midi_to_hz(midi_note):
    """Turn a note number into a frequency in Hz."""
    return 440 * 2 ** ((midi_note - 69) / 12)

def add_sound(buffer, sound, start_time, volume=1.0):
    """Mix a short sound into the big audio buffer at a given time."""
    start = int(start_time * SR)
    if start >= len(buffer):
        return
    end = min(start + len(sound), len(buffer))
    buffer[start:end] += sound[: end - start] * volume


# HEART (ECG) -> DRUMS
def load_heartbeats(record_path, seconds):
    """Read beat times (in seconds) and beat types from the MIT-BIH files."""
    fs = wfdb.rdheader(record_path).fs            # samples per second (360)
    ann = wfdb.rdann(record_path, "atr", sampto=int(seconds * fs))
    times, kinds = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol in BEAT_SYMBOLS:
            times.append(sample / fs)
            kinds.append(symbol)
    return times, kinds


def make_kick():
    t = np.arange(int(0.25 * SR)) / SR
    freq = 50 + 100 * np.exp(-t * 30)             # pitch drops fast = "thump"
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) * np.exp(-t * 12)


def make_snare():
    t = np.arange(int(0.2 * SR)) / SR
    noise = np.random.default_rng(0).uniform(-1, 1, len(t))
    return noise * np.exp(-t * 25) * 0.6


def heart_to_drums(times, kinds, total_samples):
    out = np.zeros(total_samples)
    kick, snare = make_kick(), make_snare()
    for t, kind in zip(times, kinds):
        if kind == "N":
            add_sound(out, kick, t, 0.9)          # normal beat -> kick
        else:
            add_sound(out, snare, t, 0.8)         # abnormal beat -> snare
    return out


# BRAIN (EEG) -> SYNTH
def load_eeg(path):
    """Read the .arff file; return an array of shape (time, 14 electrodes)."""
    data, meta = arff.loadarff(path)
    names = list(meta.names())
    # last column is the eye state label, so skip it
    return np.column_stack([data[n] for n in names[:-1]]).astype(float)


def eeg_band_powers(channels, fs=128):
    """For every 1-second chunk, measure how strong each brain-wave band is."""
    # Electrodes 6 and 7 are O1 and O2 (back of head), where alpha is strongest
    signal = channels[:, [6, 7]].mean(axis=1)
    freqs = np.fft.rfftfreq(fs, 1 / fs)
    powers = {band: [] for band in BANDS}

    for w in range(len(signal) // fs):
        chunk = signal[w * fs:(w + 1) * fs]
        chunk = chunk - chunk.mean()              # remove the big constant offset
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(fs))) ** 2
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            powers[band].append(spectrum[mask].sum())

    # Squash each band into 0-1 so it's easy to map to sound
    result = {}
    for band, values in powers.items():
        v = np.log10(np.array(values) + 1e-9)
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)  # ignores wild spikes
        result[band] = np.clip((v - lo) / (hi - lo), 0, 1)
    return result


def eeg_to_synth(powers, total_samples):
    out = np.zeros(total_samples)
    seconds = min(len(powers["alpha"]), total_samples // SR)

    for sec in range(seconds):
        delta = powers["delta"][sec]
        alpha = powers["alpha"][sec]
        beta = powers["beta"][sec]

        # ALPHA picks which note plays
        note = SCALE[int(round(alpha * (len(SCALE) - 1)))]
        freq = midi_to_hz(note)

        # BETA controls brightness: more beta = more high harmonics
        dur = 1.5
        t = np.arange(int(dur * SR)) / SR
        tone = np.zeros_like(t)
        for k in range(1, 7):
            tone += (beta ** (k - 1)) * np.sin(2 * np.pi * freq * k * t) / k
        envelope = np.sin(np.pi * t / dur) ** 2   # smooth fade in and out
        add_sound(out, tone * envelope * 0.15, sec)

        # DELTA controls the volume of a low bass note
        bass_t = np.arange(int(1.2 * SR)) / SR
        bass = np.sin(2 * np.pi * midi_to_hz(36) * bass_t)
        bass *= np.sin(np.pi * bass_t / 1.2) ** 2
        add_sound(out, bass * 0.3 * delta, sec)

    return out


# MAIN: put it all together
def save_wav(filename, audio):
    audio = audio / np.max(np.abs(audio)) * 0.9   # keep it from clipping
    wavfile.write(filename, SR, (audio * 32767).astype(np.int16))
    print("Saved", filename)


if __name__ == "__main__":
    total = DURATION * SR

    times, kinds = load_heartbeats(ECG_RECORD, DURATION)
    print(f"Found {len(times)} heartbeats; {kinds.count('N')} normal")
    drums = heart_to_drums(times, kinds, total)

    eeg = load_eeg(EEG_FILE)
    powers = eeg_band_powers(eeg)
    synth = eeg_to_synth(powers, total)

    save_wav("drums.wav", drums)
    save_wav("synth.wav", synth)
    save_wav("biomusic.wav", drums + synth)