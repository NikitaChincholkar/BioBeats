"""
export_eeg.py
-------------
Reads the UCI "EEG Eye State.arff" dataset (14 brainwave electrodes recorded at
128 samples per second), processes the Left and Right brain hemispheres, and
converts every 0.5-second window into a musical violin note (A4 through G5).

Usage:
    python preprocess/export_eeg.py
"""

import json
import math
import zipfile
from pathlib import Path
import numpy as np
from scipy.io import arff
from scipy.ndimage import uniform_filter1d

# ==============================================================================
# TWEAKABLE CONSTANTS (Edit these to adjust how brainwaves turn into notes)
# ==============================================================================

# Recording sample rate of the UCI EEG Eye State dataset (128 samples per second)
SAMPLE_RATE_HZ = 128

# Length of the rolling average (in seconds) subtracted from each channel to
# remove the large ~4000+ uV DC offset/baseline drift.
BASELINE_WINDOW_SEC = 1.0

# Length of each musical note window (in seconds).
# At 128 Hz, 0.5 seconds is 64 samples per window.
WINDOW_SEC = 0.5

# Percentile range used to scale RMS amplitude into the 0.0 to 1.0 range.
# Values below the 5th percentile clip to 0.0; above the 95th clip to 1.0.
PERCENTILE_RANGE = (5, 95)

# Musical scale settings: starting octave and ordered 7-note natural scale
BASE_OCTAVE = 4
NOTES = [
    f"A{BASE_OCTAVE}",
    f"B{BASE_OCTAVE}",
    f"C{BASE_OCTAVE + 1}",
    f"D{BASE_OCTAVE + 1}",
    f"E{BASE_OCTAVE + 1}",
    f"F{BASE_OCTAVE + 1}",
    f"G{BASE_OCTAVE + 1}",
]

# Column indices for each brain hemisphere in EEG Eye State.arff:
# Left hemisphere  = columns 0-6  (AF3, F7, F3, FC5, T7, P7, O1)
# Right hemisphere = columns 7-13 (O2, P8, T8, FC6, F4, F8, AF4)
LEFT_CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 6]
RIGHT_CHANNEL_INDICES = [7, 8, 9, 10, 11, 12, 13]

# File and folder paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARFF_FILENAME = "EEG Eye State.arff"
EEG_ZIP_PATH = PROJECT_ROOT / "eeg+eye+state.zip"
OUTPUT_PATH = PROJECT_ROOT / "web" / "public" / "data" / "eeg_notes.json"


def find_or_extract_arff() -> Path:
    """
    Finds 'EEG Eye State.arff' inside /data/.
    If it hasn't been unzipped yet, extracts it from eeg+eye+state.zip into /data/
    without modifying any existing files in /data/.
    """
    arff_path = DATA_DIR / ARFF_FILENAME
    if arff_path.exists():
        return arff_path

    if not EEG_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {arff_path} and {EEG_ZIP_PATH} does not exist."
        )

    print(f"Extracting '{ARFF_FILENAME}' from {EEG_ZIP_PATH.name} into {DATA_DIR}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EEG_ZIP_PATH, "r") as zf:
        zf.extract(ARFF_FILENAME, path=DATA_DIR)

    return arff_path


def load_eeg_channels(arff_path: Path) -> np.ndarray:
    """
    Loads the first 14 numeric columns (the EEG electrodes) from the .arff file
    and returns a 2D numpy array of shape (num_samples, 14).
    """
    raw_data, meta = arff.loadarff(str(arff_path))
    channel_names = meta.names()[:14]  # First 14 columns are the EEG sensors

    # Stack the 14 columns into a regular 2D float array: rows = time, cols = sensors
    columns = [np.asarray(raw_data[name], dtype=np.float64) for name in channel_names]
    return np.column_stack(columns)


def remove_baseline(eeg_matrix: np.ndarray) -> np.ndarray:
    """
    Step 1: Remove the baseline.
    Raw EEG values sit around 4000+ uV with small brainwave wiggles on top.
    We subtract a rolling 1-second mean (128 samples) from each sensor column
    so that only the real brainwave fluctuations remain.
    """
    baseline_samples = int(round(BASELINE_WINDOW_SEC * SAMPLE_RATE_HZ))
    # uniform_filter1d computes a fast rolling average along time (axis=0) for each column
    rolling_mean = uniform_filter1d(eeg_matrix, size=baseline_samples, axis=0, mode="nearest")
    return eeg_matrix - rolling_mean


def compute_window_rms(signal_1d: np.ndarray, samples_per_window: int) -> np.ndarray:
    """
    Step 3: Measure amplitude per 0.5-second window using RMS (root mean square).
    Drops any leftover partial window at the very end of the recording.
    """
    num_windows = len(signal_1d) // samples_per_window
    trimmed = signal_1d[: num_windows * samples_per_window]
    windows = trimmed.reshape(num_windows, samples_per_window)
    # RMS = square root of the mean of squared values in each window
    return np.sqrt(np.mean(windows ** 2, axis=1))


def normalize_percentile(rms_values: np.ndarray) -> np.ndarray:
    """
    Step 4: Normalize RMS values to 0.0 - 1.0 using the 5th to 95th percentile.
    This clips extreme motion spikes/outliers so a single eye blink doesn't
    squash all the rest of the brainwaves into a single note.
    """
    p_low, p_high = np.percentile(rms_values, PERCENTILE_RANGE)
    if p_high <= p_low:
        return np.zeros_like(rms_values)
    scaled = (rms_values - p_low) / (p_high - p_low)
    return np.clip(scaled, 0.0, 1.0)


def amp_to_note(amp01: float) -> str:
    """
    Step 5: Pick a musical note from NOTES based on normalized amplitude (0.0 to 1.0).
    index = floor(amp01 * 7), capped at 6 -> ["A4","B4","C5","D5","E5","F5","G5"].
    """
    num_notes = len(NOTES)
    index = min(int(math.floor(amp01 * num_notes)), num_notes - 1)
    return NOTES[index]


def export_eeg() -> Path:
    """
    Runs the full EEG preprocessing pipeline and writes web/public/data/eeg_notes.json.
    """
    arff_path = find_or_extract_arff()
    eeg_matrix = load_eeg_channels(arff_path)

    # 1. Subtract rolling 1-second baseline from each of the 14 channels
    centered_eeg = remove_baseline(eeg_matrix)

    # 2. Group channels into Left hemisphere (cols 0-6) and Right hemisphere (cols 7-13)
    #    and average the channels within each group
    left_signal = np.mean(centered_eeg[:, LEFT_CHANNEL_INDICES], axis=1)
    right_signal = np.mean(centered_eeg[:, RIGHT_CHANNEL_INDICES], axis=1)

    # 3. Measure RMS amplitude per 0.5-second window (64 samples per window)
    samples_per_window = int(round(WINDOW_SEC * SAMPLE_RATE_HZ))
    left_rms = compute_window_rms(left_signal, samples_per_window)
    right_rms = compute_window_rms(right_signal, samples_per_window)

    # 4. Normalize each hemisphere's RMS to 0-1 using its 5th to 95th percentile
    left_amp01 = normalize_percentile(left_rms)
    right_amp01 = normalize_percentile(right_rms)

    # 5. Map each normalized amplitude to a musical note (A4 - G5)
    left_notes = [amp_to_note(float(a)) for a in left_amp01]
    right_notes = [amp_to_note(float(a)) for a in right_amp01]

    # Prepare the output JSON structure (round amplitudes to 3 decimal places for readability)
    output_data = {
        "window_sec": WINDOW_SEC,
        "left": left_notes,
        "right": right_notes,
        "left_amp": [round(float(a), 3) for a in left_amp01],
        "right_amp": [round(float(a), 3) for a in right_amp01],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    total_sec = len(left_notes) * WINDOW_SEC
    print(
        f"Exported {len(left_notes)} windows ({total_sec:.1f}s at {WINDOW_SEC}s/window) "
        f"-> {OUTPUT_PATH}"
    )
    return OUTPUT_PATH


if __name__ == "__main__":
    export_eeg()
