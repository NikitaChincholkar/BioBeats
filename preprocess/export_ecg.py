"""
export_ecg.py
-------------
Reads heartbeat annotations from the MIT-BIH Arrhythmia Database using the
`wfdb` library and exports them to a clean JSON file for the web audio player.

Usage:
    python preprocess/export_ecg.py          # Exports default record 100
    python preprocess/export_ecg.py 119      # Exports record 119 (many abnormal beats)
"""

import json
import sys
import zipfile
from pathlib import Path
import wfdb

# ==============================================================================
# TWEAKABLE CONSTANTS (Edit these to change which heartbeats are exported)
# ==============================================================================

# Which MIT-BIH patient record to export if none is given on the command line.
# "100" has mostly normal heartbeats; "119" has frequent premature ventricular contractions (V).
DEFAULT_RECORD = "100"

# How many seconds of the recording to export (from the start of the file).
MAX_DURATION_SEC = 120.0

# Doctor annotation symbols that represent real heartbeats.
# Non-beat markers (like '+' for rhythm change or '~' for signal noise) are skipped.
VALID_BEAT_SYMBOLS = {
    "N",  # Normal beat
    "L",  # Left bundle branch block beat
    "R",  # Right bundle branch block beat
    "B",  # Bundle branch block beat (unspecified)
    "A",  # Atrial premature beat
    "a",  # Aberrated atrial premature beat
    "J",  # Nodal (junctional) premature beat
    "S",  # Supraventricular premature or ectopic beat
    "V",  # Premature ventricular contraction (PVC)
    "r",  # R-on-T premature ventricular contraction
    "F",  # Fusion of ventricular and normal beat
    "e",  # Atrial escape beat
    "j",  # Nodal (junctional) escape beat
    "n",  # Supraventricular escape beat
    "E",  # Ventricular escape beat
    "/",  # Paced beat
    "f",  # Fusion of paced and normal beat
    "Q",  # Unclassifiable beat
}

# File and folder paths (relative to the project root folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MITDB_SUBDIR = DATA_DIR / "mit-bih-arrhythmia-database-1.0.0"
MITDB_ZIP_PATH = PROJECT_ROOT / "mitdb-1.0.0.zip"
OUTPUT_DIR = PROJECT_ROOT / "web" / "public" / "data"


def find_or_extract_record(record_id: str) -> Path:
    """
    Finds the MIT-BIH record files (.hea, .atr, .dat) inside /data/.
    If they haven't been unzipped into /data/ yet, extracts just the needed
    files from mitdb-1.0.0.zip without modifying any existing files in /data/.
    """
    # Check if the record already exists directly in /data/ or /data/mit-bih-arrhythmia-database-1.0.0/
    direct_path = DATA_DIR / record_id
    nested_path = MITDB_SUBDIR / record_id

    if direct_path.with_suffix(".hea").exists() and direct_path.with_suffix(".atr").exists():
        return direct_path

    if nested_path.with_suffix(".hea").exists() and nested_path.with_suffix(".atr").exists():
        return nested_path

    # If not found on disk yet, extract the record files from mitdb-1.0.0.zip into /data/
    if not MITDB_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Could not find record {record_id} in {DATA_DIR} and {MITDB_ZIP_PATH} does not exist."
        )

    print(f"Extracting record {record_id} from {MITDB_ZIP_PATH.name} into {DATA_DIR}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prefix = f"mit-bih-arrhythmia-database-1.0.0/{record_id}."
    with zipfile.ZipFile(MITDB_ZIP_PATH, "r") as zf:
        for member in zf.namelist():
            if member.startswith(prefix):
                target_file = DATA_DIR / member
                if not target_file.exists():
                    zf.extract(member, path=DATA_DIR)

    return nested_path


def export_ecg(record_id: str = DEFAULT_RECORD) -> Path:
    """
    Reads the doctor's beat annotations for `record_id` and saves the first
    MAX_DURATION_SEC seconds of real heartbeats to web/public/data/ecg_<record>.json.
    """
    record_base_path = find_or_extract_record(record_id)

    # Read the ".atr" (annotation) file using wfdb
    ann = wfdb.rdann(str(record_base_path), "atr")

    # `ann.fs` is the sampling rate in Hz (360 samples per second for MIT-BIH)
    sample_rate = float(ann.fs)

    beats = []
    # Pair up each annotation's sample number with its symbol (e.g., "N" or "V")
    for sample_index, symbol in zip(ann.sample, ann.symbol):
        # Convert sample number into seconds from the beginning of the recording
        time_sec = sample_index / sample_rate

        # Stop once we pass the 120-second limit
        if time_sec > MAX_DURATION_SEC:
            break

        # Keep only real heartbeats; skip rhythm labels or noise markers
        if symbol in VALID_BEAT_SYMBOLS:
            beats.append({
                "t": round(time_sec, 3),
                "type": symbol,
            })

    # Prepare the output JSON structure
    output_data = {
        "record": str(record_id),
        "beats": beats,
    }

    # Ensure the output folder exists and write the JSON file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"ecg_{record_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Count normal vs. abnormal beats for a helpful summary message
    normal_count = sum(1 for b in beats if b["type"] == "N")
    abnormal_count = len(beats) - normal_count
    print(
        f"Exported {len(beats)} beats for record {record_id} "
        f"({normal_count} normal 'N', {abnormal_count} abnormal) -> {output_path}"
    )
    return output_path


if __name__ == "__main__":
    # Allow passing one or more record IDs on the command line, e.g.:
    #   python preprocess/export_ecg.py 119
    records_to_run = sys.argv[1:] if len(sys.argv) > 1 else [DEFAULT_RECORD]
    for rec in records_to_run:
        export_ecg(rec)
