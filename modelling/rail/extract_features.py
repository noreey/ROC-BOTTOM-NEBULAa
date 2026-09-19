"""
Rail Corrugation feature extraction.

The core physics: corrugation shows up as vibration energy concentrated at a frequency that
scales with train speed (freq = speed / wavelength), not at a fixed frequency -- so raw-Hz
frequency bins would smear the same physical defect across different bins depending on how fast
the train happened to be going in that recording. Instead, each axle box's vibration/shock signal
is FFT'd, converted from frequency to WAVELENGTH (wavelength = speed / freq), and binned into
wavelength ranges that match the corrugation wavelengths described in the Info Kit (a few cm to
several tens of cm) -- so the same physical defect lands in the same bin regardless of speed.

Speed itself isn't given directly -- "Rotating speed" is the raw 0/1 toggle from a 90-tooth wheel
sensor, so it's reconstructed by counting rising edges over the 1-second window and combining with
the wheel's known 0.85 m diameter.

Side I = positions 1,3,5,7 (8 cars x 4 = 32 axle boxes); Side II = positions 2,4,6,8 (32 axle
boxes) -- per the Info Kit's stated geometry, judged independently since a file can show
corrugation on one side while the other is normal.

Output: one row of ~50 compact features per file, cached to a CSV so this only has to run once
per file (the 5.5GB of raw data collapses to a few hundred KB of features).
"""
import glob
import os
import re
import time

import numpy as np
import pandas as pd

DATA_DIR = "/tmp/nebula-ps/PS3/02_Datasets/Rail_Corrugation"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CACHE = f"{OUT_DIR}/rail_features.csv"

WHEEL_DIAMETER_M = 0.85
WHEEL_TEETH = 90
WHEEL_CIRCUMFERENCE_M = np.pi * WHEEL_DIAMETER_M
SAMPLE_RATE_HZ = 10000  # 10,000 Hz, 1 second per file -> 10000 samples

# Wavelength bins in meters, matching "a few cm to dozens of cm" corrugation wavelengths.
WAVELENGTH_BINS = [0.02, 0.05, 0.10, 0.20, 0.50, np.inf]  # 5 bins between these 6 edges

COL_RE = re.compile(r"^(Vibration|Shock) of bearing in position (\d+) of car (\d+)$")


def estimate_speed_mps(rotating_speed_col: np.ndarray) -> float:
    """Reconstruct linear speed from the raw 0/1 toothed-wheel pulse train."""
    transitions = np.diff(rotating_speed_col.astype(np.int8))
    rising_edges = int(np.sum(transitions == 1))
    revolutions = rising_edges / WHEEL_TEETH
    duration_s = len(rotating_speed_col) / SAMPLE_RATE_HZ
    revolutions_per_sec = revolutions / duration_s
    return revolutions_per_sec * WHEEL_CIRCUMFERENCE_M


def wavelength_band_energies(signal: np.ndarray, speed_mps: float) -> np.ndarray:
    """FFT the signal, convert freq axis to wavelength using speed, bin energy by wavelength.
    Returns an array of len(WAVELENGTH_BINS)-1 band energies (fraction of total spectral energy)."""
    n = len(signal)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)

    # skip DC (freq=0); guard against near-zero speed producing absurd wavelengths
    mask = freqs > 0.5
    freqs = freqs[mask]
    power = power[mask]

    if speed_mps < 0.3 or len(power) == 0 or power.sum() == 0:
        # train effectively stationary or no signal -- can't map to wavelength meaningfully
        return np.zeros(len(WAVELENGTH_BINS) - 1)

    wavelengths = speed_mps / freqs
    total_power = power.sum()
    band_energy = np.zeros(len(WAVELENGTH_BINS) - 1)
    for i in range(len(WAVELENGTH_BINS) - 1):
        lo, hi = WAVELENGTH_BINS[i], WAVELENGTH_BINS[i + 1]
        band_mask = (wavelengths >= lo) & (wavelengths < hi)
        band_energy[i] = power[band_mask].sum() / total_power
    return band_energy


def extract_features_from_file(path: str) -> dict:
    df = pd.read_csv(path, dtype=np.float32)

    speed_mps = estimate_speed_mps(df["Rotating speed"].values)

    # map (car, position) -> side
    side_boxes = {"I": [], "II": []}  # list of (vib_col, shock_col)
    for col in df.columns:
        m = COL_RE.match(col)
        if not m:
            continue
        sig_type, position, car = m.group(1), int(m.group(2)), int(m.group(3))
        side = "I" if position % 2 == 1 else "II"
        # collect vib/shock pair per (car, position)
        key = (car, position)
        entry = next((e for e in side_boxes[side] if e[0] == key), None)
        if entry is None:
            entry = [key, None, None]
            side_boxes[side].append(entry)
        idx = 1 if sig_type == "Vibration" else 2
        entry[idx] = col

    n_bins = len(WAVELENGTH_BINS) - 1
    features = {"speed_mps": speed_mps}

    for side in ["I", "II"]:
        vib_band_matrix = []
        shock_band_matrix = []
        vib_rms_list = []
        shock_rms_list = []
        for _, vib_col, shock_col in side_boxes[side]:
            if vib_col is not None:
                vib_sig = df[vib_col].values
                vib_band_matrix.append(wavelength_band_energies(vib_sig, speed_mps))
                vib_rms_list.append(np.sqrt(np.mean(vib_sig.astype(np.float64) ** 2)))
            if shock_col is not None:
                shock_sig = df[shock_col].values
                shock_band_matrix.append(wavelength_band_energies(shock_sig, speed_mps))
                shock_rms_list.append(np.sqrt(np.mean(shock_sig.astype(np.float64) ** 2)))

        vib_band_matrix = np.array(vib_band_matrix) if vib_band_matrix else np.zeros((1, n_bins))
        shock_band_matrix = np.array(shock_band_matrix) if shock_band_matrix else np.zeros((1, n_bins))

        for b in range(n_bins):
            features[f"side{side}_vib_band{b}_mean"] = vib_band_matrix[:, b].mean()
            features[f"side{side}_vib_band{b}_max"] = vib_band_matrix[:, b].max()
            features[f"side{side}_shock_band{b}_mean"] = shock_band_matrix[:, b].mean()
            features[f"side{side}_shock_band{b}_max"] = shock_band_matrix[:, b].max()

        features[f"side{side}_vib_rms_mean"] = float(np.mean(vib_rms_list)) if vib_rms_list else 0.0
        features[f"side{side}_vib_rms_max"] = float(np.max(vib_rms_list)) if vib_rms_list else 0.0
        features[f"side{side}_shock_rms_mean"] = float(np.mean(shock_rms_list)) if shock_rms_list else 0.0
        features[f"side{side}_shock_rms_max"] = float(np.max(shock_rms_list)) if shock_rms_list else 0.0

    return features


def process_folder(folder: str, filenames: list, cache_path: str, time_budget_s: float = 280):
    if os.path.exists(cache_path):
        done = pd.read_csv(cache_path)
        done_files = set(done["filename"])
    else:
        done_files = set()
        done = pd.DataFrame()

    remaining = [f for f in filenames if f not in done_files]
    print(f"{len(done_files)} already done, {len(remaining)} remaining in {folder}")

    start = time.time()
    new_rows = []
    for i, fn in enumerate(remaining):
        if time.time() - start > time_budget_s:
            print(f"Time budget reached, processed {i}/{len(remaining)} this run")
            break
        path = f"{DATA_DIR}/{folder}/{fn}"
        feats = extract_features_from_file(path)
        feats["filename"] = fn
        new_rows.append(feats)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(remaining)} done ({time.time()-start:.0f}s elapsed)")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([done, new_df], ignore_index=True) if len(done) else new_df
        combined.to_csv(cache_path, index=False)
        print(f"Saved {len(combined)} total rows to {cache_path}")
    return len(remaining) - len(new_rows)  # files still left to do


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "train"
    if which == "train":
        labels = pd.read_csv(f"{DATA_DIR}/Train_Labels.csv")
        left = process_folder("Train", labels["filename"].tolist(), FEATURES_CACHE)
    else:
        test_files = sorted(os.path.basename(p) for p in glob.glob(f"{DATA_DIR}/Test/*.csv"))
        left = process_folder("Test", test_files, f"{OUT_DIR}/rail_test_features.csv")
    print(f"\n{left} files still remaining -- rerun this script to continue if >0")
