"""
Rail Corrugation predictor — 3-class classification (Normal / Side I / Side II) from axle-box
vibration + shock recordings, via wavelength-domain FFT features and a gradient-boosted (XGBoost)
classifier.

Why wavelength, not raw frequency: corrugation shows up as vibration energy at a frequency that
scales with train speed (freq = speed / wavelength) — the same physical defect appears at
different Hz depending on how fast the train happened to be going in that 1-second recording. So
each axle box's signal is FFT'd, converted from frequency to wavelength using the train's speed
(reconstructed from the raw toothed-wheel pulse sensor, since it isn't given directly), and binned
into wavelength ranges matching the corrugation wavelengths described in the Info Kit (a few cm to
several tens of cm) — putting the same physical defect in the same bin regardless of speed.

Side I = axle-box positions 1,3,5,7 (8 cars x 4 = 32 boxes); Side II = positions 2,4,6,8 — judged
independently since one side can show corrugation while the other is normal.

Cross-validated performance (leave-one-out, since Side I is only 14 of 272 training files -- a
5-fold validation fold has just 2-3 Side I examples, too few to compare models reliably; LOOCV
uses all 272 individually): macro F1 = 0.850 out-of-fold with XGBoost + balanced class weighting,
up from 0.800 for the originally-shipped sklearn GradientBoostingClassifier on the same features
(also tested and rejected: no class weighting at all, 0.825; splitting into two independent
per-side binary detectors, ~0.80; random forest variants, 0.64-0.68). For comparison, always
predicting "Normal" scores ~0.33 on this metric despite ~86% plain accuracy, which is exactly the
trap macro F1 is designed to expose. See train_classifier.py / extract_features.py in the
modelling scratch folder for full derivation and the model-comparison experiments.
"""
import os
import re

import joblib
import numpy as np
import pandas as pd

SUBSYSTEM_NAME = "Rail Corrugation"
OUTPUT_FILENAME = "rail_predictions.csv"
OUTPUT_COLUMNS = ["file_id", "prediction"]
ACCEPTED_FILE_TYPES = ["csv"]

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rail_model.joblib")
_model_bundle = None  # lazy-loaded on first predict() call

WHEEL_DIAMETER_M = 0.85
WHEEL_TEETH = 90
WHEEL_CIRCUMFERENCE_M = np.pi * WHEEL_DIAMETER_M
SAMPLE_RATE_HZ = 10000
WAVELENGTH_BINS = [0.02, 0.05, 0.10, 0.20, 0.50, np.inf]

COL_RE = re.compile(r"^(Vibration|Shock) of bearing in position (\d+) of car (\d+)$")


def _load_model():
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = joblib.load(_MODEL_PATH)
    return _model_bundle


def _estimate_speed_mps(rotating_speed_col: np.ndarray) -> float:
    transitions = np.diff(rotating_speed_col.astype(np.int8))
    rising_edges = int(np.sum(transitions == 1))
    revolutions = rising_edges / WHEEL_TEETH
    duration_s = len(rotating_speed_col) / SAMPLE_RATE_HZ
    revolutions_per_sec = revolutions / duration_s
    return revolutions_per_sec * WHEEL_CIRCUMFERENCE_M


def _wavelength_band_energies(signal: np.ndarray, speed_mps: float) -> np.ndarray:
    n = len(signal)
    fft_vals = np.fft.rfft(signal - np.mean(signal))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)

    mask = freqs > 0.5
    freqs = freqs[mask]
    power = power[mask]

    if speed_mps < 0.3 or len(power) == 0 or power.sum() == 0:
        return np.zeros(len(WAVELENGTH_BINS) - 1)

    wavelengths = speed_mps / freqs
    total_power = power.sum()
    band_energy = np.zeros(len(WAVELENGTH_BINS) - 1)
    for i in range(len(WAVELENGTH_BINS) - 1):
        lo, hi = WAVELENGTH_BINS[i], WAVELENGTH_BINS[i + 1]
        band_mask = (wavelengths >= lo) & (wavelengths < hi)
        band_energy[i] = power[band_mask].sum() / total_power
    return band_energy


def _extract_features(df: pd.DataFrame) -> dict:
    speed_mps = _estimate_speed_mps(df["Rotating speed"].values)

    side_boxes = {"I": [], "II": []}
    for col in df.columns:
        m = COL_RE.match(col)
        if not m:
            continue
        sig_type, position, car = m.group(1), int(m.group(2)), int(m.group(3))
        side = "I" if position % 2 == 1 else "II"
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
        vib_band_matrix, shock_band_matrix = [], []
        vib_rms_list, shock_rms_list = [], []
        for _, vib_col, shock_col in side_boxes[side]:
            if vib_col is not None:
                vib_sig = df[vib_col].values
                vib_band_matrix.append(_wavelength_band_energies(vib_sig, speed_mps))
                vib_rms_list.append(np.sqrt(np.mean(vib_sig.astype(np.float64) ** 2)))
            if shock_col is not None:
                shock_sig = df[shock_col].values
                shock_band_matrix.append(_wavelength_band_energies(shock_sig, speed_mps))
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


# Wavelength bin labels, for reporting which physical defect scale the energy sits at.
BIN_LABELS = ["2-5 cm", "5-10 cm", "10-20 cm", "20-50 cm", "over 50 cm"]

# Below this top-class probability the call is treated as borderline rather than decided.
CONFIDENT_PROB = 0.70


def predict(uploaded_files) -> pd.DataFrame:
    """uploaded_files: list of Streamlit UploadedFile objects, each a 10,000-row x 129-column
    axle-box vibration/shock recording (1 second at 10 kHz).

    Returns a DataFrame whose FIRST TWO columns are exactly file_id, prediction (the submission
    schema). Further columns are operator-facing extras, stripped before the CSV is exported.
    """
    bundle = _load_model()
    model, le, feature_cols = bundle["model"], bundle["label_encoder"], bundle["feature_cols"]

    rows = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, dtype=np.float32)
        except Exception:
            raise ValueError(
                f"'{f.name}' could not be read as a Rail recording. Rail expects the axle-box "
                f"files (Test1.csv ... Test68.csv, about 17 MB each, 129 columns). A labels file "
                f"or a predictions file will fail here.")
        if "Rotating speed" not in df.columns:
            raise ValueError(
                f"'{f.name}' is not a Rail Corrugation recording - it has no 'Rotating speed' "
                f"column (found {len(df.columns)} columns). Rail needs Test1.csv ... Test68.csv "
                f"from Rail_Corrugation/Test, about 17 MB each. Watch out: the SHM files are "
                f"named train01.csv and are about 6 MB with a single column - those belong in "
                f"the SHM subsystem, not here.")
        feats = _extract_features(df)
        X = np.array([[feats[c] for c in feature_cols]])
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = le.inverse_transform([pred_idx])[0]

        speed_mps = feats["speed_mps"]
        # Which wavelength band holds the most vibration energy, per side — this is the physical
        # signature a track engineer reads: corrugation is a defect at a specific wavelength.
        side_peaks = {}
        side_bands = {}
        for side in ["I", "II"]:
            band_means = [float(feats[f"side{side}_vib_band{b}_mean"]) for b in range(len(BIN_LABELS))]
            side_bands[side] = band_means
            side_peaks[side] = (BIN_LABELS[int(np.argmax(band_means))], float(np.max(band_means)))

        rows.append({
            "file_id": f.name,
            "prediction": pred_label,
            "confidence": float(np.max(probs)),
            "speed_kmh": speed_mps * 3.6,
            "p_normal": float(probs[list(le.classes_).index("Normal")]),
            "p_side_I": float(probs[list(le.classes_).index("Side I")]),
            "p_side_II": float(probs[list(le.classes_).index("Side II")]),
            "side_I_peak_wavelength": side_peaks["I"][0],
            "side_II_peak_wavelength": side_peaks["II"][0],
            "side_I_vib_rms": feats["sideI_vib_rms_mean"],
            "side_II_vib_rms": feats["sideII_vib_rms_mean"],
            "_bands": side_bands,
        })

    return pd.DataFrame(rows)


def interpret(row) -> dict:
    """Turn one classified recording into a track-maintenance instruction."""
    label = row["prediction"]
    conf = float(row["confidence"])
    speed = float(row["speed_kmh"])
    confident = conf >= CONFIDENT_PROB

    if label == "Normal":
        if confident:
            status, band = "good", "No corrugation detected"
            headline = "Track section reads normal on both sides"
            action = "No action. Continue routine monitoring on the scheduled interval."
        else:
            status, band = "warning", "Normal, but borderline"
            headline = "Reads normal, but the call is not clear-cut"
            action = ("No immediate action, but don't treat this section as cleared. Re-measure "
                      "on the next run before removing it from the watch list.")
    else:
        side = "Side I (axle-box positions 1, 3, 5, 7)" if label == "Side I" else \
               "Side II (axle-box positions 2, 4, 6, 8)"
        peak = row["side_I_peak_wavelength"] if label == "Side I" else row["side_II_peak_wavelength"]
        if confident:
            status, band = "serious", "Corrugation detected"
            headline = f"Rail corrugation on {label}"
            action = (f"Raise a track inspection for this section, {side}. Energy is concentrated "
                      f"at a {peak} wavelength — check for that corrugation pitch and schedule "
                      f"grinding if confirmed.")
        else:
            status, band = "warning", "Possible corrugation"
            headline = f"Possible corrugation on {label} — needs confirmation"
            action = (f"Add this section to the watch list for {side} rather than dispatching "
                      f"grinding straight away. The model is only {conf * 100:.0f}% confident; "
                      f"a second measurement run should settle it.")

    evidence = [
        f"Class probabilities — Normal {row['p_normal'] * 100:.0f}%, "
        f"Side I {row['p_side_I'] * 100:.0f}%, Side II {row['p_side_II'] * 100:.0f}%",
        f"Train speed during the recording: {speed:.0f} km/h "
        f"(reconstructed from the toothed-wheel pulse sensor)",
        f"Side I — peak vibration energy at a {row['side_I_peak_wavelength']} wavelength, "
        f"RMS {row['side_I_vib_rms']:.1f}",
        f"Side II — peak vibration energy at a {row['side_II_peak_wavelength']} wavelength, "
        f"RMS {row['side_II_vib_rms']:.1f}",
        "Method: each axle box's signal is converted from frequency to wavelength using the "
        "train's own speed, so the same physical defect lands in the same band whether the train "
        "was doing 40 or 80 km/h.",
    ]

    return {
        "status": status,
        "band": band,
        "headline": headline,
        "action": action,
        "evidence": evidence,
        # Faults first, then least-confident calls — the ones a human most needs to look at.
        "sort_value": (0 if label != "Normal" else 1, conf),
        "metric_label": "Classification",
        "metric_value": label,
    }


def detail_chart(row):
    """The wavelength signature — where each rail side's vibration energy actually sits.

    This is the picture a track engineer reads directly: corrugation is a defect with a
    characteristic pitch, so a healthy side is spread broadly while a corrugated one spikes in
    one wavelength band. Showing both sides on the same axes makes the asymmetry the finding is
    based on visible at a glance, instead of asking anyone to trust a class probability.
    """
    bands = row.get("_bands")
    if not bands:
        return None
    data = pd.DataFrame([
        {"band": BIN_LABELS[i], "series": f"Side {side}", "value": bands[side][i]}
        for side in ["I", "II"] for i in range(len(BIN_LABELS))
    ])
    return ("grouped_bands", data, {
        "x_title": "Corrugation wavelength",
        "y_title": "Share of vibration energy",
        "note": "A corrugated side concentrates its energy in one wavelength band; a healthy "
                "side stays spread out. The flagged side is the one with the spike.",
    })


def batch_chart(df):
    """How the inspected sections break down — the fleet-level read for a supervisor."""
    counts = df["prediction"].value_counts()
    order = ["Normal", "Side I", "Side II"]
    status_of = {"Normal": "good", "Side I": "serious", "Side II": "serious"}
    data = pd.DataFrame([
        {"label": lbl, "value": int(counts.get(lbl, 0)), "status": status_of[lbl]}
        for lbl in order if counts.get(lbl, 0) > 0
    ])
    return ("composition", data, {
        "value_title": "Sections inspected",
        "note": "Each recording is one track section, judged independently per rail side.",
    })
