"""
SHM (Structural Health Monitoring) predictor.

Physics-based, not a trained black-box model: rainflow-counts each uploaded stress time series,
then applies Miner's linear damage rule with constants (m, C) fit against the 64 labelled training
files (see the modelling scratch folder's build_shm_model.py / verify.py for how they were derived
and cross-validated).

    D = S(m) / C ,   S(m) = sum_i  count_i * (range_i / 2) ** m

Cross-validated performance on the training set (5-fold repeated across 5 seeds, and leave-one-out):
MAPE ~2.5%, i.e. a subsystem score of ~0.975 on the metric max(0, 1-MAPE).

Input files: one raw stress reading per line, no header, e.g. test03.csv.

--------------------------------------------------------------------------------------------
OPERATOR LAYER
--------------------------------------------------------------------------------------------
A raw damage number like "0.4409" means nothing to a maintainer. Two things make it actionable,
and both are derived from the data we were actually given rather than invented thresholds:

1. WHAT FRACTION OF THE FATIGUE BUDGET THIS SEGMENT ATE. Miner's rule defines failure at D = 1.0,
   so D is already a fraction of the total budget. At a constant loading rate, 1/D equivalent
   segments remain before the limit is reached. That extrapolation assumes this exact loading
   repeats forever, which is why it's labelled as a rate, not a countdown.

2. HOW IT COMPARES TO THE HEALTHY BASELINE. The Info Kit states every training sample is a
   healthy operating condition, so the 64 training labels are a reference distribution of what
   normal looks like on this line. A segment in the top 5% of that distribution is unusually
   severe loading — not damage, but a rougher ride than the fleet normally sees, which is exactly
   what SHM exists to surface.

Bands below are the percentiles of those 64 healthy labels, computed once and hard-coded so the
app needs no training data at runtime.
"""
import numpy as np
import pandas as pd
import rainflow

SUBSYSTEM_NAME = "SHM (Structural Health Monitoring)"
OUTPUT_FILENAME = "shm_predictions.csv"
OUTPUT_COLUMNS = ["file_id", "prediction"]
ACCEPTED_FILE_TYPES = ["csv", "txt"]

# Fitted constants. C is carried at full precision rather than the 6-significant-figure value it
# was originally quoted at: rounding it shifted every prediction by a constant factor of ~3e-7,
# which is meaningless next to the model's 2.5% MAPE but meant the app and the standalone
# modelling script disagreed in the last few decimal places. They now agree exactly.
M_EXPONENT = 5.0
C_CONSTANT = 735168789.7249526

# Percentiles of the 64 labelled TRAINING damage values (all healthy operating conditions).
BASELINE_P25 = 0.045889
BASELINE_MEDIAN = 0.098931
BASELINE_P75 = 0.387633
BASELINE_P95 = 0.780506
BASELINE_MAX = 0.928339


def _predict_damage_from_series(stress_series: np.ndarray) -> float:
    cycles = list(rainflow.count_cycles(stress_series))
    if not cycles:
        return 0.0
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[1] for c in cycles])
    amps = ranges / 2.0
    S = float(np.sum(counts * np.power(amps, M_EXPONENT)))
    return S / C_CONSTANT


def predict(uploaded_files) -> pd.DataFrame:
    """uploaded_files: list of Streamlit UploadedFile objects (or anything np.loadtxt can read),
    each containing one raw stress reading per line, no header.

    Returns a DataFrame whose FIRST TWO columns are exactly file_id, prediction (the submission
    schema for shm_predictions.csv). Any further columns are operator-facing extras that the app
    strips before exporting the submission CSV.
    """
    rows = []
    for f in uploaded_files:
        try:
            data = np.loadtxt(f)
        except Exception:
            raise ValueError(
                f"'{f.name}' is not an SHM stress recording. SHM expects one raw stress reading "
                f"per line with no header (test01.csv ... test16.csv, about 6 MB each). A file "
                f"with column headers - a Rail recording, a labels file, or a predictions file "
                f"this app produced - will fail here.")
        if data.ndim != 1:
            raise ValueError(
                f"'{f.name}' has {data.shape[1]} columns; SHM expects a single column of stress "
                f"values. A 129-column file is a Rail Corrugation recording - select the Rail "
                f"Corrugation subsystem for it.")
        damage = _predict_damage_from_series(data)
        rows.append({
            "file_id": f.name,
            "prediction": damage,
            "pct_of_fatigue_limit": damage * 100.0,
            "segments_to_limit": (1.0 / damage) if damage > 0 else float("inf"),
            "vs_typical_segment": (damage / BASELINE_MEDIAN) if BASELINE_MEDIAN else float("nan"),
            "peak_stress": float(np.max(np.abs(data))) if len(data) else 0.0,
        })
    return pd.DataFrame(rows)


def interpret(row) -> dict:
    """Turn one result row into an operator-facing verdict. Returns status/headline/action/evidence.

    `status` is one of good / warning / serious / critical — the app pairs it with an icon and a
    written label so the colour is never carrying the meaning by itself.
    """
    d = float(row["prediction"])
    pct = d * 100.0
    segs = row["segments_to_limit"]
    ratio = row["vs_typical_segment"]

    if d >= BASELINE_MAX:
        status = "critical"
        band = "Beyond healthy baseline"
        headline = f"Loading severity above anything in the healthy reference set ({pct:.0f}% of fatigue budget)"
        action = ("Flag this segment for engineering review. Confirm the measurement point is "
                  "reading correctly, then check the route section and load condition it was "
                  "recorded under before treating it as a real structural finding.")
    elif d >= BASELINE_P95:
        status = "serious"
        band = "Severe loading (top 5%)"
        headline = f"Severe loading — this segment consumed {pct:.0f}% of the fatigue budget"
        action = ("Schedule an inspection of this measurement point. At this loading rate the "
                  f"Miner's limit is reached in about {segs:.1f} more equivalent segments.")
    elif d >= BASELINE_P75:
        status = "warning"
        band = "Heavy loading (top 25%)"
        headline = f"Heavy loading — {pct:.0f}% of the fatigue budget consumed in this segment"
        action = ("No immediate action. Keep this measurement point on the watch list and "
                  "re-check on the next download; repeated segments at this level shorten the "
                  "inspection interval.")
    elif d >= BASELINE_P25:
        status = "good"
        band = "Typical loading"
        headline = f"Normal loading — {pct:.1f}% of the fatigue budget consumed"
        action = "No action. Continue routine monitoring."
    else:
        status = "good"
        band = "Light loading (bottom 25%)"
        headline = f"Light loading — {pct:.1f}% of the fatigue budget consumed"
        action = "No action. Continue routine monitoring."

    evidence = [
        f"Cumulative damage D = {d:.4f} (Miner's rule; failure is defined at D = 1.0)",
        f"{ratio:.1f}x the median healthy segment (baseline median D = {BASELINE_MEDIAN:.3f})",
        f"At this loading rate, ~{segs:.1f} equivalent segments to reach the D = 1.0 limit",
        f"Peak absolute stress in the recording: {row['peak_stress']:.1f}",
        f"Healthy-baseline reference range: D = {BASELINE_P25:.3f} (25th pct) to "
        f"{BASELINE_P95:.3f} (95th pct), max observed {BASELINE_MAX:.3f}",
    ]

    return {
        "status": status,
        "band": band,
        "headline": headline,
        "action": action,
        "evidence": evidence,
        "sort_value": d,
        "metric_label": "Damage D",
        "metric_value": f"{d:.4f}",
    }


def batch_chart(df):
    """Damage per file against the healthy-fleet reference levels — the magnitude comparison a
    supervisor scans to find which segments are carrying the load."""
    data = pd.DataFrame({"label": df["file_id"], "value": df["prediction"]})
    return ("magnitude_bars", data, {
        "value_title": "Cumulative damage D",
        "reference_lines": [(BASELINE_MEDIAN, "Healthy median"),
                            (BASELINE_P95, "95th pct healthy")],
        "note": "Dashed lines are the healthy-fleet reference levels, from the 64 labelled "
                "training segments.",
    })


def detail_chart(row):
    """Where this segment sits against the healthy baseline — a single-value gauge is clearer
    than a bar of one, so this shows the distribution position rather than the raw number again.
    """
    d = float(row["prediction"])
    data = pd.DataFrame([
        {"label": "This segment", "value": d, "highlight": True},
        {"label": "Healthy 95th pct", "value": BASELINE_P95, "highlight": False},
        {"label": "Healthy median", "value": BASELINE_MEDIAN, "highlight": False},
        {"label": "Healthy 25th pct", "value": BASELINE_P25, "highlight": False},
    ])
    return ("ranked_bars", data, {
        "value_title": "Cumulative damage D",
        "note": "Where this segment's loading sits against the healthy reference distribution.",
    })
