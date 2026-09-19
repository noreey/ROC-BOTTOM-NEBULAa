"""
ACV predictor — refrigerant-leak localisation via cross-car anomaly ranking.

No supervised training: with only 6 labelled training cases, fitting a model would badly
overfit. Instead, for each uploaded file independently, the same car's telemetry is compared
against its 7 siblings at each timestamp (same conditions -> directly comparable), and cars are
ranked by how anomalous they look on average across every shared numeric parameter, excluding
ambient outside-air temperature readings (shared by the whole train, so they're noise rather than
ACV-specific signal). See build_acv_model.py in the modelling scratch folder for full derivation
and validation (0.979 average rank-decay score across the 6 labelled training cases, vs. 0.5625
expected from a blind guess).

Handles a real edge case found in the training data: some files declare column headers for all
8 cars but only actually log telemetry for a subset -- cars with no usable data are screened out
of the comparison and appended at the end of the ranking rather than skewing the z-scores.

--------------------------------------------------------------------------------------------
OPERATOR LAYER
--------------------------------------------------------------------------------------------
"01|08|04|03|02|05|07|06" tells a technician almost nothing. What they need is: which car do I
open up first, how sure are we, and what did the system actually see. So alongside the ranking
this module reports:

  - the margin between the top-ranked car and the runner-up. A 1.45x gap is a confident call;
    a 1.02x gap means the top two are a coin flip and both should be checked. This is honest
    uncertainty rather than a fabricated confidence percentage.
  - the specific parameters that made the top car look wrong, IN ITS OWN UNITS and with a
    direction -- "indoor temperature running 2.4 above the train average" is a diagnosis a
    technician can act on; "z-score 1.8" is not.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

SUBSYSTEM_NAME = "ACV (Air Conditioning & Ventilation)"
OUTPUT_FILENAME = "acv_predictions.csv"
OUTPUT_COLUMNS = ["file_id", "ranked_cars"]
ACCEPTED_FILE_TYPES = ["xlsx"]

CAR_COL_RE = re.compile(r"^Car (\d+) - (.+)$")

# Margin (top car's anomaly score / runner-up's) above which the localisation is treated as a
# confident single-car call rather than a shortlist.
CONFIDENT_MARGIN = 1.25
AMBIGUOUS_MARGIN = 1.08


def _parse_car_columns(columns):
    car_cols = defaultdict(dict)
    for col in columns:
        m = CAR_COL_RE.match(col)
        if m:
            car_cols[m.group(1)][m.group(2)] = col
    return car_cols


def _rank_cars_in_file(df: pd.DataFrame, data_coverage_threshold: float = 0.3):
    """Returns (ranked_cars, scores, per_param_detail).

    per_param_detail maps car -> {param: (mean_abs_z, mean_signed_raw_offset)} so the app can
    explain a flagged car in the parameter's own units and with a direction.
    """
    car_cols = _parse_car_columns(df.columns)
    all_cars = sorted(car_cols.keys())
    if len(all_cars) < 2:
        return all_cars, {}, {}

    param_sets = [set(car_cols[c].keys()) for c in all_cars]
    common_params = set.intersection(*param_sets)

    valid_col_per_car = {}
    load_flag_col_per_car = {}
    feature_params = []
    for p in common_params:
        pl = p.lower()
        if "information valid" in pl:
            for c in all_cars:
                valid_col_per_car[c] = car_cols[c][p]
        elif pl in ("load halved", "load shedding"):
            for c in all_cars:
                load_flag_col_per_car[c] = car_cols[c][p]
        elif "outdoor" in pl or ("outside" in pl and "temperature" in pl):
            # Ambient outside-air temperature is shared by every car on the same train at the
            # same moment (it's weather, not ACV behaviour) -- including it just adds noise that
            # dilutes the genuinely car-specific signals. Dropping it raised mean rank-decay
            # score on the 6 labelled training cases from 0.938 to 0.979.
            continue
        else:
            feature_params.append(p)

    car_fill_rate = {}
    for c in all_cars:
        fracs = [pd.to_numeric(df[car_cols[c][p]], errors="coerce").notna().mean()
                 for p in feature_params]
        car_fill_rate[c] = float(np.mean(fracs)) if fracs else 0.0
    cars = [c for c in all_cars if car_fill_rate[c] >= data_coverage_threshold]
    no_data_cars = [c for c in all_cars if c not in cars]

    if len(cars) < 2:
        return all_cars, {}, {}

    numeric_params = []
    coerced = {}
    for p in feature_params:
        series_dict = {}
        ok = True
        for c in cars:
            s = pd.to_numeric(df[car_cols[c][p]], errors="coerce")
            if s.notna().mean() < 0.5:
                ok = False
                break
            series_dict[c] = s
        if ok:
            numeric_params.append(p)
            coerced[p] = series_dict

    abs_z_sum = {c: 0.0 for c in cars}
    abs_z_count = {c: 0 for c in cars}
    detail = {c: {} for c in cars}

    for p in numeric_params:
        mat = pd.DataFrame(coerced[p])
        row_mean = mat.mean(axis=1)
        row_std = mat.std(axis=1)
        valid_rows = row_std > 1e-9
        z = mat.sub(row_mean, axis=0).div(row_std, axis=0)
        raw_offset = mat.sub(row_mean, axis=0)

        for c in cars:
            mask = valid_rows.copy()
            if c in valid_col_per_car:
                vf = pd.to_numeric(df[valid_col_per_car[c]], errors="coerce")
                mask &= (vf != 0).fillna(True)
            if c in load_flag_col_per_car:
                lf = pd.to_numeric(df[load_flag_col_per_car[c]], errors="coerce")
                mask &= (lf.fillna(0) == 0)
            col_z = z.loc[mask, c].dropna()
            if len(col_z):
                abs_z_sum[c] += col_z.abs().sum()
                abs_z_count[c] += len(col_z)
                detail[c][p] = (float(col_z.abs().mean()),
                                float(raw_offset.loc[mask, c].dropna().mean()))

    scores = {c: (abs_z_sum[c] / abs_z_count[c] if abs_z_count[c] else 0.0) for c in cars}
    ranked = sorted(cars, key=lambda c: -scores[c]) + no_data_cars
    return ranked, scores, detail


def predict(uploaded_files) -> pd.DataFrame:
    """uploaded_files: list of Streamlit UploadedFile objects, each a .xlsx case file with
    per-car telemetry columns named "Car <NN> - <parameter>".

    Returns a DataFrame whose FIRST TWO columns are exactly file_id, ranked_cars (the submission
    schema). Further columns are operator-facing extras, stripped before the CSV is exported.
    """
    rows = []
    for f in uploaded_files:
        try:
            df = pd.read_excel(f)
        except Exception:
            raise ValueError(
                f"'{f.name}' could not be opened as a spreadsheet. ACV expects the .xlsx case "
                f"files (acv_test_case.xlsx), not a .csv.")
        if not _parse_car_columns(df.columns):
            raise ValueError(
                f"'{f.name}' has no per-car telemetry columns. ACV expects columns named "
                f"'Car 01 - <parameter>' and so on, as in acv_test_case.xlsx.")
        ranked, scores, detail = _rank_cars_in_file(df)

        top = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None
        top_score = scores.get(top, 0.0)
        second_score = scores.get(runner_up, 0.0)
        margin = (top_score / second_score) if second_score > 1e-9 else float("inf")

        # The parameters that made the top car look wrong, strongest first.
        top_params = sorted(detail.get(top, {}).items(), key=lambda kv: -kv[1][0])[:3]
        param_summary = "; ".join(
            f"{p} {'+' if off >= 0 else ''}{off:.2f} vs train avg" for p, (_z, off) in top_params
        )

        rows.append({
            "file_id": f.name,
            "ranked_cars": "|".join(ranked),
            "most_likely_car": top,
            "runner_up_car": runner_up,
            "margin_over_runner_up": margin,
            "top_anomaly_score": top_score,
            "driving_parameters": param_summary,
            "_detail": detail,
            "_scores": scores,
            "_ranked": ranked,
        })
    return pd.DataFrame(rows)


def interpret(row) -> dict:
    """Turn one ranked file into a dispatchable instruction for a technician."""
    top = row["most_likely_car"]
    runner = row["runner_up_car"]
    margin = float(row["margin_over_runner_up"])
    ranked = row.get("_ranked", [])
    detail = row.get("_detail", {}) or {}

    if margin >= CONFIDENT_MARGIN:
        status = "serious"
        band = "Confident localisation"
        headline = f"Car {top} is the most likely faulty unit"
        action = (f"Send a technician to Car {top} first. Its telemetry stands clearly apart "
                  f"from the other seven cars ({margin:.2f}x the next-most-anomalous car), so a "
                  f"single-car inspection is justified before widening the search.")
    elif margin >= AMBIGUOUS_MARGIN:
        status = "warning"
        band = "Probable, with a close second"
        headline = f"Car {top} is most likely, but Car {runner} is close behind"
        action = (f"Start with Car {top}, but have Car {runner} on the same work order — the gap "
                  f"between them is only {margin:.2f}x, so one inspection may not settle it.")
    else:
        status = "warning"
        band = "Ambiguous — shortlist, not a single call"
        shortlist = ", ".join(ranked[:3])
        headline = f"No clear single culprit — treat Cars {shortlist} as a shortlist"
        action = (f"Inspect Cars {shortlist} together. The top candidates are within {margin:.2f}x "
                  f"of each other, which is inside the noise for this method — acting on the "
                  f"top car alone is not supported by the data here.")

    evidence = [f"Full ranking, most to least anomalous: {' > '.join(ranked)}"]
    top_params = sorted(detail.get(top, {}).items(), key=lambda kv: -kv[1][0])[:4]
    for p, (z, off) in top_params:
        direction = "above" if off >= 0 else "below"
        evidence.append(
            f"Car {top} — {p}: averaging {abs(off):.2f} {direction} the train average "
            f"(deviation {z:.2f}x the spread between cars)"
        )
    evidence.append(
        "Method: each car is compared against its 7 siblings at the same timestamp, so route, "
        "weather and schedule are automatically controlled for. Rows where a car reported its "
        "data invalid, or was in load-shedding, are excluded."
    )

    return {
        "status": status,
        "band": band,
        "headline": headline,
        "action": action,
        "evidence": evidence,
        "sort_value": -margin,  # least confident first — those need a human decision
        "metric_label": "Most likely car",
        "metric_value": f"Car {top}",
    }


def detail_chart(row):
    """Anomaly score per car, so the operator can SEE whether the top car stands apart or the
    field is bunched — the difference between "go open Car 01" and "check the top three".
    Returns (kind, tidy_dataframe, meta) for the app to render.
    """
    scores = row.get("_scores") or {}
    ranked = row.get("_ranked") or []
    if not scores:
        return None
    top = row["most_likely_car"]
    data = pd.DataFrame([
        {"label": f"Car {c}", "value": scores.get(c, 0.0), "highlight": (c == top)}
        for c in ranked if c in scores
    ])
    return ("ranked_bars", data, {
        "value_title": "Anomaly score (mean deviation from the other cars)",
        "note": "Taller means the car's telemetry departs further from its seven siblings. "
                "A clear leader is a confident call; a bunched field is a shortlist.",
    })
