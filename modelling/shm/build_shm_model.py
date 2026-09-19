"""
SHM subsystem — cumulative fatigue damage regression via rainflow counting + Miner's rule.

Pipeline:
  1. Rainflow-count each stress time series -> list of (range, count) pairs (ASTM E1049 algorithm,
     via the `rainflow` package). This is done once per file; count is independent of the unknown
     S-N constants.
  2. Miner's rule:  D = sum_i n_i / N_i ,  N_i = C / sigma_i^m   (sigma_i = amplitude = range_i / 2)
     => D = (1/C) * sum_i  n_i * sigma_i^m   =  S(m) / C
     S(m) can be computed for any m from the stored (range, count) pairs without re-running
     rainflow, so we grid-search m and, for each m, fit the single scalar C that minimises MAPE
     against the 64 known training damage values (closed-form-ish 1-D search, since MAPE is what
     the subsystem is actually scored on).
  3. 5-fold CV on the 64 training files to pick (m, C) that generalise, not just fit train.
  4. Refit on all 64 training files with the chosen m, apply to the 16 test files.

Outputs: shm_predictions.csv (file_id, prediction) in this directory.
"""
import glob
import os
import time
import numpy as np
import pandas as pd
import rainflow
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold

DATA_DIR = "/tmp/nebula-ps/PS3/02_Datasets/SHM"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Step 1: rainflow-count every train + test file ----------

def rainflow_cycles(path):
    data = np.loadtxt(path)
    cycles = list(rainflow.count_cycles(data))  # [(range, count), ...]
    ranges = np.array([c[0] for c in cycles])
    counts = np.array([c[1] for c in cycles])
    amps = ranges / 2.0
    return amps, counts


def S_of_m(amps, counts, m):
    # sum_i n_i * sigma_i^m
    return float(np.sum(counts * np.power(amps, m)))


print("Step 1: rainflow-counting all Train + Test files (581,120 samples each)...")
t0 = time.time()

train_labels = pd.read_csv(f"{DATA_DIR}/Train_Labels.csv")
train_files = sorted(train_labels["filename"].tolist())
test_files = sorted(os.path.basename(p) for p in glob.glob(f"{DATA_DIR}/Test/*.csv"))

train_cycles = {}
for fn in train_files:
    train_cycles[fn] = rainflow_cycles(f"{DATA_DIR}/Train/{fn}")

test_cycles = {}
for fn in test_files:
    test_cycles[fn] = rainflow_cycles(f"{DATA_DIR}/Test/{fn}")

print(f"  done in {time.time() - t0:.1f}s  ({len(train_files)} train, {len(test_files)} test files)")

# ---------- Step 2+3: grid-search m, fit C per fold, cross-validate ----------

y = train_labels.set_index("filename").loc[train_files, "damage"].values

def fit_C_for_m(m, filenames, y_true):
    """Given m, find the scalar C minimising MAPE on these files (1-D search)."""
    S_vals = np.array([S_of_m(*train_cycles[fn], m) for fn in filenames])

    def mape_given_C(C):
        pred = S_vals / C
        return float(np.mean(np.abs(y_true - pred) / y_true))

    # C must be positive; S/C should be roughly the scale of y (~0.1-1), so search a wide range
    C0 = np.median(S_vals / np.clip(y_true, 1e-6, None))
    res = minimize_scalar(mape_given_C, bounds=(C0 * 1e-4, C0 * 1e4), method="bounded",
                           options={"xatol": C0 * 1e-8})
    return res.x, res.fun


print("Step 2: grid-searching Miner's exponent m with 5-fold CV...")
m_grid = np.arange(1.0, 12.01, 0.25)
kf = KFold(n_splits=5, shuffle=True, random_state=0)

cv_results = []
for m in m_grid:
    fold_mapes = []
    for train_idx, val_idx in kf.split(train_files):
        fold_train_files = [train_files[i] for i in train_idx]
        fold_val_files = [train_files[i] for i in val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]

        C, _ = fit_C_for_m(m, fold_train_files, y_train_fold)
        S_val = np.array([S_of_m(*train_cycles[fn], m) for fn in fold_val_files])
        pred_val = S_val / C
        mape = float(np.mean(np.abs(y_val_fold - pred_val) / y_val_fold))
        fold_mapes.append(mape)

    cv_results.append((m, float(np.mean(fold_mapes)), float(np.std(fold_mapes))))

cv_df = pd.DataFrame(cv_results, columns=["m", "cv_mape_mean", "cv_mape_std"])
best_row = cv_df.loc[cv_df["cv_mape_mean"].idxmin()]
best_m = float(best_row["m"])
print(cv_df.to_string(index=False))
print(f"\n  Best m by 5-fold CV: {best_m}  (CV MAPE = {best_row['cv_mape_mean']:.4f}, "
      f"score = {1 - best_row['cv_mape_mean']:.4f})")

# ---------- Step 4: refit on ALL 64 training files at best_m, predict test ----------

print(f"\nStep 3: refitting C at m={best_m} on all 64 training files...")
final_C, final_train_mape = fit_C_for_m(best_m, train_files, y)
print(f"  Final C = {final_C:.6g}   full-train-set MAPE = {final_train_mape:.4f}  "
      f"(score = {1 - final_train_mape:.4f})  <- optimistic, CV number above is the honest estimate")

S_train_final = np.array([S_of_m(*train_cycles[fn], best_m) for fn in train_files])
pred_train_final = S_train_final / final_C
diag = pd.DataFrame({"filename": train_files, "true_damage": y, "pred_damage": pred_train_final})
diag["abs_pct_err"] = (diag["pred_damage"] - diag["true_damage"]).abs() / diag["true_damage"] * 100
diag.to_csv(f"{OUT_DIR}/train_diagnostics.csv", index=False)
print(f"  Per-file training diagnostics written to train_diagnostics.csv "
      f"(worst 5 by error:)")
print(diag.sort_values("abs_pct_err", ascending=False).head(5).to_string(index=False))

print("\nStep 4: predicting the 16 held-out test files...")
S_test = np.array([S_of_m(*test_cycles[fn], best_m) for fn in test_files])
pred_test = S_test / final_C
pred_df = pd.DataFrame({"file_id": test_files, "prediction": pred_test})
pred_df.to_csv(f"{OUT_DIR}/shm_predictions.csv", index=False)
print(pred_df.to_string(index=False))
print(f"\nWrote {OUT_DIR}/shm_predictions.csv")

cv_df.to_csv(f"{OUT_DIR}/cv_results.csv", index=False)
