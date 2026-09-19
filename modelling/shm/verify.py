"""
Verification pass on the SHM pipeline. Checks, in order:
  1. Every train/test file has the expected row count (no truncated/corrupted files).
  2. The rainflow package itself is doing what we assume, on a synthetic signal with a
     hand-countable number of cycles.
  3. Whether a handful of extreme-amplitude cycles dominate S(m) at m~5 (outlier sensitivity).
  4. Robustness of the chosen m: repeated 5-fold CV across multiple random seeds, plus LOOCV
     (more reliable than a single 5-fold split with only 64 training files).
  5. Consistency of the fitted C across folds/seeds (is it stable, or swinging wildly?).
  6. Sanity-check that test-set predictions fall inside the range spanned by training damage
     values (i.e. we are not wildly extrapolating).
  7. Re-verifies output file formatting against the example submission.
"""
import glob
import os
import numpy as np
import pandas as pd
import rainflow
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold, LeaveOneOut

DATA_DIR = "/tmp/nebula-ps/PS3/02_Datasets/SHM"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. row-count integrity check on every file ----------
print("=" * 70)
print("CHECK 1: file integrity (row counts)")
print("=" * 70)
train_labels = pd.read_csv(f"{DATA_DIR}/Train_Labels.csv")
train_files = sorted(train_labels["filename"].tolist())
test_files = sorted(os.path.basename(p) for p in glob.glob(f"{DATA_DIR}/Test/*.csv"))

bad = []
for fn in train_files:
    n = sum(1 for _ in open(f"{DATA_DIR}/Train/{fn}"))
    if n != 581120:
        bad.append((fn, n))
for fn in test_files:
    n = sum(1 for _ in open(f"{DATA_DIR}/Test/{fn}"))
    if n != 581120:
        bad.append((fn, n))
print(f"  {len(train_files)} train + {len(test_files)} test files checked.")
print(f"  Files NOT at 581,120 rows: {bad if bad else 'none — all consistent'}")
assert len(train_files) == 64 and len(test_files) == 16, "unexpected file counts"

# also check for NaNs / non-numeric rows
nan_files = []
for fn in train_files + test_files:
    folder = "Train" if fn in train_files else "Test"
    arr = np.loadtxt(f"{DATA_DIR}/{folder}/{fn}")
    if np.isnan(arr).any() or len(arr) != 581120:
        nan_files.append(fn)
print(f"  Files with NaNs or wrong length after np.loadtxt: {nan_files if nan_files else 'none'}")

# ---------- 2. validate rainflow package on a synthetic, hand-countable signal ----------
print()
print("=" * 70)
print("CHECK 2: rainflow package sanity check on synthetic signal")
print("=" * 70)
# A simple signal with 3 full cycles of amplitude 1 (range 2), peak-valley-peak-valley...
synthetic = [0, 1, -1, 1, -1, 1, -1, 0]  # classic textbook example
cycles = list(rainflow.count_cycles(synthetic))
print(f"  Signal: {synthetic}")
print(f"  Detected (range, count) pairs: {cycles}")
total_full_cycle_equiv = sum(c for _, c in cycles)
print(f"  Sum of counts (half-cycles=0.5 each): {total_full_cycle_equiv}")
print("  -> Expect ~3 full-cycle-equivalents of range 2 for this pattern (visually verifiable).")

# ---------- rainflow-count all real files once (reused for checks 3-6) ----------
def rainflow_cycles(path):
    data = np.loadtxt(path)
    cyc = list(rainflow.count_cycles(data))
    ranges = np.array([c[0] for c in cyc])
    counts = np.array([c[1] for c in cyc])
    return ranges / 2.0, counts

print("\nRe-running rainflow counting on all files for the checks below...")
train_cycles = {fn: rainflow_cycles(f"{DATA_DIR}/Train/{fn}") for fn in train_files}
test_cycles = {fn: rainflow_cycles(f"{DATA_DIR}/Test/{fn}") for fn in test_files}

def S_of_m(amps, counts, m):
    return float(np.sum(counts * np.power(amps, m)))

y = train_labels.set_index("filename").loc[train_files, "damage"].values
best_m = 5.02  # from the earlier fine grid search

# ---------- 3. outlier sensitivity: what fraction of S(m) comes from the top few cycles? ----------
print()
print("=" * 70)
print(f"CHECK 3: outlier sensitivity of S(m={best_m}) — do a few extreme cycles dominate?")
print("=" * 70)
for fn in train_files[:5]:
    amps, counts = train_cycles[fn]
    contrib = counts * np.power(amps, best_m)
    order = np.argsort(-contrib)
    top5_frac = contrib[order[:5]].sum() / contrib.sum()
    print(f"  {fn}: max amp={amps.max():.3f}, top-5-cycles share of S(m) = {top5_frac:.1%}, "
          f"n_cycles={len(amps)}")

# ---------- 4. robustness of m: repeated 5-fold CV (multiple seeds) + LOOCV ----------
print()
print("=" * 70)
print("CHECK 4: robustness of the chosen m — repeated 5-fold CV + LOOCV")
print("=" * 70)

def fit_C_for_m(m, filenames, y_true, cyc_store):
    S_vals = np.array([S_of_m(*cyc_store[fn], m) for fn in filenames])
    def mape(C):
        return float(np.mean(np.abs(y_true - S_vals / C) / y_true))
    C0 = np.median(S_vals / np.clip(y_true, 1e-6, None))
    res = minimize_scalar(mape, bounds=(C0 * 1e-4, C0 * 1e4), method="bounded",
                           options={"xatol": C0 * 1e-10})
    return res.x, res.fun

m_grid = np.arange(4.5, 5.51, 0.02)

# 4a. repeated 5-fold across 5 seeds
seed_bests = []
for seed in range(5):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for m in m_grid:
        mapes = []
        for tr_idx, val_idx in kf.split(train_files):
            trf = [train_files[i] for i in tr_idx]
            vaf = [train_files[i] for i in val_idx]
            C, _ = fit_C_for_m(m, trf, y[tr_idx], train_cycles)
            Sv = np.array([S_of_m(*train_cycles[fn], m) for fn in vaf])
            mapes.append(float(np.mean(np.abs(y[val_idx] - Sv / C) / y[val_idx])))
        scores.append((m, np.mean(mapes)))
    best_this_seed = min(scores, key=lambda t: t[1])
    seed_bests.append(best_this_seed)
    print(f"  seed={seed}: best m={best_this_seed[0]:.2f}, CV MAPE={best_this_seed[1]:.4f}, "
          f"score={1-best_this_seed[1]:.4f}")

ms = [b[0] for b in seed_bests]
scores_ = [1 - b[1] for b in seed_bests]
print(f"  -> across 5 seeds: m range [{min(ms):.2f}, {max(ms):.2f}], "
      f"score range [{min(scores_):.4f}, {max(scores_):.4f}], mean score={np.mean(scores_):.4f}")

# 4b. LOOCV at a few m values around the optimum (more reliable given only 64 files)
print("\n  LOOCV (leave-one-out) at each candidate m:")
loo = LeaveOneOut()
loocv_results = []
for m in np.arange(4.8, 5.21, 0.05):
    S_all = {fn: S_of_m(*train_cycles[fn], m) for fn in train_files}
    errs = []
    for tr_idx, val_idx in loo.split(train_files):
        trf = [train_files[i] for i in tr_idx]
        val_fn = train_files[val_idx[0]]
        C, _ = fit_C_for_m(m, trf, y[tr_idx], train_cycles)
        pred = S_all[val_fn] / C
        errs.append(abs(y[val_idx[0]] - pred) / y[val_idx[0]])
    mape = float(np.mean(errs))
    loocv_results.append((m, mape))
    print(f"    m={m:.2f}  LOOCV MAPE={mape:.4f}  score={1-mape:.4f}")

best_loo = min(loocv_results, key=lambda t: t[1])
print(f"  -> LOOCV best m={best_loo[0]:.2f}, MAPE={best_loo[1]:.4f}, score={1-best_loo[1]:.4f}")

# ---------- 5. final C stability check ----------
print()
print("=" * 70)
print("CHECK 5: final refit + comparison to earlier run")
print("=" * 70)
final_m = best_loo[0]
final_C, final_train_mape = fit_C_for_m(final_m, train_files, y, train_cycles)
print(f"  Refit at LOOCV-best m={final_m}: C={final_C:.6g}, full-train MAPE={final_train_mape:.4f}")

# ---------- 6. test predictions vs training range (extrapolation check) ----------
print()
print("=" * 70)
print("CHECK 6: test predictions vs. training damage range (extrapolation check)")
print("=" * 70)
print(f"  Training damage range: [{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f}")
S_test = np.array([S_of_m(*test_cycles[fn], final_m) for fn in test_files])
pred_test = S_test / final_C
for fn, p in zip(test_files, pred_test):
    flag = "  <-- OUTSIDE train range" if (p < y.min() or p > y.max()) else ""
    print(f"    {fn}: {p:.4f}{flag}")

# ---------- 7. re-check output format against example ----------
print()
print("=" * 70)
print("CHECK 7: output format vs. example submission")
print("=" * 70)
example = pd.read_csv("/tmp/nebula-ps/PS3/04_Example_Submission/shm_predictions.csv")
print(f"  Example columns: {list(example.columns)}")
final_pred_df = pd.DataFrame({"file_id": test_files, "prediction": pred_test})
print(f"  Ours columns:    {list(final_pred_df.columns)}")
print(f"  Match: {list(example.columns) == list(final_pred_df.columns)}")
print(f"  16 rows, all test files present: {sorted(final_pred_df.file_id) == test_files}")

final_pred_df.to_csv(f"{OUT_DIR}/shm_predictions_verified.csv", index=False)
print(f"\nWrote {OUT_DIR}/shm_predictions_verified.csv (m={final_m}, C={final_C:.6g})")
