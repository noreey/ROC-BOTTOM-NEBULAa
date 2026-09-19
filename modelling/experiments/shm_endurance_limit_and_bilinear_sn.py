"""SHM: two standard fatigue-engineering refinements the current single-slope model ignores.

1. ENDURANCE LIMIT. Real S-N curves have a cut-off below which cycles are treated as
   non-damaging. The current model counts every cycle, including tiny ones.

2. BILINEAR S-N CURVE. Design codes (Eurocode, IIW) use two slopes joined at a knee, not one
   straight line in log-log. Continuity at the knee fixes the second constant, so the extra
   parameters are the second slope m2 and the knee amplitude.

Both are grid-searched with the SAME repeated 5-fold CV used to pick the current m=5.0, so an
improvement has to survive out-of-fold validation, not just fit the training set better.
"""
import pickle, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold

DATA = "/tmp/nebula-ps/PS3/02_Datasets/SHM"
cycles = pickle.load(open("train_cycles.pkl", "rb"))   # fn -> (amps, counts)
labels = pd.read_csv(f"{DATA}/Train_Labels.csv")
files = sorted(labels["filename"]); y = labels.set_index("filename").loc[files, "damage"].values

def S_single(fn, m, cut=0.0):
    a, c = cycles[fn]
    if cut > 0:
        k = a >= cut
        a, c = a[k], c[k]
    return float(np.sum(c * np.power(a, m))) if len(a) else 0.0

def S_bilinear(fn, m1, m2, knee):
    a, c = cycles[fn]
    hi, lo = a >= knee, a < knee
    s = float(np.sum(c[hi] * np.power(a[hi], m1)))
    # continuity at the knee: sigma_k^m1 == K * sigma_k^m2  ->  K = sigma_k^(m1-m2)
    s += float(np.sum(c[lo] * np.power(a[lo], m2))) * (knee ** (m1 - m2))
    return s

def cv_mape(Sfn, seeds=(0, 1, 2)):
    S = np.array([Sfn(f) for f in files])
    if not np.all(np.isfinite(S)) or S.min() <= 0:
        return 9.9
    out = []
    for sd in seeds:
        kf = KFold(5, shuffle=True, random_state=sd)
        fold = []
        for tr, va in kf.split(files):
            def mape_C(C): return float(np.mean(np.abs(y[tr] - S[tr]/C) / y[tr]))
            C0 = np.median(S[tr]/np.clip(y[tr], 1e-9, None))
            r = minimize_scalar(mape_C, bounds=(C0*1e-3, C0*1e3), method="bounded",
                                options={"xatol": C0*1e-9})
            fold.append(float(np.mean(np.abs(y[va] - S[va]/r.x) / y[va])))
        out.append(np.mean(fold))
    return float(np.mean(out))

base = cv_mape(lambda f: S_single(f, 5.0))
print(f"BASELINE  m=5.0, no cut-off           CV MAPE = {base:.5f}  (score {1-base:.4f})")

print("\n--- 1. endurance limit (cycles below this amplitude ignored) ---")
best = (base, None)
for cut in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]:
    v = cv_mape(lambda f, c=cut: S_single(f, 5.0, c))
    flag = "  <-- better" if v < base - 1e-6 else ""
    print(f"  cut-off {cut:4.1f}                      CV MAPE = {v:.5f}{flag}")
    if v < best[0]: best = (v, f"cut-off {cut}")

print("\n--- 2. bilinear S-N (second slope below the knee) ---")
for knee in [3.0, 5.0, 8.0, 12.0]:
    for m2 in [3.0, 7.0, 9.0]:
        v = cv_mape(lambda f, k=knee, mm=m2: S_bilinear(f, 5.0, mm, k))
        flag = "  <-- better" if v < base - 1e-6 else ""
        print(f"  knee {knee:5.1f}  m2={m2:4.1f}              CV MAPE = {v:.5f}{flag}")
        if v < best[0]: best = (v, f"knee {knee} m2 {m2}")

print(f"\nBest overall: {best[1] or 'BASELINE (no change beats it)'}  CV MAPE = {best[0]:.5f}")
