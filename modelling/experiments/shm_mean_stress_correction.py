import numpy as np, pandas as pd, rainflow, pickle, os, time
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold

DATA_DIR = "/tmp/nebula-ps/PS3/02_Datasets/SHM"
CACHE = "cycles_with_mean.pkl"

labels = pd.read_csv(f"{DATA_DIR}/Train_Labels.csv")
train_files = sorted(labels["filename"].tolist())
y = labels.set_index("filename").loc[train_files, "damage"].values

if os.path.exists(CACHE):
    cycles = pickle.load(open(CACHE, "rb"))
else:
    cycles = {}
    t0 = time.time()
    for fn in train_files:
        data = np.loadtxt(f"{DATA_DIR}/Train/{fn}")
        cyc = list(rainflow.extract_cycles(data))
        rngs = np.array([c[0] for c in cyc])
        means = np.array([c[1] for c in cyc])
        counts = np.array([c[2] for c in cyc])
        cycles[fn] = (rngs/2.0, means, counts)
    pickle.dump(cycles, open(CACHE, "wb"))
    print(f"cached in {time.time()-t0:.1f}s")

def S_of_m_alpha(amps, means, counts, m, alpha):
    eff = np.clip(amps + alpha*means, 0, None)
    return float(np.sum(counts * np.power(eff, m)))

def fit_C(S_vals, y_true):
    def mape_given_C(C):
        pred = S_vals/C
        return float(np.mean(np.abs(y_true-pred)/y_true))
    C0 = np.median(S_vals/np.clip(y_true,1e-6,None))
    res = minimize_scalar(mape_given_C, bounds=(C0*1e-4, C0*1e4), method="bounded", options={"xatol": C0*1e-8})
    return res.x, res.fun

kf = KFold(n_splits=5, shuffle=True, random_state=0)
folds = list(kf.split(train_files))

results = []
for m in [4.0, 4.5, 5.0, 5.5, 6.0]:
    for alpha in [-0.3,-0.2,-0.1,-0.05,0.0,0.05,0.1,0.2,0.3]:
        S_all = {fn: S_of_m_alpha(*cycles[fn], m, alpha) for fn in train_files}
        fold_mapes = []
        for tr_idx, va_idx in folds:
            tr_files = [train_files[i] for i in tr_idx]
            va_files = [train_files[i] for i in va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            S_tr = np.array([S_all[f] for f in tr_files])
            C, _ = fit_C(S_tr, y_tr)
            S_va = np.array([S_all[f] for f in va_files])
            pred = S_va/C
            mape = float(np.mean(np.abs(y_va-pred)/y_va))
            fold_mapes.append(mape)
        results.append((m, alpha, np.mean(fold_mapes)))

res_df = pd.DataFrame(results, columns=["m","alpha","cv_mape"])
res_df = res_df.sort_values("cv_mape")
print(res_df.head(15).to_string(index=False))
print("\nbaseline (m=5.0, alpha=0.0):")
print(res_df[(res_df.m==5.0)&(res_df.alpha==0.0)])
