"""Round 2: narrow in on XGBoost / GB hyperparameters and weighting scheme, since round 1 showed
(a) XGBoost depth=3 balanced beats the current GB, and (b) sample weighting matters a lot but
"balanced" isn't obviously optimal -- try a softer weighting too."""
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/tmp/nebula-ps/PS3/02_Datasets/Rail_Corrugation"

features = pd.read_csv(f"{OUT_DIR}/rail_features.csv")
labels = pd.read_csv(f"{DATA_DIR}/Train_Labels.csv")
df = features.merge(labels, on="filename")
feature_cols = [c for c in df.columns if c not in ("filename", "label")]
X_all = df[feature_cols].values
le = LabelEncoder()
y_all = le.fit_transform(df["label"])

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]  # more seeds this round since std was high


def repeated_cv_macro_f1(fit_predict_fn, X, y, seeds=SEEDS):
    scores = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), dtype=int)
        for tr_idx, va_idx in skf.split(X, y):
            oof[va_idx] = fit_predict_fn(X[tr_idx], y[tr_idx], X[va_idx])
        scores.append(f1_score(y, oof, average="macro"))
    return float(np.mean(scores)), float(np.std(scores))


def soft_weight(y_tr, power):
    """power=0 -> no weighting, power=1 -> full 'balanced', in between -> softer."""
    if power == 0:
        return None
    bal = compute_sample_weight("balanced", y_tr)
    return bal ** power


def make_xgb(depth, lr, n_est, weight_power, subsample=1.0, colsample=1.0, reg_lambda=1.0):
    def fn(X_tr, y_tr, X_va):
        clf = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                 objective="multi:softprob", num_class=3, random_state=0,
                                 eval_metric="mlogloss", verbosity=0, subsample=subsample,
                                 colsample_bytree=colsample, reg_lambda=reg_lambda)
        sw = soft_weight(y_tr, weight_power)
        clf.fit(X_tr, y_tr, sample_weight=sw)
        return clf.predict(X_va)
    return fn


def make_gb(depth, lr, n_est, weight_power, subsample=1.0):
    def fn(X_tr, y_tr, X_va):
        clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                                          random_state=0, subsample=subsample)
        sw = soft_weight(y_tr, weight_power)
        clf.fit(X_tr, y_tr, sample_weight=sw)
        return clf.predict(X_va)
    return fn


candidates = {}
for wp in [0.0, 0.3, 0.5, 0.7, 1.0]:
    candidates[f"XGB_d3_lr0.05_n300_wp{wp}"] = make_xgb(3, 0.05, 300, wp)
for depth in [2, 3, 4]:
    candidates[f"XGB_d{depth}_lr0.05_n300_wp0.5"] = make_xgb(depth, 0.05, 300, 0.5)
for lr in [0.02, 0.03, 0.1]:
    candidates[f"XGB_d3_lr{lr}_n300_wp0.5"] = make_xgb(3, lr, 300, 0.5)
for n_est in [100, 500]:
    candidates[f"XGB_d3_lr0.05_n{n_est}_wp0.5"] = make_xgb(3, 0.05, n_est, 0.5)
candidates["XGB_d3_lr0.05_n300_wp0.5_sub0.8"] = make_xgb(3, 0.05, 300, 0.5, subsample=0.8, colsample=0.8)
candidates["XGB_d3_lr0.05_n300_wp0.5_reg5"] = make_xgb(3, 0.05, 300, 0.5, reg_lambda=5.0)
for wp in [0.0, 0.3, 0.5]:
    candidates[f"GB_d3_lr0.05_n200_wp{wp}"] = make_gb(3, 0.05, 200, wp)

print(f"Testing {len(candidates)} configs, {len(SEEDS)} seeds each...\n")
results = []
t0 = time.time()
for name, fn in candidates.items():
    mean_f1, std_f1 = repeated_cv_macro_f1(fn, X_all, y_all)
    results.append((name, mean_f1, std_f1))
    print(f"  {name:38s} macro F1 = {mean_f1:.4f} +/- {std_f1:.4f}  ({time.time()-t0:.0f}s)")

res_df = pd.DataFrame(results, columns=["model", "mean_macro_f1", "std_macro_f1"]).sort_values("mean_macro_f1", ascending=False)
print("\n=== Ranked (top 15) ===")
print(res_df.head(15).to_string(index=False))
res_df.to_csv(f"{OUT_DIR}/experiment2_results.csv", index=False)
