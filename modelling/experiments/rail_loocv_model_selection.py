"""Round 4: LOOCV (leave-one-out) for the strongest few candidates from rounds 1-2.
With only 14/24 minority-class examples, 5-fold CV has ~2-5 per validation fold -- LOOCV removes
that fold-composition noise entirely (272 individual train/predict runs, one held-out row each),
giving the most stable comparison possible on this dataset size."""
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, confusion_matrix
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
classes = list(le.classes_)

loo = LeaveOneOut()


def loocv_macro_f1(fit_predict_fn, X, y):
    oof = np.zeros(len(y), dtype=int)
    for tr_idx, va_idx in loo.split(X):
        oof[va_idx] = fit_predict_fn(X[tr_idx], y[tr_idx], X[va_idx])
    return f1_score(y, oof, average="macro"), oof


def gb_baseline(X_tr, y_tr, X_va):
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
    sw = compute_sample_weight("balanced", y_tr)
    clf.fit(X_tr, y_tr, sample_weight=sw)
    return clf.predict(X_va)


def xgb_wp1(X_tr, y_tr, X_va):
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             objective="multi:softprob", num_class=3, random_state=0,
                             eval_metric="mlogloss", verbosity=0)
    sw = compute_sample_weight("balanced", y_tr)
    clf.fit(X_tr, y_tr, sample_weight=sw)
    return clf.predict(X_va)


def xgb_wp0(X_tr, y_tr, X_va):
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             objective="multi:softprob", num_class=3, random_state=0,
                             eval_metric="mlogloss", verbosity=0)
    clf.fit(X_tr, y_tr)
    return clf.predict(X_va)


candidates = {
    "GB_baseline (current, shipped)": gb_baseline,
    "XGB_d3_lr0.05_n300_balanced": xgb_wp1,
    "XGB_d3_lr0.05_n300_noweight": xgb_wp0,
}

t0 = time.time()
for name, fn in candidates.items():
    f1, oof = loocv_macro_f1(fn, X_all, y_all)
    cm = confusion_matrix(y_all, oof)
    print(f"{name}: LOOCV macro F1 = {f1:.4f}   ({time.time()-t0:.0f}s elapsed)")
    print(f"  confusion matrix (rows=true,cols=pred) {classes}:")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    print()
