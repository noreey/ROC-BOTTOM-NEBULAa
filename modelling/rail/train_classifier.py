"""
Rail Corrugation classifier: trains on the cached wavelength-band features, validated with
LEAVE-ONE-OUT CV (not just 5-fold) because with only 14 Side I / 24 Side II examples, a 5-fold
validation fold holds just 2-5 minority examples -- one flip changes recall by 20-30%, so a single
5-fold split is too noisy to reliably compare models. LOOCV (272 individual train/predict runs)
removes that fold-composition noise and was used here to pick between candidates.

Evaluated on macro F1 (the actual scoring metric, which treats each class equally regardless of
how rare it is -- a model that just predicts "Normal" for everything scores ~0.33 on this metric
despite ~86% plain accuracy, so it's a meaningful check against that trap).

Model history:
  - GradientBoosting (sklearn), balanced sample_weight: LOOCV macro F1 = 0.7997 (originally
    picked from a single 5-fold split showing 0.7967, before LOOCV was used to validate it)
  - XGBoost, same features, balanced sample_weight: LOOCV macro F1 = 0.8501  <- now the default
    (also beat: XGBoost with no class weighting at all, 0.8247; a two-independent-binary-detector
    decomposition by side, ~0.80; RandomForest variants, 0.64-0.68 -- all tested the same way)
  XGBoost's own regularization (per-leaf min child weight, column/row subsampling machinery) turns
  out to matter more than the exact class-weighting scheme on a dataset this small -- sklearn's
  GradientBoostingClassifier doesn't expose those knobs.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/tmp/nebula-ps/PS3/02_Datasets/Rail_Corrugation"

features = pd.read_csv(f"{OUT_DIR}/rail_features.csv")
labels = pd.read_csv(f"{DATA_DIR}/Train_Labels.csv")
df = features.merge(labels, on="filename")
print(f"Loaded {len(df)} training rows")
print(df["label"].value_counts())

feature_cols = [c for c in df.columns if c not in ("filename", "label")]
X = df[feature_cols].values
le = LabelEncoder()
y = le.fit_transform(df["label"])
print("classes:", list(le.classes_))

loo = LeaveOneOut()


def make_xgb():
    return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                              objective="multi:softprob", num_class=3, random_state=0,
                              eval_metric="mlogloss", verbosity=0)


candidates = {
    "RandomForest_balanced": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=6, class_weight="balanced",
        min_samples_leaf=2, random_state=0, n_jobs=-1),
    "GradientBoosting_balanced": lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
    "XGBoost_balanced": make_xgb,
}
uses_sample_weight = {"GradientBoosting_balanced", "XGBoost_balanced"}

best_name, best_score, best_oof = None, -1, None
for name, make_clf in candidates.items():
    oof_pred = np.zeros(len(y), dtype=int)
    for train_idx, val_idx in loo.split(X):
        clf = make_clf()
        if name in uses_sample_weight:
            sw = compute_sample_weight("balanced", y[train_idx])
            clf.fit(X[train_idx], y[train_idx], sample_weight=sw)
        else:
            clf.fit(X[train_idx], y[train_idx])
        oof_pred[val_idx] = clf.predict(X[val_idx])

    macro_f1 = f1_score(y, oof_pred, average="macro")
    print(f"\n=== {name} ===  LOOCV macro F1 = {macro_f1:.4f}")
    print(classification_report(y, oof_pred, target_names=le.classes_, zero_division=0))
    if macro_f1 > best_score:
        best_score, best_name, best_oof = macro_f1, name, oof_pred

print(f"\nBest model: {best_name} with LOOCV macro F1 = {best_score:.4f}")
print("Confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(confusion_matrix(y, best_oof), index=le.classes_, columns=le.classes_))

# refit best model on ALL training data
final_clf = candidates[best_name]()
if best_name in uses_sample_weight:
    sw = compute_sample_weight("balanced", y)
    final_clf.fit(X, y, sample_weight=sw)
else:
    final_clf.fit(X, y)

import joblib
joblib.dump({"model": final_clf, "label_encoder": le, "feature_cols": feature_cols},
            f"{OUT_DIR}/rail_model.joblib")
print(f"\nSaved final model to {OUT_DIR}/rail_model.joblib")

# predict on test set
test_features = pd.read_csv(f"{OUT_DIR}/rail_test_features.csv")
X_test = test_features[feature_cols].values
test_pred = final_clf.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred)

pred_df = pd.DataFrame({"file_id": test_features["filename"], "prediction": test_pred_labels})
pred_df.to_csv(f"{OUT_DIR}/rail_predictions.csv", index=False)
print(f"\nWrote {OUT_DIR}/rail_predictions.csv")
print(pred_df["prediction"].value_counts())
