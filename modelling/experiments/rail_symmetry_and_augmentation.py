"""Rail: two untried ideas, both grounded in the physical symmetry of the problem.

The two rail sides are structurally identical — Side I is axle-box positions 1,3,5,7 and Side II
is 2,4,6,8. Nothing about the sensors or the vehicle distinguishes them. The current model does
not exploit that at all, and that costs it twice over:

IDEA 1 — CONTRAST FEATURES.
  The question "is Side I corrugated?" is really "does Side I look worse than Side II on this
  same run?". A same-run comparison controls for speed, track section, train and weather
  automatically, exactly like the cross-car comparison that works so well for ACV. The current
  features are absolute levels per side, so the model has to learn that contrast implicitly from
  49 raw numbers with only 14 Side I examples to learn it from. Handing it the difference and
  ratio directly is free — they are derived from features already computed.

IDEA 2 — MIRROR AUGMENTATION.
  Because the sides are interchangeable, swapping the Side I and Side II features of a training
  file produces another perfectly valid training file, with its label flipped (Side I <-> Side II,
  Normal stays Normal). That doubles the training set and, more importantly, balances the two
  fault classes: 14 Side I + 24 Side II become 38 of each. Side I is the class dragging macro F1
  down, so this attacks the actual bottleneck rather than tuning hyperparameters again.

IDEA 3 — SYMMETRIC BINARY.
  Taken to its conclusion: train ONE detector for "is the focal side corrugated?", show it each
  file twice (once per side as focal), and at predict time run both orientations. Every one of the
  38 fault examples teaches the single detector, instead of being split across two classes.

VALIDATION. Leave-one-out, matching how the shipped 0.850 was measured. The mirrored copy of a
file is always held out together with its original — otherwise the model would train on a mirror
image of the very file it is being tested on, which is leakage and would inflate the score.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

OUT = "/tmp/claude-0/-home-claude/b7e316f9-ef0a-5a55-8b34-e9848c469ecf/scratchpad/rail"
DATA = "/tmp/nebula-ps/PS3/02_Datasets/Rail_Corrugation"

feat = pd.read_csv(f"{OUT}/rail_features.csv")
lab = pd.read_csv(f"{DATA}/Train_Labels.csv")
df = feat.merge(lab, on="filename").reset_index(drop=True)
BASE = [c for c in df.columns if c not in ("filename", "label")]
I_COLS = [c for c in BASE if c.startswith("sideI_")]
II_COLS = [c for c in BASE if c.startswith("sideII_")]
assert len(I_COLS) == len(II_COLS)
PAIRS = [(a, a.replace("sideI_", "sideII_", 1)) for a in I_COLS]
OTHER = [c for c in BASE if not c.startswith("side")]
EPS = 1e-9


def side_view(row, swap):
    """Feature vector with a chosen side as 'focal'. swap=True mirrors the two sides."""

    f, o = [], []
    for a, b in PAIRS:
        fa, ob = (b, a) if swap else (a, b)
        f.append(row[fa]); o.append(row[ob])
    return np.array(f), np.array(o)


def build(rows, swap, contrast):
    F, O, X = [], [], []
    for _, r in rows.iterrows():
        f, o = side_view(r, swap)
        v = list(f) + list(o) + [r[c] for c in OTHER]
        if contrast:
            d = f - o
            q = f / (np.abs(o) + EPS)
            v += list(d) + list(np.clip(q, 0, 50))
        X.append(v)
    return np.array(X)


def mk():
    return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             random_state=0, eval_metric="logloss", verbosity=0)


def mk3():
    return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             objective="multi:softprob", num_class=3, random_state=0,
                             eval_metric="mlogloss", verbosity=0)


CLASSES = ["Normal", "Side I", "Side II"]
y_txt = df["label"].values
n = len(df)


def loocv_threeclass(contrast, mirror):
    """3-class model, optionally with contrast features and mirror augmentation."""
    le = LabelEncoder().fit(CLASSES)
    preds = []
    Xo = build(df, False, contrast)
    Xm = build(df, True, contrast) if mirror else None
    flip = {"Normal": "Normal", "Side I": "Side II", "Side II": "Side I"}
    ym = np.array([flip[t] for t in y_txt]) if mirror else None
    for i in range(n):
        keep = np.arange(n) != i
        Xtr, ytr = Xo[keep], y_txt[keep]
        if mirror:                       # mirror of the held-out file is excluded too
            Xtr = np.vstack([Xtr, Xm[keep]])
            ytr = np.concatenate([ytr, ym[keep]])
        yi = le.transform(ytr)
        m = mk3()
        m.fit(Xtr, yi, sample_weight=compute_sample_weight("balanced", yi))
        preds.append(le.inverse_transform(m.predict(Xo[i:i + 1]))[0])
    return np.array(preds)


def loocv_symmetric(contrast):
    """One binary 'is the focal side corrugated?' detector, shown each file in both orientations."""
    Xo, Xm = build(df, False, contrast), build(df, True, contrast)
    # focal-side label: for the unswapped view the focal side is Side I
    yo = (y_txt == "Side I").astype(int)
    ym = (y_txt == "Side II").astype(int)
    preds = []
    for i in range(n):
        keep = np.arange(n) != i
        Xtr = np.vstack([Xo[keep], Xm[keep]])
        ytr = np.concatenate([yo[keep], ym[keep]])
        m = mk()
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        m.set_params(scale_pos_weight=spw)
        m.fit(Xtr, ytr)
        p1 = m.predict_proba(Xo[i:i + 1])[0, 1]     # Side I corrugated?
        p2 = m.predict_proba(Xm[i:i + 1])[0, 1]     # Side II corrugated?
        if p1 < 0.5 and p2 < 0.5:
            preds.append("Normal")
        else:
            preds.append("Side I" if p1 >= p2 else "Side II")
    return np.array(preds)


def report(name, preds):
    f1 = f1_score(y_txt, preds, average="macro", labels=CLASSES)
    per = f1_score(y_txt, preds, average=None, labels=CLASSES)
    print(f"\n{name}")
    print(f"  macro F1 = {f1:.4f}   "
          f"(Normal {per[0]:.3f}  Side I {per[1]:.3f}  Side II {per[2]:.3f})")
    cm = confusion_matrix(y_txt, preds, labels=CLASSES)
    print("  " + pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string().replace("\n", "\n  "))
    return f1


print("Rail symmetry experiments — LOOCV, 272 files")
print(f"features: {len(BASE)} base, {len(PAIRS)} side-pairs")
res = {}
res["A. baseline (shipped, 0.850)"] = report("A. baseline — 3-class, no contrast, no mirror",
                                             loocv_threeclass(False, False))
res["B. + contrast"] = report("B. 3-class + contrast features", loocv_threeclass(True, False))
res["C. + mirror"] = report("C. 3-class + mirror augmentation", loocv_threeclass(False, True))
res["D. + both"] = report("D. 3-class + contrast + mirror", loocv_threeclass(True, True))
res["E. symmetric binary"] = report("E. symmetric binary detector", loocv_symmetric(True))

print("\n" + "=" * 62)
for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
    print(f"  {k:34s} {v:.4f}")
best = max(res, key=res.get)
print(f"\nBest: {best} = {res[best]:.4f}  (shipped baseline {res['A. baseline (shipped, 0.850)']:.4f})")
