"""ACV: does leave-one-out z-scoring beat the current include-self version?

The flaw in the current method: to judge how anomalous car X is, it compares X against the mean
and spread of ALL EIGHT cars -- including X itself. A genuinely faulty car therefore drags the
very baseline it is being measured against, diluting its own signal. Comparing each car against
the OTHER SEVEN only is the methodologically correct comparison.

Also tested: robust statistics (median / MAD) instead of mean / SD, since with 8 samples one
outlier moves the mean a lot.
"""
import pickle, re, numpy as np, pandas as pd
from collections import defaultdict

CAR_RE = re.compile(r"^Car (\d+) - (.+)$")
DATA = "/tmp/nebula-ps/PS3/02_Datasets/ACV"

def parse(cols):
    d = defaultdict(dict)
    for c in cols:
        m = CAR_RE.match(c)
        if m: d[m.group(1)][m.group(2)] = c
    return d

def rank(df, mode, cov=0.3):
    car_cols = parse(df.columns); all_cars = sorted(car_cols)
    if len(all_cars) < 2: return all_cars
    common = set.intersection(*[set(car_cols[c]) for c in all_cars])
    valid, loadf, feats = {}, {}, []
    for p in common:
        pl = p.lower()
        if "information valid" in pl:
            for c in all_cars: valid[c] = car_cols[c][p]
        elif pl in ("load halved", "load shedding"):
            for c in all_cars: loadf[c] = car_cols[c][p]
        elif "outdoor" in pl or ("outside" in pl and "temperature" in pl):
            continue
        else: feats.append(p)
    fill = {c: float(np.mean([pd.to_numeric(df[car_cols[c][p]], errors="coerce").notna().mean()
                              for p in feats])) if feats else 0.0 for c in all_cars}
    cars = [c for c in all_cars if fill[c] >= cov]; nodata = [c for c in all_cars if c not in cars]
    if len(cars) < 2: return all_cars
    nump, coerced = [], {}
    for p in feats:
        sd, ok = {}, True
        for c in cars:
            s = pd.to_numeric(df[car_cols[c][p]], errors="coerce")
            if s.notna().mean() < 0.5: ok = False; break
            sd[c] = s
        if ok: nump.append(p); coerced[p] = sd
    tot = {c: 0.0 for c in cars}; cnt = {c: 0 for c in cars}
    for p in nump:
        mat = pd.DataFrame(coerced[p])
        n = mat.shape[1]
        for c in cars:
            others = mat.drop(columns=[c])
            if mode == "current":
                centre, spread = mat.mean(axis=1), mat.std(axis=1)
            elif mode == "loo":
                centre, spread = others.mean(axis=1), others.std(axis=1)
            elif mode == "robust":
                centre = mat.median(axis=1)
                spread = (mat.sub(centre, axis=0)).abs().median(axis=1) * 1.4826
            elif mode == "loo_robust":
                centre = others.median(axis=1)
                spread = (others.sub(centre, axis=0)).abs().median(axis=1) * 1.4826
            ok_rows = spread > 1e-9
            z = (mat[c] - centre) / spread
            mask = ok_rows.copy()
            if c in valid:
                vf = pd.to_numeric(df[valid[c]], errors="coerce"); mask &= (vf != 0).fillna(True)
            if c in loadf:
                lf = pd.to_numeric(df[loadf[c]], errors="coerce"); mask &= (lf.fillna(0) == 0)
            zz = z[mask].dropna()
            if len(zz): tot[c] += zz.abs().sum(); cnt[c] += len(zz)
    sc = {c: (tot[c]/cnt[c] if cnt[c] else 0.0) for c in cars}
    return sorted(cars, key=lambda c: -sc[c]) + nodata

labels = pd.read_csv(f"{DATA}/Train_Labels.csv", dtype={"faulty_car": str})
labels["faulty_car"] = labels["faulty_car"].str.zfill(2)
dfs = {fn: pickle.load(open(f".cache_{fn}.pkl", "rb")) for fn in labels["filename"]}

for mode in ["current", "loo", "robust", "loo_robust"]:
    tot, detail = 0.0, []
    for _, r in labels.iterrows():
        rk = rank(dfs[r["filename"]], mode)
        pos = rk.index(r["faulty_car"]) + 1
        s = (8 - (pos - 1)) / 8
        tot += s; detail.append(f"{r['filename'][-6:-5]}:#{pos}")
    print(f"{mode:12s} mean rank-decay = {tot/len(labels):.4f}   {' '.join(detail)}")
