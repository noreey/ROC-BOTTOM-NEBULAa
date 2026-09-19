"""Decide the SHM candidates with LOOCV — deterministic, no fold-composition noise, so a
difference here is real rather than a lucky split."""
import pickle, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

DATA="/tmp/nebula-ps/PS3/02_Datasets/SHM"
cycles=pickle.load(open("train_cycles.pkl","rb"))
labels=pd.read_csv(f"{DATA}/Train_Labels.csv")
files=sorted(labels["filename"]); y=labels.set_index("filename").loc[files,"damage"].values
n=len(files)

def S_single(fn,m,cut=0.0):
    a,c=cycles[fn]
    if cut>0:
        k=a>=cut; a,c=a[k],c[k]
    return float(np.sum(c*np.power(a,m))) if len(a) else 0.0
def S_bi(fn,m1,m2,knee):
    a,c=cycles[fn]; hi,lo=a>=knee,a<knee
    return (float(np.sum(c[hi]*np.power(a[hi],m1)))
            + float(np.sum(c[lo]*np.power(a[lo],m2)))*(knee**(m1-m2)))

def loocv(Sfn):
    S=np.array([Sfn(f) for f in files])
    if not np.all(np.isfinite(S)) or S.min()<=0: return 9.9
    errs=[]
    for i in range(n):
        tr=np.arange(n)!=i
        def mc(C): return float(np.mean(np.abs(y[tr]-S[tr]/C)/y[tr]))
        C0=np.median(S[tr]/np.clip(y[tr],1e-9,None))
        r=minimize_scalar(mc,bounds=(C0*1e-3,C0*1e3),method="bounded",options={"xatol":C0*1e-9})
        errs.append(abs(y[i]-S[i]/r.x)/y[i])
    return float(np.mean(errs))

cands={
 "BASELINE m=5.0":            lambda f: S_single(f,5.0),
 "cut-off 3.0":               lambda f: S_single(f,5.0,3.0),
 "cut-off 4.0":               lambda f: S_single(f,5.0,4.0),
 "cut-off 4.5":               lambda f: S_single(f,5.0,4.5),
 "bilinear knee5 m2=6":       lambda f: S_bi(f,5.0,6.0,5.0),
 "bilinear knee5 m2=7":       lambda f: S_bi(f,5.0,7.0,5.0),
 "bilinear knee5 m2=8":       lambda f: S_bi(f,5.0,8.0,5.0),
 "bilinear knee4 m2=7":       lambda f: S_bi(f,5.0,7.0,4.0),
 "bilinear knee6 m2=7":       lambda f: S_bi(f,5.0,7.0,6.0),
}
res={}
for k,fn in cands.items():
    v=loocv(fn); res[k]=v
    print(f"  {k:26s} LOOCV MAPE = {v:.5f}   score = {1-v:.4f}")
b=min(res,key=res.get)
print(f"\nBest: {b}  ({res[b]:.5f})   vs baseline {res['BASELINE m=5.0']:.5f}")
print(f"Improvement in score: {res['BASELINE m=5.0']-res[b]:+.5f}")
