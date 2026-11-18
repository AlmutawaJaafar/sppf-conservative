"""E29: an informative-data configuration (sigma_v = 0.1, m in {32, 64}) at n = 240.

Runs the closed-loop experiment of E17 (exact error moments; minimal, sharp, rank-5 and rank-20 certificates) and the
offline prediction of E22 with the observation noise lowered from 0.5 to 0.1, and reports how much the data reduce
the nodal variances: info = 1 - P_KF[j,j] / Var(x_t)_j at t = T (exact Kalman filter), median and maximum over nodes.
usage: python e29_informative.py [unc]  ->  results/e29_informative.csv
"""
import sys, functools, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, ".")
import sppf_certified as sc, e17_lowrank as e17, e22_design as e22
SV = 0.1
e17.Model = functools.partial(sc.Model, sv=SV); e22.Model = functools.partial(sc.Model, sv=SV)

def info(a):
    n, sw, Pe, m, b, seed, T = a
    mod = sc.Model(*sc.GRIDS[n], Pe=Pe, sw=sw, m=m, seed=seed, sv=SV); F, Q, J = mod.F, mod.Q, mod.HRH
    I = np.eye(n); Sig = 0.01 * I; Pk = 0.01 * I; Yinv = 0.01 * I; M = 0.01 * I; mask = mod.dist <= b; wu = 0.0
    for t in range(T):
        Sig = sc.sym(F @ Sig @ F.T + Q); Pk = sc.sym(F @ Pk @ F.T + Q); Pk = sc.inv_from_chol(sc.chol(sc.inv_from_chol(sc.chol(Pk)) + J))
        Pbar = sc.sym(F @ Yinv @ F.T + Q); S = sc.inv_from_chol(sc.chol(Pbar)) * mask
        S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I
        Y = S + J; Yinv = sc.inv_from_chol(sc.chol(Y)); A = Yinv @ S
        M = sc.sym(A @ sc.sym(F @ M @ F.T + Q) @ A.T + Yinv @ J @ Yinv)
        if t % 10 == 9 or t >= T - 50: wu = max(wu, e17.worst(M, Yinv))     # uncorrected worst-direction ratio
    r = 1 - np.diag(Pk) / np.diag(Sig); return dict(info_median=np.median(r), info_max=r.max(), worst_unc=wu)

def job(a):
    out = e17.job(a); d = e22.job(a)
    out.update({k: d[k] for k in ("nus_1", "price_sharp_pred", "price_r5_pred", "price_r20_pred")}); out.update(info(a)); out["sv"] = SV
    return out

if __name__ == "__main__":
    jobs = [(240, 0.05, Pe, m, b, 0, 600) for Pe in (10.0, 50.0, 90.0) for m in (32, 64) for b in (1, 2, 3, 4)]
    if len(sys.argv) > 1 and sys.argv[1] == "unc":                   # add the uncorrected ratio to stored results
        d = pd.read_csv("results/e29_informative.csv")
        with Pool(2) as p: w = p.map(info, jobs)
        d["worst_unc"] = [x["worst_unc"] for x in w]; d.to_csv("results/e29_informative.csv", index=False); print(d[["Pe", "m", "b", "worst_unc"]]); sys.exit()
    with Pool(1) as p: rows = p.map(job, jobs)
    d = pd.DataFrame(rows); d.to_csv("results/e29_informative.csv", index=False)
    pd.set_option("display.width", 250); print(d.round(4).to_string())
