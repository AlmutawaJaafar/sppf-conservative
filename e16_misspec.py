"""E16: model misspecification. The filter (exact KF and Algorithm 1 with the minimal certificate)
uses an assumed process-noise covariance cq*Q and observation-noise covariance cr*R, while the data
come from Q and R. True error moments are computed exactly. Reported: worst-direction ratio
lmax(P^{-1} M) at the end of the run (mean of the last 50 of T steps) and its maximum over time,
for the exact KF, the uncorrected sparse filter and the certified covariance."""
import sys, numpy as np, pandas as pd, scipy.linalg as sla
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol

def lmax(A): return np.linalg.eigvalsh(sym(A))[-1]   # full solver (subset driver can fail)
def lmin(A): return np.linalg.eigvalsh(sym(A))[0]

def worst(P_inv_chol_T, M):          # lmax(P^{-1} M) with P^{-1} = L L'
    return lmax(sym(P_inv_chol_T @ M @ P_inv_chol_T.T))

def job(a):
    cq, cr, b, m, seed, T = a
    Nx, Ny = GRIDS[240]; mod = Model(Nx, Ny, Pe=50.0, sw=0.05, m=m, seed=seed)
    n, F, Qt = mod.n, mod.F, mod.Q; Qf = cq * Qt
    Rt = mod.sv ** 2; Rf = cr * Rt
    J = mod.H.T @ mod.H / Rf; mask = mod.dist <= b; I = np.eye(n); p0 = 0.01
    Yinv = p0 * I; M = p0 * I; gam = 1.0
    Pk = p0 * I; Mk = p0 * I
    rec = []
    for t in range(T):
        # exact KF with the assumed model
        Ppk = sym(F @ Pk @ F.T + Qf); Ypk = inv_from_chol(chol(Ppk)) + J
        Pk = inv_from_chol(chol(Ypk)); Ak = Pk @ inv_from_chol(chol(Ppk)); Kk = Pk @ mod.H.T / Rf
        Mk = sym(Ak @ sym(F @ Mk @ F.T + Qt) @ Ak.T + Rt * (Kk @ Kk.T))
        # Algorithm 1 with the assumed model
        Pbar = sym(F @ Yinv @ F.T + Qf); Sp = inv_from_chol(chol(Pbar)); B = Sp * mask
        lmB = lmin(B); s = max(0.0, 1e-8 * np.mean(np.diag(B)) - lmB); S = B + s * I
        LS = chol(S); Y = S + J; LY = chol(Y); Yn = inv_from_chol(LY)
        gam = max(1.0, (1 + 1e-10) * lmax(LS.T @ (gam * (Pbar - Qf) + Qf) @ LS))
        A = Yn @ S; K = Yn @ mod.H.T / Rf
        M = sym(A @ sym(F @ M @ F.T + Qt) @ A.T + Rt * (K @ K.T)); Yinv = Yn
        rk = worst(chol(Ypk).T, Mk); ru = worst(LY.T, M)
        rec.append((rk, ru, ru / gam, gam))
    r = np.array(rec); e = r[-50:].mean(0)
    return dict(cq=cq, cr=cr, b=b, m=m, seed=seed, kf_end=e[0], unc_end=e[1], cert_end=e[2], gamma_end=e[3],
                kf_max=r[:, 0].max(), unc_max=r[:, 1].max(), cert_max=r[:, 2].max())

if __name__ == "__main__":
    jobs = [(cq, cr, b, m, seed, 600) for (cq, cr) in [(1, 1), (0.5, 1), (0.25, 1), (2, 1), (1, 0.5), (1, 2)]
            for b in (2, 3) for m in (8, 32) for seed in (0, 1)]
    with Pool(2) as p: rows = p.map(job, jobs)
    d = pd.DataFrame(rows); d.to_csv("results/e16_misspec.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["cq", "cr"])[["kf_end", "unc_end", "cert_end", "cert_max", "gamma_end"]].agg(["min", "max"]).round(3))
