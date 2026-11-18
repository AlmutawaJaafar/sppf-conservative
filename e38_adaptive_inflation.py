"""E38: innovation-based adaptive inflation of the sparse filter's reported covariance (baseline, Suppl. Sec. S4).

The practitioner's fix: estimate a scalar inflation from the filter's own innovations and report lambda_t Y_t^{-1}.
As for the certificate, the gain and the estimate are unchanged; only the reported covariance is rescaled.
  NIS_t = d_t' (H S_t^{-1} H' + R)^{-1} d_t / m,   d_t = y_t - H xbar_t   (normalised innovation squared, per trajectory)
  lambda_t = max(1, l_t),  l_t = rho l_{t-1} + (1 - rho) NIS_t,  l_0 = 1,  rho = 0.98 (window about 50 steps).
Monte Carlo (true states, observations, estimates; x_0 ~ N(0, p0 I), xhat_0 = 0), n = 240, Pe = 50, sigma_w = 0.05,
m = 8, layout 0, T = 600, 1000 trajectories. Reported: mean lambda, NEES/n of the inflated covariance, and the
coverage of nominal 95% intervals for the domain mean and for the worst functional of the uncorrected filter
(exact, from the error second moment at that step), mean over the last 50 steps and minimum over all steps (Monte Carlo minima are biased low by sampling error),
for the uncorrected, the adaptively inflated and the basic certified covariance.
usage: python e38_adaptive_inflation.py  ->  results/e38_adaptive_inflation.csv
"""
import sys, numpy as np, pandas as pd, scipy.linalg as sla
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol, lmax, Z95

def run(b, p0=0.01, T=600, N=1000, rho=0.98, seed=3):
    Nx, Ny = GRIDS[240]; mod = Model(Nx, Ny, Pe=50.0, sw=0.05, m=8, seed=0)
    n, F, Q, H, R = mod.n, mod.F, mod.Q, mod.H, mod.R
    mask = mod.dist <= b; I = np.eye(n); rng = np.random.default_rng(seed); Lq = chol(Q)
    Yinv = p0 * I; M = p0 * I; gam = 1.0
    X = np.sqrt(p0) * rng.standard_normal((n, N)); Xh = np.zeros((n, N)); ell = np.ones(N)
    rec = {k: [] for k in ("lam", "nees_tr", "nees_ad", "mean_tr", "mean_ad", "mean_c", "worst_tr", "worst_ad", "worst_c")}
    amean = np.ones(n) / n
    for t in range(T):
        Pbar = sym(F @ Yinv @ F.T + Q); Sp = inv_from_chol(chol(Pbar)); B = Sp * mask
        s = max(0.0, 1e-8 * np.mean(np.diag(B)) - sla.eigh(B, eigvals_only=True, subset_by_index=[0, 0])[0])
        S = B + s * I; LS = chol(S); Sinv = inv_from_chol(LS)
        Y = S + mod.HRH; LY = chol(Y); Yinv_new = inv_from_chol(LY)
        gam = max(1.0, (1 + 1e-10) * lmax(LS.T @ (gam * (Pbar - Q) + Q) @ LS))        # basic certificate
        A = Yinv_new @ S; K = Yinv_new @ H.T / mod.sv ** 2
        M = sym(A @ sym(F @ M @ F.T + Q) @ A.T + mod.sv ** 2 * (K @ K.T))
        # Monte Carlo
        X = F @ X + Lq @ rng.standard_normal((n, N)); Yo = X[mod.sens] + mod.sv * rng.standard_normal((mod.m, N))
        Xb = F @ Xh; D = Yo - Xb[mod.sens]
        Sd = sym(H @ Sinv @ H.T + R); nis = np.einsum("ij,ij->j", D, np.linalg.solve(Sd, D)) / mod.m
        ell = rho * ell + (1 - rho) * nis; lam = np.maximum(1.0, ell)
        Xh = Xb + K @ D; E = Xh - X; Yinv = Yinv_new
        W = sym(LY.T @ M @ LY); w, v = sla.eigh(W, subset_by_index=[n - 1, n - 1]); aw = LY @ v[:, 0]
        q = np.einsum("ij,ij->j", E, LY @ (LY.T @ E)) / n                                  # e' Y e / n
        rec["lam"].append(lam.mean()); rec["nees_tr"].append(q.mean()); rec["nees_ad"].append((q / lam).mean())
        for nm, a in (("mean", amean), ("worst", aw)):
            pr = np.abs(a @ E); vP = a @ Yinv @ a
            rec[f"{nm}_tr"].append(np.mean(pr <= Z95 * np.sqrt(vP)))
            rec[f"{nm}_ad"].append(np.mean(pr <= Z95 * np.sqrt(lam * vP)))
            rec[f"{nm}_c"].append(np.mean(pr <= Z95 * np.sqrt(gam * vP)))
    r = {k: np.array(v) for k, v in rec.items()}
    out = dict(b=b, p0=p0)
    for k, v in r.items():
        out[k + "_last50"] = v[-50:].mean(); out[k + "_min"] = v.min(); out[k + "_argmin"] = int(v.argmin())
    out["lam_max"] = r["lam"].max()
    return out

if __name__ == "__main__":
    rows = [run(1), run(2), run(2, p0=1.0)]
    d = pd.DataFrame(rows); d.to_csv("results/e38_adaptive_inflation.csv", index=False)
    pd.set_option("display.width", 250); print(d.round(3).T.to_string())
