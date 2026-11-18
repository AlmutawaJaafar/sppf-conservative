"""E22: validation of Algorithm 2 (offline design) against the closed-loop runs of E17.

For each E17 configuration the working filter is run without data to t = T (it is data-independent), and from
its final S, Y, Y_{T-1} the offline quantities of Section 4.5 are computed:
  * the rates nu#_(i) (leading eigenvalues of the pencil (S F Y_{T-1}^{-1} F' S, Y));
  * the steady-state sharp certificate g* = smallest fixed point of f(g) = lmax(g A + N, Y), A = S F Y_{T-1}^{-1} F' S,
    N = S Q S + H'R^{-1}H (bisection; infinite if nu#_(1) >= 1);
  * the predicted steady-state price of the sharp certificate, sqrt(g* tr Y^{-1} / tr P_KF) - 1, with P_KF the exact
    Kalman covariance at T (also data-independent);
  * for n = 240 also the steady state of the scalar-plus-rank-p map (p = 5, 20) with S, Y frozen, by iteration.
The observed prices are those of the closed-loop runs (E17: mean over the last 50 of 600 steps).
usage: python e22_design.py [s240|s600|h600|h1200]  ->  results/e22_design_<which>.csv
"""
import sys, numpy as np, pandas as pd, scipy.linalg as sla
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol
import e17_lowrank as e17

def lmax_pencil(X, Y):
    n = X.shape[0]
    return sla.eigh(X, Y, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]

def job(a):
    n, sw, Pe, m, b, seed, T = a
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    F, Q, J, H, sv = mod.F, mod.Q, mod.HRH, mod.H, mod.sv
    I = np.eye(n); mask = mod.dist <= b; p0 = 0.01
    Yinv = p0 * I; Pk = p0 * I
    for t in range(T):
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I
        Y = S + J; Yprev = Yinv; Yinv = inv_from_chol(chol(Y))
        Pk = sym(F @ Pk @ F.T + Q); Pk = inv_from_chol(chol(inv_from_chol(chol(Pk)) + J))
    A = sym(S @ F @ Yprev @ F.T @ S); N = sym(S @ Q @ S + J)
    nus = e17.gen_top(A, Y, 25)[0]
    c = lmax_pencil(N, Y); trY = np.trace(Yinv); trK = np.trace(Pk)
    out = dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, nus_1=nus[0], nus_2=nus[1], nus_6=nus[5], nus_21=nus[20],
               k_above=int((nus >= 1).sum()))
    if nus[0] < 1:
        lo, hi = 0.0, max(1.0, c / (1 - nus[0])) * 1.0001
        for _ in range(60):                              # h(g) = f(g) - g is convex, positive at 0, negative at hi
            mid = 0.5 * (lo + hi)
            if lmax_pencil(mid * A + N, Y) > mid: lo = mid
            else: hi = mid
        g = hi; out["g_sharp_pred"] = g; out["price_sharp_pred"] = np.sqrt(g * trY / trK) - 1
    else:
        out["g_sharp_pred"] = np.inf; out["price_sharp_pred"] = np.inf
    if n == 240:                                         # steady state of the rank-p map with S, Y frozen
        for p in (5, 20):
            C = Yinv.copy(); g_old = np.inf
            for it in range(3000):
                X = sym(S @ sym(F @ C @ F.T + Q) @ S + J)
                lam, V = e17.gen_top(X, Y, p + 1); g = lam[p]
                C = sym(g * Yinv + (V[:, :p] * (lam[:p] - g)) @ V[:, :p].T)
                if abs(g - g_old) <= 1e-9 * g: break
                g_old = g
            out[f"price_r{p}_pred"] = np.sqrt(np.trace(C) / trK) - 1; out[f"it_r{p}"] = it + 1
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "s240"
    jobs = e17.configs(which)
    with Pool(2) as p: rows = p.map(job, jobs)
    d = pd.DataFrame(rows); d.to_csv(f"results/e22_design_{which}.csv", index=False)
    pd.set_option("display.width", 220); print(d.round(4).head(30).to_string())
