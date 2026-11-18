"""E27: fixed-shape (scalar-plus-rank-p) certificates, Theorem "Certificates of prescribed shape".

For each configuration the working filter is run without data to t = T (data-independent), S, Y are frozen, and
the scalar-plus-rank-p map of Corollary "Scalar plus rank p" is iterated to a candidate shape
Psi_p = g Y^{-1} + Z Z^T. Reported:
  * test  = lmax(Psi^{-1} Y^{-1} X(Psi) Y^{-1})   (<= 1: Psi is a super-solution, so h* <= 1)
  * r     = lmax(Psi^{-1} Y^{-1} S F Psi F^T S Y^{-1})   (rate of the shape; < 1 => bounded certificate)
  * k     = lmax(Psi^{-1} Y^{-1} (S Q S + J) Y^{-1})
  * hstar = fixed point of phi(h) = lmax(Psi^{-1} Y^{-1} X(h Psi) Y^{-1})  (bisection; <= k/(1-r))
  * rhoAF = spectral radius of the working error map A F, A = Y^{-1} S  (lower bound: r >= rhoAF^2 for every shape)
  * nus_1, nus_(p+1): rates of Proposition "Growth of the scalar part"
  * price_shape = sqrt(hstar * tr Psi / tr P_KF) - 1   (guaranteed steady-state price of the fixed-shape certificate)
usage: python e27_shape.py [s240|s600|s240a|s600a|check] [workers]  (s240a, s600a: layout 0 only, as reported)  ->  results/e27_shape_<which>.csv
"""
import sys, numpy as np, pandas as pd, scipy.linalg as sla
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol

def top_pencil(X, Y, k):
    n = Y.shape[0]; w, V = sla.eigh(X, Y, subset_by_index=[n - k, n - 1]); return w[::-1], V[:, ::-1]

def lmax_rel(Mx, Lc):                     # lmax(Psi^{-1} Mx) with Psi = Lc Lc^T
    Z = sla.solve_triangular(Lc, sla.solve_triangular(Lc, Mx, lower=True).T, lower=True)
    return np.linalg.eigvalsh(sym(Z))[-1]

def job(a):
    n, sw, Pe, m, b, seed, T, ps = a
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    F, Q, J = mod.F, mod.Q, mod.HRH; I = np.eye(n); mask = mod.dist <= b; p0 = 0.01
    Yinv = p0 * I; Pk = p0 * I
    for t in range(T):
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I
        Y = S + J; Yinv = inv_from_chol(chol(Y))
        Pk = sym(F @ Pk @ F.T + Q); Pk = inv_from_chol(chol(inv_from_chol(chol(Pk)) + J))
    AF = Yinv @ S @ F; rhoAF = np.max(np.abs(np.linalg.eigvals(AF)))
    Nh = sym(Yinv @ sym(S @ Q @ S + J) @ Yinv)
    A0 = lambda C: sym(AF @ C @ AF.T)
    nus = top_pencil(sym(S @ F @ Yinv @ F.T @ S), Y, max(ps) + 1)[0]
    rows = []
    for p in ps:
        C = Yinv.copy(); g_old = np.inf
        for it in range(6000):
            X = sym(S @ sym(F @ C @ F.T + Q) @ S + J); lam, V = top_pencil(X, Y, p + 1); g = lam[p]
            C = sym(g * Yinv + (V[:, :p] * (lam[:p] - g)) @ V[:, :p].T) if p > 0 else g * Yinv
            if (not np.isfinite(g)) or g > 1e8 or abs(g - g_old) <= 1e-11 * g: break
            g_old = g
        Psi = C; Lc = chol(Psi)
        r = lmax_rel(A0(Psi), Lc); k = lmax_rel(Nh, Lc); test = lmax_rel(A0(Psi) + Nh, Lc)
        if r < 1:                                            # phi is a contraction: bisection on phi(h) - h
            lo, hi = 0.0, k / (1 - r) * 1.0001
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if lmax_rel(mid * A0(Psi) + Nh, Lc) > mid: lo = mid
                else: hi = mid
            hstar = hi; price = np.sqrt(hstar * np.trace(Psi) / np.trace(Pk)) - 1
        else:
            hstar = np.inf; price = np.inf
        rows.append(dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, p=p, iters=it + 1, nus_1=nus[0], nus_p1=nus[p],
                         rhoAF=rhoAF, test=test, r=r, k=k, hstar=hstar, price_shape=price))
    return rows

CONF = {
    "s240": [(240, 0.05, Pe, m, b, s, 600, (0, 5, 20)) for Pe in (10.0, 50.0, 90.0) for m in (8, 32)
             for b in (1, 2, 3, 4) for s in (0, 1, 2)],
    "s600": [(600, 0.05, 10.0, 8, b, s, 600, (0, 1, 2, 5)) for b in (1, 2, 3) for s in (0, 1, 2)],
    "s240a": [(240, 0.05, Pe, m, b, 0, 600, (0, 5, 20)) for Pe in (10.0, 50.0, 90.0) for m in (8, 32)
              for b in (1, 2, 3, 4)],
    "s600a": [(600, 0.05, 10.0, 8, b, 0, 600, (0, 1, 2, 5)) for b in (1, 2, 3)],
    "check": [(240, 0.05, 50.0, 8, 1, 0, 600, (0, 5, 20)), (600, 0.05, 10.0, 8, 2, 0, 600, (0, 1))],
}
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "check"
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as pool:
        out = [r for rows in pool.map(job, CONF[which]) for r in rows]
    df = pd.DataFrame(out); df.to_csv(f"results/e27_shape_{which}.csv", index=False)
    print(df.to_string(float_format=lambda x: f"{x:.5g}"))
