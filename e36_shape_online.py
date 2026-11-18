"""E36: the fixed-shape certificate h_t Psi of Theorem "Certificates of prescribed shape", run online.

For each configuration:
  1. offline (no data): the working filter is run to t = T, S and Y are frozen, the rank-p map of Corollary
     "Scalar plus rank p" is iterated to its limit and scaled by 1 + 1e-3 to a shape Psi_p = g Y^{-1} + Z Z^T, as in
     Algorithm 2 (same construction as e27_shape.py / e30_shape_cn.py); the super-solution test
     lmax(Psi^{-1} L(Psi)) and the rate r(Psi) are recorded;
  2. online: the filter is run from t = 0 with the shape held fixed, Psi_t = Psi_p for all t, and the scalar
     h_0 = lmax(Psi^{-1} P_0), h_t = lmax(Psi^{-1} L_t(h_{t-1} Psi)) (one generalised eigenvalue per step).
     Alongside, the exact error second moment M_t, the Kalman covariance and the adaptive rank-p certificate
     C_t^(p) (as used in all other experiments) are propagated.
Reported: largest exact worst-direction ratio lmax((h_t Psi)^{-1} M_t) (checked every 10th step and in the last 50;
<= 1 by the theorem), largest h_t, and the price relative to the Kalman filter, mean over the last 50 steps, of the
fixed-shape and of the adaptive rank-p certificate.
usage: python e36_shape_online.py [upwind|cn] [workers]  ->  results/e36_shape_online_<which>.csv (appended per run)
"""
import sys, os, time, numpy as np, pandas as pd, scipy.linalg as sla
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol
from e17_lowrank import gen_top, worst, SAFE
import e26_variants as e26

PS = (5, 20)

def lmax_pencil(X, Psi):
    n = X.shape[0]; return sla.eigh(X, Psi, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]

def S_of(F, Yinv, Q, mask, I):
    Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
    return S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I

def job(a):
    variant, n, sw, Pe, m, b, seed, T = a
    t0 = time.perf_counter()
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    F = e26.propagators("cn", mod, Pe)[0](0) if variant == "cn" else mod.F
    Q, J, H, sv = mod.Q, mod.HRH, mod.H, mod.sv
    I = np.eye(n); mask = mod.dist <= b; p0 = 0.01
    # 1. offline shape (no data)
    Yinv = p0 * I
    for t in range(T):
        S = S_of(F, Yinv, Q, mask, I); Y = S + J; Yinv = inv_from_chol(chol(Y))
    Psi, info = {}, {}
    AF = Yinv @ S @ F; Nh = sym(Yinv @ sym(S @ Q @ S + J) @ Yinv)
    for p in PS:
        C = Yinv.copy(); g_old = np.inf
        for it in range(6000):
            X = sym(S @ sym(F @ C @ F.T + Q) @ S + J); lam, V = gen_top(X, Y, p + 1); g = lam[p]
            C = sym(g * Yinv + (V[:, :p] * (lam[:p] - g)) @ V[:, :p].T)
            if (not np.isfinite(g)) or g > 1e8 or abs(g - g_old) <= 1e-11 * g: break
            g_old = g
        P_ = (1 + 1e-3) * C; Psi[p] = P_
        info[f"test_{p}"] = lmax_pencil(sym(AF @ P_ @ AF.T) + Nh, P_)     # < 1: strict super-solution
        info[f"r_{p}"] = lmax_pencil(sym(AF @ P_ @ AF.T), P_)
    # 2. online
    Yinv = p0 * I; M = p0 * I; Pk = p0 * I
    h = {p: p0 * lmax_pencil(I, Psi[p]) for p in PS}
    Cad = {p: p0 * I for p in PS}
    wmax = {f"{k}_{p}": 0.0 for k in ("worst_fix", "worst_ad") for p in PS}
    hmax = {p: h[p] for p in PS}
    acc = {f"{k}_{p}": [] for k in ("price_fix", "price_ad") for p in PS}
    for t in range(T):
        S = S_of(F, Yinv, Q, mask, I); Y = S + J; Yn = inv_from_chol(chol(Y)); A = Yn @ S; K = Yn @ H.T / sv ** 2
        M = sym(A @ sym(F @ M @ F.T + Q) @ A.T + sv ** 2 * (K @ K.T))
        Pk = sym(F @ Pk @ F.T + Q); Pk = inv_from_chol(chol(inv_from_chol(chol(Pk)) + J))
        trk = np.trace(Pk)
        for p in PS:
            Lh = sym(Yn @ sym(S @ sym(F @ (h[p] * Psi[p]) @ F.T + Q) @ S + J) @ Yn)       # L_t(h_{t-1} Psi)
            h[p] = SAFE * lmax_pencil(Lh, Psi[p]); hmax[p] = max(hmax[p], h[p])
            X = sym(S @ sym(F @ Cad[p] @ F.T + Q) @ S + J); lam, V = gen_top(X, Y, p + 1); g = SAFE * lam[p]
            Cad[p] = sym(g * Yn + (V[:, :p] * (SAFE * lam[:p] - g)) @ V[:, :p].T)
            if (t % 10 == 9) or (t >= T - 50):
                wmax[f"worst_fix_{p}"] = max(wmax[f"worst_fix_{p}"], worst(M, h[p] * Psi[p]))
                wmax[f"worst_ad_{p}"] = max(wmax[f"worst_ad_{p}"], worst(M, Cad[p]))
            if t >= T - 50:
                acc[f"price_fix_{p}"].append(np.sqrt(h[p] * np.trace(Psi[p]) / trk) - 1)
                acc[f"price_ad_{p}"].append(np.sqrt(np.trace(Cad[p]) / trk) - 1)
        Yinv = Yn
    out = dict(variant=variant, n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, T=T, **info, **wmax,
               **{f"hmax_{p}": hmax[p] for p in PS}, **{f"hT_{p}": h[p] for p in PS},
               **{k: float(np.mean(v)) for k, v in acc.items()}, sec=time.perf_counter() - t0)
    return out

CONF = {
    "upwind": [("upwind", 240, 0.05, Pe, m, b, 0, 600) for Pe in (10.0, 50.0, 90.0) for m in (8, 32) for b in (1, 2, 3, 4)]
              + [("upwind", 600, 0.05, 10.0, 8, b, 0, 600) for b in (1, 2, 3)],
    "cn": None,
}
if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "upwind"
    if which == "cn":
        st = pd.read_csv("results/e26_cn.csv"); st = st[st.failed == False]
        jobs = [("cn", 240, 0.05, r.Pe, int(r.m), int(r.b), 0, 600) for r in st.itertuples()]
    else:
        jobs = CONF[which]
    out_f = f"results/e36_shape_online_{which}.csv"
    done = set()
    if os.path.exists(out_f):
        d0 = pd.read_csv(out_f); done = {(r.n, r.Pe, r.m, r.b) for r in d0.itertuples()}
    jobs = [j for j in jobs if (j[1], j[3], j[4], j[5]) not in done]           # resumable
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as pool:
        for r in pool.imap_unordered(job, jobs):
            pd.DataFrame([r]).to_csv(out_f, mode="a", header=not os.path.exists(out_f), index=False)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
