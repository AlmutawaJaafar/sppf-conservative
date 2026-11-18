"""E17: scalar-plus-low-rank certified covariance (Lemma 1, Corollary 3) on the configurations of Table 2.

Working filter as in Algorithm 1 (unchanged estimate). Certified posterior of rank r:
    X_t = S_t (F C_{t-1} F' + Q) S_t + J,   (lam_i, v_i) generalized eigenpairs of (X_t, Y_t), v'Yv = 1,
    C_t = g Y_t^{-1} + sum_{i<=r} (lam_i - g) v_i v_i',   g = lam_{r+1}.
r = 0 is the sharp certificate of Theorem 1; the basic (formerly "minimal") certificate is also tracked.
True error moments are computed exactly.  Reported per run (mean over the last 50 of T steps):
price = sqrt(tr C / tr P_KF) - 1, scalar part g, and the worst-direction ratio lmax(C^{-1} M),
whose maximum over checked steps (every 10th step and the last 50) must be <= 1.

usage: python e17_lowrank.py [sweep|hard]  ->  results/e17_lowrank_<which>.csv
"""
import sys, time, numpy as np, pandas as pd, scipy.linalg as sla
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol

RANKS = (0, 5, 20)
SAFE = 1 + 1e-10


def gen_top(X, Y, k):
    """Top-k generalized eigenpairs (descending), V'YV = I. Falls back to the full solver."""
    n = X.shape[0]
    try:
        lam, V = sla.eigh(X, Y, subset_by_index=[n - k, n - 1])
    except (np.linalg.LinAlgError, ValueError):
        lam, V = sla.eigh(X, Y); lam, V = lam[n - k:], V[:, n - k:]
    return lam[::-1], V[:, ::-1]


def worst(M, C):
    L = chol(C); Li = sla.solve_triangular(L, np.eye(len(C)), lower=True)
    return np.linalg.eigvalsh(sym(Li @ M @ Li.T))[-1]


def job(a):
    n, sw, Pe, m, b, seed, T = a
    t0 = time.perf_counter()
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    F, Q, J, H, sv = mod.F, mod.Q, mod.HRH, mod.H, mod.sv
    I = np.eye(n); mask = mod.dist <= b; p0 = 0.01
    Yinv = p0 * I; M = p0 * I; Pk = p0 * I; gmin = 1.0
    C = {r: p0 * I for r in RANKS}
    acc = {k: [] for k in ["price_min", "g_min"] + [f"{q}_{r}" for r in RANKS for q in ("price", "g")]}
    wmax = {r: 0.0 for r in RANKS}; wmax["min"] = 0.0
    for t in range(T):
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        lmB = np.linalg.eigvalsh(S)[0]; S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - lmB) * I
        Y = S + J; Yn = inv_from_chol(chol(Y)); A = Yn @ S; K = Yn @ H.T / sv ** 2
        M = sym(A @ sym(F @ M @ F.T + Q) @ A.T + sv ** 2 * (K @ K.T))
        Pk = sym(F @ Pk @ F.T + Q); Pk = inv_from_chol(chol(inv_from_chol(chol(Pk)) + J))
        LS = chol(S)
        gmin = max(1.0, SAFE * np.linalg.eigvalsh(sym(LS.T @ (gmin * (Pbar - Q) + Q) @ LS))[-1])
        gr = {}
        for r in RANKS:
            X = sym(S @ sym(F @ C[r] @ F.T + Q) @ S + J)
            lam, V = gen_top(X, Y, r + 1); g = SAFE * lam[r]; gr[r] = g
            C[r] = sym(g * Yn + (V[:, :r] * (SAFE * lam[:r] - g)) @ V[:, :r].T) if r else g * Yn
        trk = np.trace(Pk); check = (t % 10 == 9) or (t >= T - 50)
        if t >= T - 50:
            acc["price_min"].append(np.sqrt(gmin * np.trace(Yn) / trk) - 1); acc["g_min"].append(gmin)
            for r in RANKS:
                acc[f"price_{r}"].append(np.sqrt(np.trace(C[r]) / trk) - 1)
                acc[f"g_{r}"].append(gr[r])
        if check:
            wmax["min"] = max(wmax["min"], worst(M, gmin * Yn))
            for r in RANKS: wmax[r] = max(wmax[r], worst(M, C[r]))
        Yinv = Yn
    out = dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, sec=time.perf_counter() - t0,
               price_min=np.mean(acc["price_min"]), g_min=np.mean(acc["g_min"]), worst_min=wmax["min"])
    for r in RANKS:
        out[f"price_{r}"] = np.mean(acc[f"price_{r}"]); out[f"worst_{r}"] = wmax[r]; out[f"g_{r}"] = np.mean(acc[f"g_{r}"])
    return out


def configs(which):
    rows = []
    if which == "sweep":       # all n=240 runs, and n=600 with one sensor layout (seed 0)
        for sw in (0.05, 0.2):
            for Pe in (10.0, 50.0, 90.0):
                for m in (8, 32):
                    for b in (1, 2, 3, 4):
                        for seed in (0, 1, 2): rows.append((240, sw, Pe, m, b, seed, 600))
        for Pe in (10.0, 50.0, 90.0):
            for m in (8, 32):
                for b in (1, 2, 3, 4): rows.append((600, 0.05, Pe, m, b, 0, 600))
    elif which == "lownoise":   # Section 5.6 regime: weak process noise, T = 150 as in Supplement Fig. S3
        for b in (1, 2, 3, 4): rows.append((240, 0.01, 10.0, 8, b, 0, 150))
    elif which in ("s240", "s600", "h600", "h1200"):
        allr = configs("sweep") + configs("hard")
        return [r for r in allr if (which[0] == "s") == (r in configs("sweep")) and r[0] == int(which[1:])]
    elif which == "t1200":     # settled n=1200 runs at b = 3, 4 (one layout), for the sharp/rank-20 prices (printed by gen_tab3.py; not shown in the paper)
        for m in (8, 32):
            for b in (3, 4): rows.append((1200, 0.05, 50.0, m, b, 0, 600))
    elif which == "t1200b":    # the n=1200, b=2, m=32 cell of Table 2 (one layout)
        rows.append((1200, 0.05, 50.0, 32, 2, 0, 600))
    elif which == "hard":      # the remaining unsettled runs of Table 2 (n=600 seeds 1, 2; n=1200 b <= 2)
        for (b, seed) in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]: rows.append((600, 0.05, 10.0, 8, b, seed, 600))
        for (m, b, seed) in [(8, 2, 0), (8, 2, 1), (8, 2, 2), (8, 1, 0), (32, 1, 0)]:
            rows.append((1200, 0.05, 50.0, m, b, seed, 600))
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    jobs = configs(which)
    jobs.sort(key=lambda a: -a[0])
    tag = which
    if len(sys.argv) > 2:                       # optional slice "i:j" of the job list
        i, j = (int(x) for x in sys.argv[2].split(":")); jobs = jobs[i:j]; tag = f"{which}_{i}_{j}"
    with Pool(2) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(job, jobs)):
            rows.append(r); print(i + 1, len(jobs), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(f"results/e17_lowrank_{tag}.csv", index=False)
