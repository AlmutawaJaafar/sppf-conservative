"""E19: rank diagnostic for the low-rank certificates (Proposition 2).
nu#_(i) = i-th largest eigenvalue of the pencil (S_t F Y_{t-1}^{-1} F' S_t, Y_t). The scalar part of the
rank-p certificate satisfies g_t >= nu#_(p+1) g_{t-1}, so it must grow if nu#_(p+1) > 1 persistently.
The rates depend only on the working filter. For each run of E17 they are computed at steps
560, 570, ..., 600; reported are the geometric means of nu#_(i) for i = 1..25 and the number k of
indices whose geometric mean exceeds one."""
import sys, numpy as np, pandas as pd, scipy.linalg as sla
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol
import e17_lowrank as e17

K = 25

def job(a):
    n, sw, Pe, m, b, seed, T = a
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    F, Q, J = mod.F, mod.Q, mod.HRH; I = np.eye(n); mask = mod.dist <= b
    Yinv = 0.01 * I; logs = []
    samples = set(range(T - 41, T, 10))
    for t in range(T):
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I
        Y = S + J
        if t in samples:
            Bm = sym(S @ F @ Yinv @ F.T @ S)
            lam = e17.gen_top(Bm, Y, K)[0]
            logs.append(np.log(np.maximum(lam, 1e-300)))
        Yinv = inv_from_chol(chol(Y))
    gm = np.exp(np.mean(logs, axis=0))
    out = dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, k_above=int((gm > 1).sum()))
    for i in range(K): out[f"nus_{i+1}"] = gm[i]
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "s240"
    jobs = e17.configs(which)
    with Pool(2) as p: rows = p.map(job, jobs)
    pd.DataFrame(rows).to_csv(f"results/e19_rank_rates_{which}.csv", index=False)
    print(pd.DataFrame(rows)[["n", "Pe", "m", "b", "seed", "k_above", "nus_1", "nus_2", "nus_6", "nus_21"]].round(4).to_string())
