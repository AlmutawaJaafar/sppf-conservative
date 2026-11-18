"""E14: per-step growth rates of the certificate along the working filter (Theorem growth):
nu_t = lmax(F Y_{t-1}^{-1} F' S_t) (lower rate) and u_t (1 - eta_t) (upper rate), for every T=600 run
of the main table. These depend only on the working filter, not on the certificate."""
import sys, numpy as np, pandas as pd, scipy.linalg as sla
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol, lmax, lmin

def job(a):
    n_, sw, Pe, m, b, seed = a
    Nx, Ny = GRIDS[n_]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    n, F, Q = mod.n, mod.F, mod.Q; mask = mod.dist <= b; Yinv = 0.01 * np.eye(n)
    nu, up, nus = [], [], []
    for t in range(600):
        Pbar = sym(F @ Yinv @ F.T + Q); Sp = inv_from_chol(chol(Pbar)); B = Sp * mask
        lmB = lmin(B); s = max(0.0, 1e-8 * np.mean(np.diag(B)) - lmB); S = B + s * np.eye(n)
        if (t >= 500 and t % 5 == 4) or t % 100 == 99:
            LS = chol(S)
            nu.append(lmax(LS.T @ (Pbar - Q) @ LS))
            u = lmax(LS.T @ Pbar @ LS); eta = sw ** 2 / lmax(Pbar)
            up.append(u * (1 - eta))
            Y = S + mod.HRH; X = sym(S @ (Pbar - Q) @ S)       # sharp rate: lmax of pencil (S (P-Q) S, Y)
            nus.append(sla.eigh(X, Y, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0])
        Yinv = inv_from_chol(chol(S + mod.HRH))
    nu, up, nus = np.array(nu), np.array(up), np.array(nus); last = slice(-20, None); l50 = slice(-10, None)   # samples every 5 steps over 501-600 / 551-600
    g = lambda x: float(np.exp(np.mean(np.log(x))))
    return dict(n=n_, sw=sw, Pe=Pe, m=m, b=b, seed=seed,
                nu_gm_501_600=g(nu[last]), nu_gm_551_600=g(nu[l50]), nu_max_501_600=float(nu[last].max()),
                nu_min_501_600=float(nu[last].min()), up_gm_551_600=g(up[l50]), up_am_551_600=float(up[l50].mean()),
                up_max_501_600=float(up[last].max()), nus_gm_501_600=g(nus[last]), nus_max_501_600=float(nus[last].max()))

if __name__ == "__main__":
    from gen_tab3 import load_t600
    d = load_t600()
    jobs = [tuple(r) for r in d[["n", "sw", "Pe", "m", "b", "seed"]].itertuples(index=False)]
    jobs.sort(key=lambda a: -a[0])
    with Pool(2) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(job, jobs)):
            rows.append(r); print(f"{i+1}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv("results/e14_growth_rates.csv", index=False); print("wrote", len(rows))
