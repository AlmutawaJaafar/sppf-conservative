"""Repeat the n=99,904 timings of e5 (factorisations, mean solve, eps=0.002 Lanczos) to measure variability."""
import time, math, csv, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from sppf_certified import grid_for, lap1d, upwind1d, banded_S, SpChol, lanczos_max
b = 2; Nx, Ny = grid_for(100000); n = Nx * Ny; m = max(8, n // 100)
rows = []
for rep in range(3):
    rng = np.random.default_rng(rep)
    F = (sp.identity(n) + 0.01 * (sp.kron(sp.identity(Ny), lap1d(Nx)) + sp.kron(lap1d(Ny), sp.identity(Nx)))
         - 0.5 * sp.kron(sp.identity(Ny), upwind1d(Nx))).tocsr()
    S = banded_S(Nx, Ny, b); sens = rng.choice(n, m, replace=False); hrh = np.zeros(n); hrh[sens] = 4.0
    Y = (S + sp.diags(hrh)).tocsc()
    t0 = time.perf_counter(); chS = SpChol(S); tS = time.perf_counter() - t0
    t0 = time.perf_counter(); luY = splu(Y, permc_spec="MMD_AT_PLUS_A"); tY = time.perf_counter() - t0
    t0 = time.perf_counter(); luY.solve(rng.standard_normal(n)); tm = time.perf_counter() - t0
    Gmv = lambda v: chS.LSt(1.05 * (F @ luY.solve(F.T @ chS.LS(v))) + 0.0025 * chS.LS(v))
    k = math.ceil((math.log(1.648 * math.sqrt(n) * 1000 / 1e-6) / math.sqrt(0.002) + 1) / 2)
    t0 = time.perf_counter(); lanczos_max(Gmv, n, k, rng); tl = time.perf_counter() - t0
    r = dict(rep=rep, n=n, t_factor_S=tS, t_factor_Y=tY, t_mean_solve=tm, k=k, t_lanczos_eps0002=tl,
             t_update=tY + tm); rows.append(r); print(r, flush=True)
with open("results/e5_timing_repeat.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
