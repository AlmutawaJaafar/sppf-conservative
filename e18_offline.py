"""E18: offline (steady-state) certificate. Run the working filter to T=600 (no data needed), take the
converged S, P (prior) and iterate the scalar map f(g) = max{1, lmax((g(P-Q)+Q) S)} from g=1 to its
smallest fixed point (or report divergence). Compare with the online minimal certificate at T and report
the relative change of S and P over the last 50 steps (how converged the working filter is)."""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol

SAFE = 1 + 1e-10

def job(a):
    n, sw, Pe, m, b, seed, T = a
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    F, Q, J = mod.F, mod.Q, mod.HRH; I = np.eye(n); mask = mod.dist <= b
    Yinv = 0.01 * I; g = 1.0; hist = []
    for t in range(T):
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I
        LS = chol(S); g = max(1.0, SAFE * np.linalg.eigvalsh(sym(LS.T @ (g * (Pbar - Q) + Q) @ LS))[-1])
        if t == T - 51: S50, P50 = S.copy(), Pbar.copy()
        Yinv = inv_from_chol(chol(S + J)); hist.append(g)
    # offline: scalar map with the converged S, Pbar; G0 = LS'(Pbar-Q)LS, G1 = LS' Q LS
    A0 = sym(LS.T @ (Pbar - Q) @ LS); A1 = sym(LS.T @ Q @ LS)
    f = lambda x: max(1.0, SAFE * np.linalg.eigvalsh(x * A0 + A1)[-1])
    x = 1.0; it = 0
    while it < 100000:
        y = f(x); it += 1
        if abs(y - x) <= 1e-12 * y or y > 1e12: x = y; break
        x = y
    dS = np.linalg.norm(S - S50, 2) / np.linalg.norm(S, 2); dP = np.linalg.norm(Pbar - P50, 2) / np.linalg.norm(Pbar, 2)
    slope = np.linalg.eigvalsh(A0)[-1]        # f'(g) <= lmax(A0); a fixed point exists iff growth slope allows it
    return dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, gamma_online_T=hist[-1],
                gamma_online_mean_last50=np.mean(hist[-50:]), gamma_offline=x, iterations=it,
                lmax_A0=slope, dS_rel_last50=dS, dP_rel_last50=dP)

if __name__ == "__main__":
    jobs = [(240, sw, Pe, m, b, 0, 600) for sw in (0.05, 0.2) for Pe in (10.0, 50.0, 90.0) for m in (8, 32) for b in (1, 2, 3, 4)]
    with Pool(2) as p: rows = p.map(job, jobs)
    d = pd.DataFrame(rows); d.to_csv("results/e18_offline.csv", index=False)
    d["rel"] = d.gamma_offline / d.gamma_online_T - 1
    pd.set_option("display.width", 200); print(d.round(5).to_string())
