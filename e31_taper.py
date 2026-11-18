"""E31: a Schur-product taper of the prior precision as the working precision (baseline, Section 5.7).

The upwind model of Section 5.1 (n = 240, sigma_w = 0.05, Pe in {10, 50, 90}, m in {8, 32}, b = 1..4, layout 0, T = 600)
is run with S_t = P_{t|t-1}^{-1} o GC(d / (K (b + 1/2) / 2)), where GC is the Gaspari--Cohn function of the Euclidean
grid distance d (positive definite in 2D, so S_t is positive definite by the Schur product theorem); the support
radius is K (b + 1/2), K in {1, 2, 3, 4}. The true error second moment is propagated exactly, and the minimal, sharp,
rank-5 and rank-20 certificates are computed as in E17. Reported per run: mean number of nonzeros per row of the
taper, prices (mean over the last 50 steps), worst-direction ratios of the certified and of the uncorrected
covariance, and the offline prediction of the sharp price. The loop is shared with e26_variants.py.
usage: python e31_taper.py taper[K]  ->  results/e31_taper[K].csv   (taper = K 1; taper2, taper3, taper4)
"""
import sys, time, numpy as np, pandas as pd, scipy.linalg as sla, scipy.sparse as sp
from multiprocessing import Pool
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol, lap1d, upwind1d
from e17_lowrank import gen_top, worst, RANKS, SAFE

PER = 50

def central1d(N):          # (u_{i+1} - u_{i-1}) / 2 with homogeneous Dirichlet ends
    return sp.diags([-0.5 * np.ones(N - 1), 0.5 * np.ones(N - 1)], [-1, 1])

def propagators(variant, mod, Pe, Fo=0.01):
    Nx, Ny, n = mod.Nx, mod.Ny, mod.n; Ix, Iy = sp.identity(Nx), sp.identity(Ny)
    Lap = (sp.kron(Iy, lap1d(Nx)) + sp.kron(lap1d(Ny), Ix)).toarray()
    if variant.startswith("taper"):
        return (lambda t: mod.F), 1
    if variant == "cn":
        L = Fo * Lap - Fo * Pe * sp.kron(Iy, central1d(Nx)).toarray(); I = np.eye(n)
        F = np.linalg.solve(I - L / 2, I + L / 2); return lambda t: F, 1
    Dx = sp.kron(Iy, upwind1d(Nx)).toarray(); I = np.eye(n)
    Fs = [I + Fo * Lap - Fo * Pe * (1 + 0.5 * np.sin(2 * np.pi * k / PER)) * Dx for k in range(PER)]
    return lambda t: Fs[t % PER], PER

def gc(r):
    r = np.abs(r); out = np.zeros_like(r); a = r <= 1; c = (r > 1) & (r < 2)
    x = r[a]; out[a] = -0.25 * x**5 + 0.5 * x**4 + 0.625 * x**3 - 5 / 3 * x**2 + 1
    x = r[c]; out[c] = x**5 / 12 - 0.5 * x**4 + 0.625 * x**3 + 5 / 3 * x**2 - 5 * x + 4 - 2 / (3 * x)
    return out

def lmax(X, Y):
    n = X.shape[0]; return sla.eigh(X, Y, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]

def job(a):
    try:
        return job_(a)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as err:
        variant, n, sw, Pe, m, b, seed, T = a
        return dict(variant=variant, n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, failed=True, reason=str(err)[:60])

def job_(a):
    variant, n, sw, Pe, m, b, seed, T = a
    t0 = time.perf_counter()
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    Ft, P = propagators(variant, mod, Pe)
    Q, J, H, sv = mod.Q, mod.HRH, mod.H, mod.sv
    I = np.eye(n); p0 = 0.01
    if variant.startswith("taper"):              # Schur-product taper with the Gaspari--Cohn function of Euclidean distance
        K = float(variant[5:] or 1)              # support radius K (b + 1/2)
        Nx_, Ny_ = GRIDS[n]; xy = np.array([(i % Nx_, i // Nx_) for i in range(n)], float)
        de = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)); mask = gc(de / (K * (b + 0.5) / 2))
    else:
        mask = mod.dist <= b
    Yinv = p0 * I; M = p0 * I; Pk = p0 * I; gmin = 1.0
    C = {r: p0 * I for r in RANKS}
    acc = {k: [] for k in ["price_min"] + [f"price_{r}" for r in RANKS]}
    wmax = {r: 0.0 for r in RANKS}; wmax["min"] = 0.0; wmax["unc"] = 0.0
    last = []                                                    # (A_t, N_t, Y_t, tr Y_t^{-1}, tr Pk_t) over the last period
    for t in range(T):
        F = Ft(t)
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        lmB = np.linalg.eigvalsh(S)[0]; S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - lmB) * I
        Y = S + J; Yn = inv_from_chol(chol(Y)); A = Yn @ S; K = Yn @ H.T / sv ** 2
        M = sym(A @ sym(F @ M @ F.T + Q) @ A.T + sv ** 2 * (K @ K.T))
        Pk = sym(F @ Pk @ F.T + Q); Pk = inv_from_chol(chol(inv_from_chol(chol(Pk)) + J))
        LS = chol(S)
        gmin = max(1.0, SAFE * np.linalg.eigvalsh(sym(LS.T @ (gmin * (Pbar - Q) + Q) @ LS))[-1])
        for r in RANKS:
            X = sym(S @ sym(F @ C[r] @ F.T + Q) @ S + J)
            lam, V = gen_top(X, Y, r + 1); g = SAFE * lam[r]
            C[r] = sym(g * Yn + (V[:, :r] * (SAFE * lam[:r] - g)) @ V[:, :r].T) if r else g * Yn
        trk = np.trace(Pk)
        if not np.isfinite(gmin) or gmin > 1e50:          # sparse filter unstable: stop, as in Section 5.4
            return dict(variant=variant, n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, failed=True, reason=f"gamma>1e50 at t={t}",
                        worst_unc=wmax["unc"], worst_min=wmax["min"])
        if t >= T - P:
            last.append((sym(S @ F @ Yinv @ F.T @ S), sym(S @ Q @ S + J), Y, np.trace(Yn), trk))
        if t >= T - 50:
            acc["price_min"].append(np.sqrt(gmin * np.trace(Yn) / trk) - 1)
            for r in RANKS: acc[f"price_{r}"].append(np.sqrt(np.trace(C[r]) / trk) - 1)
        if (t % 10 == 9) or (t >= T - 50):
            wmax["min"] = max(wmax["min"], worst(M, gmin * Yn)); wmax["unc"] = max(wmax["unc"], worst(M, Yn))
            for r in RANKS: wmax[r] = max(wmax[r], worst(M, C[r]))
        Yinv = Yn
    out = dict(variant=variant, n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, failed=False, nnz_row=float((mask != 0).sum(1).mean()), price_min=np.mean(acc["price_min"]),
               worst_min=wmax["min"], worst_unc=wmax["unc"])
    for r in RANKS: out[f"price_{r}"] = np.mean(acc[f"price_{r}"]); out[f"worst_{r}"] = wmax[r]
    # offline prediction: smallest periodic fixed point of the composed sharp map
    def Phi(g):
        for (A_, N_, Y_, _, _) in last: g = lmax(g * A_ + N_, Y_)
        return g
    hi = 1.0
    while Phi(hi) > hi and hi < 1e12: hi *= 2
    if Phi(hi) > hi:
        out["price_sharp_pred"] = np.inf
    else:
        lo = 0.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if Phi(mid) > mid: lo = mid
            else: hi = mid
        g, pr = hi, []
        for (A_, N_, Y_, trY, trK) in last:
            g = lmax(g * A_ + N_, Y_); pr.append(np.sqrt(g * trY / trK) - 1)
        out["price_sharp_pred"] = np.mean(pr[-50:]) if P > 1 else pr[-1]
    out["sec"] = time.perf_counter() - t0
    return out

if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "cn"
    pes = (10.0, 50.0) if variant == "tv" else (10.0, 50.0, 90.0)   # tv: at Pe = 90 the peak Courant number 1.35 makes F_t unstable
    jobs = [(variant, 240, 0.05, Pe, m, b, 0, 600) for Pe in pes for m in (8, 32) for b in (1, 2, 3, 4)]
    rows = []
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs)):
            rows.append(r); print(i + 1, len(jobs), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            pd.DataFrame(rows).to_csv(f"results/e31_{variant}.csv", index=False)
