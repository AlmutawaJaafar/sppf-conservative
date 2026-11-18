"""E21: localized EnKF with adaptive inflation against the certified sparse filter, by Monte Carlo.

Model of Section 5.1 (n = 240, Pe = 50, sigma_w = 0.05, m = 8, one sensor layout), NTRAJ trajectories with
common truth and observations for all filters. EnKF: stochastic (perturbed observations), ensemble size Ne,
Gaspari-Cohn localization of the forecast covariance with half-width L (Euclidean grid distance), adaptive
local multiplicative inflation by relaxation to prior spread (RTPS, Whitaker and Hamill 2012) with weight alpha;
reported covariance = localized analysis ensemble covariance. (L, alpha) are tuned per Ne by time-averaged
RMSE over the last 150 steps (standard practice). T = 300 (spin-up about 100 steps). Certified filter: Algorithm 1 with b = 3 and the minimal certificate;
uncorrected sparse filter for reference. Metrics (last 50 steps, and minimum over steps 101-300):
  coverage of nominal 95% intervals for the domain mean, the node farthest from the sensors, and a cross-validated
  worst direction in whitened coordinates (leading direction of the whitened errors of half the trajectories, coverage
  on the other half); NEES/n; RMSE and reported-sd ratio relative to the exact Kalman filter.
usage: python e21_enkf.py  ->  results/e21_enkf.csv (tuning grid, layout 0)
       python e21_enkf.py layouts [b]  ->  results/e21_enkf_layouts.csv (tuned configurations, layouts 0-2,
       with raw-ensemble variances for the domain mean and the far node in addition to the localized ones)
"""
import sys, time, numpy as np, pandas as pd, scipy.linalg as sla
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol

NTRAJ, T = 300, 300

def gc(r):            # Gaspari-Cohn, r = d / L (support 2L)
    r = np.abs(r); out = np.zeros_like(r, dtype=float)
    a = r <= 1; b = (r > 1) & (r < 2)
    ra = r[a]; out[a] = -0.25 * ra**5 + 0.5 * ra**4 + 0.625 * ra**3 - 5 / 3 * ra**2 + 1
    rb = r[b]; out[b] = rb**5 / 12 - 0.5 * rb**4 + 0.625 * rb**3 + 5 / 3 * rb**2 - 5 * rb + 4 - 2 / (3 * rb)
    return out

def metrics(E, Lw, var_mean, var_far, far, n, half):
    """E: (K, n) errors; Lw: (K, n, n) or (n, n) lower Cholesky factors of the reported covariances."""
    if Lw.ndim == 2:
        U = np.linalg.solve(Lw, E.T).T
    else:
        U = np.stack([np.linalg.solve(Lw[k], E[k]) for k in range(E.shape[0])])
    A, B = U[:half], U[half:]
    ev, V = np.linalg.eigh(A.T @ A); w = V[:, -1]; zB = B @ w
    em = E.mean(axis=1)
    return dict(nees=np.mean(np.sum(U**2, 1)) / n, cov_cv=np.mean(np.abs(zB) <= 1.96),
                cov_mean=np.mean(np.abs(em) <= 1.96 * np.sqrt(var_mean)), cov_far=np.mean(np.abs(E[:, far]) <= 1.96 * np.sqrt(var_far)))

def run(Ne_L_list, seed=0, b=3):
    Nx, Ny = GRIDS[240]; mod = Model(Nx, Ny, Pe=50.0, sw=0.05, m=8, seed=seed)
    n, F, Q, H, sv, m = mod.n, mod.F, mod.Q, mod.H, mod.sv, mod.m
    R = sv**2 * np.eye(m); J = mod.HRH; p0 = 0.01; I = np.eye(n)
    rng = np.random.default_rng(100 + seed)
    sens_d = np.min(mod.dist[:, mod.sens], axis=1); far = int(np.argmax(sens_d)); half = NTRAJ // 2
    X = np.sqrt(p0) * rng.standard_normal((NTRAJ, n))
    # deterministic filters: exact KF, sparse filter (b), certificate
    Pk = p0 * I; Yinv = p0 * I; gam = 1.0; mask = mod.dist <= b
    Cr = p0 * I; RANK = 20                                  # scalar-plus-rank-20 certified covariance (Corollary 3)
    Xk = np.zeros((NTRAJ, n)); Xs = np.zeros((NTRAJ, n))
    ens = {}
    for (Ne, L, alpha) in Ne_L_list:
        dE = np.sqrt(((mod.xy[:, None, :] - mod.xy[None, :, :]) ** 2).sum(-1))       # Euclidean (GC is PD in Euclidean distance)
        rho = gc(dE / L)
        ens[(Ne, L, alpha)] = dict(Ne=Ne, alpha=alpha, rho=rho, A=np.sqrt(p0) * rng.standard_normal((NTRAJ, n, Ne)), lam=np.ones(NTRAJ), rec=[], t=0.0)
    rec_s = []
    for t in range(1, T + 1):
        X = X @ F.T + mod.sw * rng.standard_normal((NTRAJ, n)); obs = X[:, mod.sens] + sv * rng.standard_normal((NTRAJ, m))
        # exact KF
        Pp = sym(F @ Pk @ F.T + Q); Kk = Pp @ H.T @ np.linalg.inv(H @ Pp @ H.T + R); Pk = sym((I - Kk @ H) @ Pp)
        Xk = Xk @ F.T; Xk = Xk + (obs - Xk[:, mod.sens]) @ Kk.T
        trk = np.trace(Pk)
        # sparse filter + minimal certificate
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I
        LS = chol(S); gam = max(1.0, (1 + 1e-10) * np.linalg.eigvalsh(sym(LS.T @ (gam * (Pbar - Q) + Q) @ LS))[-1])
        Y = S + J; Yinv_prev = Yinv; Yinv = inv_from_chol(chol(Y)); Ks = Yinv @ H.T / sv**2
        Xr = sym(S @ sym(F @ Cr @ F.T + Q) @ S + J); lam, V = sla.eigh(Xr, Y); lam, V = lam[::-1], V[:, ::-1]
        g_r = (1 + 1e-10) * lam[RANK]; Cr = sym(g_r * Yinv + (V[:, :RANK] * ((1 + 1e-10) * lam[:RANK] - g_r)) @ V[:, :RANK].T)
        Xs = Xs @ F.T; Xs = Xs + (obs - Xs[:, mod.sens]) @ Ks.T
        if t > 100 and t % 10 == 0:
            Es = Xs - X; Ek = Xk - X
            a = np.ones(n) / n
            for nm, C in (("unc", Yinv), ("cert", gam * Yinv), ("cert20", Cr)):
                mt = metrics(Es, chol(C), a @ C @ a, C[far, far], far, n, half)
                mt.update(t=t, rmse=np.sqrt(np.mean(Es**2) / (trk / n)), sd=np.sqrt(np.trace(C) / trk), kf_rmse=np.sqrt(np.mean(Ek**2) / (trk / n)))
                rec_s.append(dict(filter=nm, **mt))
        # EnKFs
        for key, e in ens.items():
            t0 = time.perf_counter(); Ne, rho, A = e["Ne"], e["rho"], e["A"]
            A = np.einsum("ij,kjl->kil", F, A) + mod.sw * rng.standard_normal((NTRAJ, n, Ne))
            mu = A.mean(axis=2, keepdims=True); D = A - mu
            # adaptive inflation (innovation statistics, before the update)
            Ds = D[:, mod.sens, :]                                                         # (K, m, Ne)
            PHt0 = np.einsum("kil,kjl->kij", D, Ds) / (Ne - 1)                            # Pf H'   (K, n, m)
            HPH = PHt0[:, mod.sens, :]
            sd_f = np.sqrt(np.sum(D**2, axis=2) / (Ne - 1))                                # forecast spread (K, n)
            PHt = rho[None][:, :, mod.sens] * PHt0
            S_in = PHt[:, mod.sens, :] + R[None]
            K = np.linalg.solve(S_in, np.transpose(PHt, (0, 2, 1))).transpose(0, 2, 1)       # (K, n, m)
            pert = obs[:, :, None] + sv * rng.standard_normal((NTRAJ, m, Ne))
            A = A + np.einsum("kim,kml->kil", K, pert - A[:, mod.sens, :])
            # relaxation to prior spread (RTPS): adaptive, local multiplicative inflation of the analysis anomalies
            mu_a = A.mean(axis=2, keepdims=True); Da = A - mu_a; sd_a = np.sqrt(np.sum(Da**2, axis=2) / (Ne - 1))
            fac = 1 + e["alpha"] * (sd_f - sd_a) / np.maximum(sd_a, 1e-12)
            A = mu_a + Da * fac[:, :, None]; e["lam"] = np.mean(fac, axis=1)
            e["A"] = A; e["t"] += time.perf_counter() - t0
            if t > 100 and t % 10 == 0:
                mu = A.mean(axis=2); Da = A - mu[:, :, None]
                Pa = rho[None] * np.einsum("kil,kjl->kij", Da, Da) / (Ne - 1) + 1e-10 * np.mean(np.diag(Q)) * I[None]
                Ee = mu - X; La = np.linalg.cholesky(Pa)
                a = np.ones(n) / n; vm = np.einsum("i,kij,j->k", a, Pa, a); vf = Pa[:, far, far]
                # raw (unlocalized) ensemble variances: rank Ne - 1 < n, so zero for functionals outside the span
                vm_raw = np.sum(np.einsum("i,kil->kl", a, Da) ** 2, axis=1) / (Ne - 1)
                vf_raw = np.sum(Da[:, far, :] ** 2, axis=1) / (Ne - 1)
                U = np.linalg.solve(La, Ee[:, :, None])[:, :, 0]
                Au, Bu = U[:half], U[half:]; ev, V = np.linalg.eigh(Au.T @ Au); zB = Bu @ V[:, -1]
                em = Ee.mean(axis=1)
                e["rec"].append(dict(t=t, nees=np.mean(np.sum(U**2, 1)) / n, cov_cv=np.mean(np.abs(zB) <= 1.96),
                                     cov_mean=np.mean(np.abs(em) <= 1.96 * np.sqrt(vm)), cov_far=np.mean(np.abs(Ee[:, far]) <= 1.96 * np.sqrt(vf)),
                                     cov_mean_raw=np.mean(np.abs(em) <= 1.96 * np.sqrt(vm_raw)), cov_far_raw=np.mean(np.abs(Ee[:, far]) <= 1.96 * np.sqrt(vf_raw)),
                                     rmse=np.sqrt(np.mean(Ee**2) / (trk / n)), sd=np.sqrt(np.mean(np.trace(Pa, axis1=1, axis2=2)) / trk),
                                     lam=float(np.mean(e["lam"]))))
        if t % 100 == 0: print("step", t, flush=True)
    rows = []
    def summ(rs, extra):
        d = pd.DataFrame(rs); last = d[d.t > T - 50]; late = d[d.t > T - 150]
        out = dict(extra)
        for c in [c_ for c_ in ["cov_cv", "cov_mean", "cov_far", "cov_mean_raw", "cov_far_raw"] if c_ in d]:
            out[c + "_end"] = last[c].mean(); out[c + "_min"] = d[c].min()
        out.update(nees_end=last.nees.mean(), rmse_late=late.rmse.mean(), sd_end=last.sd.mean())
        if "lam" in d: out["lam_end"] = last.lam.mean()
        return out
    ds = pd.DataFrame(rec_s)
    for nm in ("unc", "cert", "cert20"):
        rows.append(summ(ds[ds["filter"] == nm].to_dict("records"), dict(method=f"sparse b={b} ({nm})", Ne=np.nan, L=np.nan, sec_per_step=np.nan)))
    for (Ne, L, alpha), e in ens.items():
        rows.append(summ(e["rec"], dict(method="EnKF", Ne=Ne, L=L, alpha=alpha, sec_per_step=e["t"] / T / NTRAJ)))
    return pd.DataFrame(rows)

B_SPARSE = 3
def _layout(seed):
    d = run([(20, 3.0, 0.5), (50, 3.0, 0.5), (100, 3.0, 0.9)], seed=seed, b=B_SPARSE); d["layout"] = seed; return d

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    if len(sys.argv) > 1 and sys.argv[1] == "layouts":
        if len(sys.argv) > 2: B_SPARSE = int(sys.argv[2])
        # the RMSE-tuned configurations of the tuning run, on three sensor layouts with independent random streams
        from multiprocessing import Pool
        with Pool(2) as p: parts = p.map(_layout, [0, 1, 2])
        tag = "" if B_SPARSE == 3 else f"_b{B_SPARSE}"
        d = pd.concat(parts); d.to_csv(f"results/e21_enkf_layouts{tag}.csv", index=False); print(d.round(4).to_string())
    else:
        grid = [(Ne, L, al) for Ne in (20, 50, 100) for L in (3.0, 6.0) for al in (0.5, 0.9)]
        d = run(grid)
        d.to_csv("results/e21_enkf.csv", index=False)
        print(d.round(4).to_string())
