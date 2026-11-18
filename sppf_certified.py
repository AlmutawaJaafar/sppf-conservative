"""
sppf_certified.py
=================
Experiments for "Certified conservative covariance for sparse-precision Kalman
filtering" (J. Almutawa). Implements Algorithm 1 of the manuscript line by line
and evaluates it with the EXACT error second moments (no Monte Carlo error),
plus a Monte Carlo cross-check.

Model (Sec. 5.1): explicit finite-difference advection-diffusion on an Nx x Ny
grid with homogeneous Dirichlet boundaries,
    F = I + Fo*L2D - Fo*Pe*Dx   (5-point Laplacian, first-order upwind in x),
point sensors at random interior nodes (seeded), Q = sw^2 I, R = sv^2 I.
Consistent initialisation: e_0 ~ (0, p0 I), P_0 = p0 I, so M_{0|0} = P_0.

Algorithm 1, as implemented here (all dense, n <= 1200; see Sec. 3.3):
  Y_0 = P_0^{-1}, gamma_0 = 1
  for t = 1..T:
    Pbar_t = F Y_{t-1}^{-1} F' + Q                      (line 4)
    B_t    = B_b(Pbar_t^{-1})                           (line 5, truncation)
    s_t    = max(0, mu_t - lmin(B_t)),  mu_t = 1e-8 * mean(diag B_t)
    S_t    = B_t + s_t I   (Cholesky check)
    Y_t    = S_t + H' R^{-1} H                          (line 6)
    xhat_t = Y_t^{-1}(S_t xbar_t + H' R^{-1} y_t)       (line 7)
    gamma_t= max(1, (1+1e-10) lmax(L_S' Pc_{t|t-1} L_S)) (line 8, dense eigensolver)
             with Pc_{t|t-1} = gamma_{t-1}(Pbar_t - Q) + Q
    report Pc_{t|t} = gamma_t Y_t^{-1}                  (line 9)
Exact error second moments (Theorem 1 proof):
  M_{t|t} = A_t (F M F' + Q) A_t' + K_t R K_t',  A_t = Y_t^{-1} S_t,  K_t = Y_t^{-1} H' R^{-1}.

Modes (python3 sppf_certified.py MODE):
  e1     guarantee + tightness sweep (regimes x sensor layouts x b), n = 240 / 600 / 1200
  e3     value of the certificate: worst-functional and marginal coverage over time,
         steady start and diffuse start, with Monte Carlo cross-check
  e4     off-diagonal decay of the exact filter's prior precision vs b (proxy)
  e4w    the same along the sparse filter's own (working) trajectory -- the object of Assumption 2
  thm2   Theorem 2 bound gamma* versus the observed certificate (n = 240)
  e1x    Table 3 configurations with b <= 2 (and all b at n = 1200) continued to T = 600
  e7     closed-loop randomised Lanczos certificate (Proposition 1) vs dense certificate
  e8     conservative downward-shift construction vs truncation + certificate
  e5     cost study to n = 1e5 (sparse update + randomised-Lanczos certificate) and
         Lanczos accuracy check against the exact eigenvalue
  e6     low-process-noise regime: gamma_t as a run-time diagnostic
  e9     independent Monte Carlo (simulated states, data and estimates) in hard regimes
  e11    other model classes: correlated process noise (ell = 0.5, 1); heterogeneous diffusion + shear flow
  e1x 1200L1 / 1200L2   additional sensor layouts at n = 1200 (E10)
Outputs: results/*.csv, results/*.npz
"""
import os, sys, csv, time, math, itertools
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.stats import norm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
Z95 = norm.ppf(0.975)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def lap1d(N):
    e = np.ones(N); return sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1])

def upwind1d(N):
    e = np.ones(N); return sp.diags([-e[:-1], e], [-1, 0])

def lap2d_var(Nx, Ny, kappa):
    """5-point variable-coefficient Laplacian div(kappa grad), homogeneous Dirichlet
    boundaries, face conductivity = mean of the two node values (reduces to the
    constant-coefficient operator for kappa = 1)."""
    n = Nx * Ny; rows, cols, vals = [], [], []
    for j in range(Ny):
        for i in range(Nx):
            k = j * Nx + i; diag = 0.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < Nx and 0 <= jj < Ny:
                    kf = 0.5 * (kappa[k] + kappa[jj * Nx + ii])
                    rows.append(k); cols.append(jj * Nx + ii); vals.append(kf); diag -= kf
                else:
                    diag -= kappa[k]                       # Dirichlet ghost node
            rows.append(k); cols.append(k); vals.append(diag)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

class Model:
    """variant: "base" (Sec. 5.1), "corrq" (correlated process noise), or
    "het" (heterogeneous diffusivity and shear flow, Q = sw^2 I)."""
    def __init__(self, Nx, Ny, Fo=0.01, Pe=50.0, sw=0.05, sv=0.5, m=8, seed=0, variant="base", ell=2.0):
        self.Nx, self.Ny, self.n = Nx, Ny, Nx * Ny
        n = self.n
        Ix, Iy = sp.identity(Nx), sp.identity(Ny)
        Dx = sp.kron(Iy, upwind1d(Nx))
        if variant == "het":
            X, Yg = np.meshgrid(np.arange(Nx) / (Nx - 1), np.arange(Ny) / (Ny - 1))
            kappa = (1.0 + 0.8 * np.sin(2 * np.pi * X) * np.sin(np.pi * Yg)).ravel()   # 0.2 .. 1.8
            vel = (0.25 + 1.5 * Yg).ravel()                                               # shear flow, 0.25 .. 1.75
            L2 = lap2d_var(Nx, Ny, kappa)
            self.Fs = (sp.identity(n) + Fo * L2 - Fo * Pe * sp.diags(vel) @ Dx).tocsr()
        else:
            L2 = sp.kron(Iy, lap1d(Nx)) + sp.kron(lap1d(Ny), Ix)
            self.Fs = (sp.identity(n) + Fo * L2 - Fo * Pe * Dx).tocsr()
        self.variant = variant
        self.F = self.Fs.toarray()
        rng = np.random.default_rng(seed)
        interior = np.array([j * Nx + i for j in range(1, Ny - 1) for i in range(1, Nx - 1)])
        self.sens = np.sort(rng.choice(interior, size=min(m, len(interior)), replace=False))
        self.m = len(self.sens)
        self.H = np.zeros((self.m, n)); self.H[np.arange(self.m), self.sens] = 1.0
        self.sw, self.sv = sw, sv
        self.R = sv ** 2 * np.eye(self.m)
        self.HRH = self.H.T @ self.H / sv ** 2          # diagonal (point sensors)
        xy = np.array([(k % Nx, k // Nx) for k in range(n)])
        self.xy = xy
        self.dist = np.abs(xy[:, None, 0] - xy[None, :, 0]) + np.abs(xy[:, None, 1] - xy[None, :, 1])
        if variant == "corrq":
            # separable exponential correlation exp(-(|dx|+|dy|)/ell): positive definite, non-diagonal
            self.Q = sw ** 2 * np.exp(-self.dist / ell); self.qiso = False
        else:
            self.Q = sw ** 2 * np.eye(n); self.qiso = True
        self.Fo, self.Pe, self.seed = Fo, Pe, seed

    def stable(self):
        return np.max(np.abs(np.linalg.eigvals(self.F))) <= 1 + 1e-9

def sym(A): return 0.5 * (A + A.T)

def chol(A): return sla.cholesky(sym(A), lower=True)

def inv_from_chol(L):
    Li = sla.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    return Li.T @ Li

def lmax(A): return sla.eigh(sym(A), eigvals_only=True, subset_by_index=[A.shape[0] - 1, A.shape[0] - 1])[0]
def lmin(A): return sla.eigh(sym(A), eigvals_only=True, subset_by_index=[0, 0])[0]

def cov_gauss(ratio):
    """P(|Z| <= 1.96 sqrt(ratio)) for Z ~ N(0,1); ratio = reported var / true var."""
    return 2 * norm.cdf(Z95 * np.sqrt(np.clip(ratio, 0, None))) - 1

# ----------------------------------------------------------------------------
# Algorithm 1 + exact error moments
# ----------------------------------------------------------------------------
def run_alg1(mod, b, T=150, p0=0.01, diag=False, mc=0, mc_seed=1, lanczos_k=None,
             keep_series=False, cert_eps=None, cert_k=None, cert_seed=12345, smode="up", mc_mode="error", cert_mode="min",
             pv_tau=1e-3, pv_k=10):
    """cert_eps/cert_k: closed-loop randomised Lanczos certificate (Proposition 1);
    smode="down": conservative construction S_t = B_b(P^-1) - s_t I with s_t = -lmin(D_t)
    (never overstates precision, so the minimal certificate is 1 while S_t stays PD)."""
    crng = np.random.default_rng(cert_seed)
    n, F, Q = mod.n, mod.F, mod.Q
    mask = mod.dist <= b
    I = np.eye(n)
    Yinv = p0 * I                 # working posterior covariance Y_{t-1}^{-1}
    gam = 1.0
    M = p0 * I                    # exact error second moment (consistent start)
    Pkf = p0 * I                  # exact Kalman filter (optimal reference)
    ser = {k: [] for k in ["gamma", "rho_tr", "nees_tr", "covm_tr", "covm_c", "cov_worst_tr",
                           "cov_worst_c", "shift", "u", "eta", "Dinf", "sd_ratio_c", "sd_ratio_tr",
                           "rmse_ratio", "lam_exact", "kappa", "infeasible"]}
    lz = {k: [] for k in (lanczos_k or [])}
    if mc:
        rng = np.random.default_rng(mc_seed)
        E = np.sqrt(p0) * rng.standard_normal((n, mc))
        Lq = chol(Q)
        if mc_mode == "state":   # simulate true states, observations and the filter's own estimate
            Xs = E.copy(); Xh = np.zeros((n, mc))
        d2s = np.min(mod.dist[:, mod.sens], axis=1)
        unobs = (d2s > 3).astype(float)
        if unobs.sum() == 0: unobs = (d2s >= np.max(d2s)).astype(float)
        funcs = {"mean": np.ones(n) / n, "unobs": unobs / unobs.sum()}
        mcrec = {f"{f}_{w}": [] for f in list(funcs) + ["worst"] for w in ("tr", "c", "pred_tr", "pred_c")}
    t0 = time.perf_counter()
    diverged_at = -1
    pv = dict(fact=0, lanczos=0, steps=0, gmin=1.0)   # predict-and-verify bookkeeping (cert_mode="pv")
    for t in range(T):
        # --- exact KF reference
        Ppk = sym(F @ Pkf @ F.T + Q)
        Pkf = inv_from_chol(chol(inv_from_chol(chol(Ppk)) + mod.HRH))
        # --- Algorithm 1
        Pbar = sym(F @ Yinv @ F.T + Q)                         # line 4
        Lp = chol(Pbar); Sp = inv_from_chol(Lp)
        B = Sp * mask                                          # line 5: truncation
        lmB = lmin(B); mu = 1e-8 * np.mean(np.diag(B))
        infeasible = 0
        if smode == "down":
            sd_ = max(0.0, -lmin(Sp - B))                      # S = B - sd I  <=  P^{-1}
            if lmB - sd_ >= mu:
                s = -sd_
            else:                                              # cannot stay PD: fall back to upward shift
                infeasible = 1; s = max(0.0, mu - lmB)
        else:
            s = max(0.0, mu - lmB)
        S = B + s * I
        LS = chol(S)                                           # practical PD check
        Y = S + mod.HRH                                        # line 6
        LY = chol(Y); Yinv_new = inv_from_chol(LY)
        Pc_prior = gam * (Pbar - Q) + Q                        # eq. (Pc-prior)
        G = LS.T @ Pc_prior @ LS
        lam = lmax(G)
        if cert_eps is not None:                               # randomised Lanczos (Prop. 1), closed loop
            th = lanczos_max(lambda v: G @ v, n, cert_k, crng)
            gam = max(1.0, th / (1 - cert_eps))
        elif cert_mode == "pv":
            # estimate-then-verify in closed loop (Lemma lmi): warm-started k-step Lanczos gives theta <= lam;
            # the candidate theta(1+tau 2^j) is accepted iff lam <= candidate (one sparse LDL' test, simulated
            # exactly here). Bookkeeping: tests per step and the minimal-certificate chain for comparison.
            pv["steps"] += 1
            gmin_prev = pv["gmin"]; pv["gmin"] = max(1.0, (1 + 1e-10) * lmax(LS.T @ (gmin_prev * (Pbar - Q) + Q) @ LS))
            th, pv_v = lanczos_max(lambda v: G @ v, n, pv_k, crng, v0=pv.get("v"), return_vec=True); pv["v"] = pv_v
            pv["lanczos"] += 1; pv.setdefault("under", []).append(th / lam); j = 0
            while True:
                c = max(1.0, th * (1 + pv_tau * 2 ** j)); pv["fact"] += 1; j += 1
                if lam <= c: break
            gam = c
        elif cert_mode == "sharp":                             # optimal one-step certificate (Prop. sharp):
            Xs_ = sym(S @ Pc_prior @ S + mod.HRH)              #   lmax of the pencil (S Pc S + J, Y)
            gam = (1 + 1e-10) * sla.eigh(Xs_, Y, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]
        else:
            gam = max(1.0, (1 + 1e-10) * lam)                  # line 8 (dense, practical)
        if gam > 1e100:                                        # certificate diverged: stop (avoid overflow)
            diverged_at = t
            break
        ser["kappa"].append(gam / max(1.0, lam)); ser["infeasible"].append(infeasible)
        if lanczos_k:
            for k in lanczos_k:
                lz[k].append(lanczos_max(lambda v: G @ v, n, k, np.random.default_rng(1000 * t + k)) / lam)
        # --- exact error moments
        A = Yinv_new @ S; K = Yinv_new @ mod.H.T / mod.sv ** 2
        Mp = sym(F @ M @ F.T + Q)
        M = sym(A @ Mp @ A.T + mod.sv ** 2 * (K @ K.T))
        Yinv = Yinv_new
        # --- metrics: W = Y^{1/2} M Y^{1/2} (P = Y^{-1});  certified: W / gamma
        W = sym(LY.T @ M @ LY)
        wev, wvec = sla.eigh(W, subset_by_index=[n - 1, n - 1])
        rho = wev[0]
        dP, dM = np.diag(Yinv), np.diag(M)
        ser["gamma"].append(gam); ser["lam_exact"].append(lam)
        ser["rho_tr"].append(rho); ser["nees_tr"].append(np.trace(W) / n)
        ser["covm_tr"].append(np.mean(cov_gauss(dP / dM)))
        ser["covm_c"].append(np.mean(cov_gauss(gam * dP / dM)))
        ser["cov_worst_tr"].append(cov_gauss(1 / rho)); ser["cov_worst_c"].append(cov_gauss(gam / rho))
        ser["shift"].append(abs(s) if smode != "down" or infeasible else 0.0)
        ser["sd_ratio_c"].append(np.sqrt(gam * np.trace(Yinv) / np.trace(Pkf)))
        ser["sd_ratio_tr"].append(np.sqrt(np.trace(Yinv) / np.trace(Pkf)))
        ser["rmse_ratio"].append(np.sqrt(np.trace(M) / np.trace(Pkf)))
        if diag:
            ser["u"].append(lmax(Lp.T @ S @ Lp))
            if mod.qiso: ser["eta"].append(mod.sw ** 2 / lmax(Pbar))          # Q = sw^2 I
            else: ser["eta"].append(sla.eigh(Q, Pbar, eigvals_only=True, subset_by_index=[0, 0])[0])
            ser["Dinf"].append(np.abs(Sp - B).sum(1).max())
        if mc:
            Wn = Lq @ rng.standard_normal((n, mc))
            V = mod.sv * rng.standard_normal((mod.m, mc))
            if mc_mode == "state":
                Xs = F @ Xs + Wn                                 # x_t = F x_{t-1} + w_t
                Yobs = Xs[mod.sens] + V                          # y_t = H x_t + v_t
                Xb = F @ Xh                                      # line 3: prior mean
                Xh = Xb + K @ (Yobs - Xb[mod.sens])              # line 7: Y^{-1}(S xbar + H'R^{-1} y)
                E = Xh - Xs
            else:
                E = A @ (F @ E - Wn) + K @ V
            aw = LY @ wvec[:, 0]                                 # worst functional for TR
            for name, a in list(funcs.items()) + [("worst", aw)]:
                vP = a @ Yinv @ a; vM = a @ M @ a; proj = a @ E
                mcrec[f"{name}_tr"].append(np.mean(np.abs(proj) <= Z95 * np.sqrt(vP)))
                mcrec[f"{name}_c"].append(np.mean(np.abs(proj) <= Z95 * np.sqrt(gam * vP)))
                mcrec[f"{name}_pred_tr"].append(cov_gauss(vP / vM))
                mcrec[f"{name}_pred_c"].append(cov_gauss(gam * vP / vM))
    out = {k: np.array(v) for k, v in ser.items()}
    out["sec"] = time.perf_counter() - t0
    out["diverged_at"] = diverged_at
    pv.pop("v", None); out["pv"] = pv
    if lanczos_k: out["lanczos"] = {k: np.array(v) for k, v in lz.items()}
    if mc: out["mc"] = {k: np.array(v) for k, v in mcrec.items()}
    return out

def lanczos_max(matvec, n, k, rng, v0=None, return_vec=False):
    """k-step Lanczos, full reorthogonalisation, uniform random start (or warm start v0); largest Ritz value."""
    k = min(k, n)
    V = np.zeros((n, k)); alpha = np.zeros(k); beta = np.zeros(k)
    v = rng.standard_normal(n) if v0 is None else v0.copy(); v /= np.linalg.norm(v)
    for j in range(k):
        V[:, j] = v
        w = matvec(v)
        alpha[j] = v @ w
        w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-14 * max(1.0, abs(alpha[j])): k = j + 1; break
        v = w / beta[j]
    Tm = np.diag(alpha[:k]) + np.diag(beta[:k - 1], 1) + np.diag(beta[:k - 1], -1)
    if not return_vec: return np.linalg.eigvalsh(Tm)[-1]
    ev, U = np.linalg.eigh(Tm); return ev[-1], V[:, :k] @ U[:, -1]

def summarize(res, nss=50):
    g = lambda k: float(np.mean(res[k][-nss:]))
    out = dict(rho_tr_max=float(res["rho_tr"].max()), rho_c_max=float((res["rho_tr"] / res["gamma"]).max()),
               rho_tr_ss=g("rho_tr"), nees_tr_ss=g("nees_tr"), covm_tr_ss=g("covm_tr"), covm_c_ss=g("covm_c"),
               covw_tr_ss=g("cov_worst_tr"), covw_c_ss=g("cov_worst_c"),
               covw_tr_min=float(res["cov_worst_tr"].min()), covw_c_min=float(res["cov_worst_c"].min()),
               gamma_ss=g("gamma"), gamma_max=float(res["gamma"].max()),
               sdc_ss=g("sd_ratio_c"), sdtr_ss=g("sd_ratio_tr"), rmse_ss=g("rmse_ratio"),
               shift_steps=int((res["shift"] > 0).sum()), shift_max=float(res["shift"].max()),
               nees_c_ss=float(np.mean(res["nees_tr"][-nss:] / res["gamma"][-nss:])), sec=res["sec"])
    if len(res["u"]):
        out.update(u_ss=g("u"), eta_ss=g("eta"), Dinf_ss=g("Dinf"))
    return out

# ----------------------------------------------------------------------------
# E1: guarantee + tightness sweep
# ----------------------------------------------------------------------------
GRIDS = {240: (20, 12), 600: (30, 20), 1200: (40, 30)}

def _e1_job(a):
    n, sw, Pe, m, b, seed = a
    Nx, Ny = GRIDS[n]
    mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    if not mod.stable(): return None
    try:
        r = run_alg1(mod, b, T=150, diag=True)
    except (np.linalg.LinAlgError, sla.LinAlgError) as ex:
        return dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, status=f"fail:{ex}")
    return dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, status="ok", **summarize(r))

def e1(which="240"):
    if which == "240":
        jobs = list(itertools.product([240], [0.2, 0.05], [10.0, 50.0, 90.0], [8, 32], [1, 2, 3, 4], [0, 1, 2]))
    elif which == "600":
        jobs = list(itertools.product([600], [0.05], [10.0, 50.0, 90.0], [8, 32], [1, 2, 3, 4], [0, 1, 2]))
    else:
        jobs = list(itertools.product([1200], [0.05], [50.0], [8, 32], [1, 2, 3, 4], [0]))
    run_pool(_e1_job, jobs, f"e1_n{which}.csv")

def _e1x_job(a):
    """E1 configuration continued to T = 600 (settling check for Table 3)."""
    n, sw, Pe, m, b, seed = a
    Nx, Ny = GRIDS[n]
    mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed)
    if not mod.stable(): return None
    r = run_alg1(mod, b, T=600, diag=True)
    g = r["gamma"]
    return dict(n=n, sw=sw, Pe=Pe, m=m, b=b, seed=seed, status="ok",
                gamma_101_150=float(g[100:150].mean()), gamma_301_350=float(g[300:350].mean()),
                gamma_501_550=float(g[500:550].mean()), **summarize(r))

def e1x(which="1200"):
    if which == "1200":
        jobs = list(itertools.product([1200], [0.05], [50.0], [8, 32], [1, 2, 3, 4], [0]))
    elif which == "600":
        jobs = list(itertools.product([600], [0.05], [10.0, 50.0, 90.0], [8, 32], [1, 2], [0, 1, 2]))
    elif which == "240b34":
        jobs = list(itertools.product([240], [0.2, 0.05], [10.0, 50.0, 90.0], [8, 32], [3, 4], [0, 1, 2]))
    elif which == "600b34":  # Pe=10 is in e1x_T600_nb34.csv
        jobs = list(itertools.product([600], [0.05], [50.0, 90.0], [8, 32], [3, 4], [0, 1, 2]))
    elif which == "1200L1":   # additional sensor layouts at n = 1200 (E10)
        jobs = list(itertools.product([1200], [0.05], [50.0], [8, 32], [2, 3, 4], [1]))
    elif which == "1200L2":
        jobs = list(itertools.product([1200], [0.05], [50.0], [8, 32], [2, 3, 4], [2]))
    elif which == "b34":  # settling check for b>=3 in the slowest regime (Pe=10)
        jobs = list(itertools.product([240, 600], [0.05], [10.0], [8, 32], [3, 4], [0, 1, 2]))
    else:
        jobs = list(itertools.product([240], [0.2, 0.05], [10.0, 50.0, 90.0], [8, 32], [1, 2], [0, 1, 2]))
    run_pool(_e1x_job, jobs, f"e1x_T600_n{which}.csv")

# ----------------------------------------------------------------------------
# E9: independent Monte Carlo (simulated states, observations and estimates) in hard regimes
# ----------------------------------------------------------------------------
E9_CASES = {
    "lownoise":  dict(n=240, sw=0.01, Pe=10.0, m=8, b=2, seed=0, p0=0.01, T=150, mc=2000),
    "diffuse":   dict(n=600, sw=0.05, Pe=10.0, m=8, b=2, seed=0, p0=1.0, T=150, mc=2000),
    "b1large":   dict(n=600, sw=0.05, Pe=10.0, m=8, b=1, seed=0, p0=0.01, T=600, mc=2000),
    "growing":   dict(n=600, sw=0.05, Pe=10.0, m=8, b=3, seed=0, p0=0.01, T=600, mc=2000),
    "growing1200": dict(n=1200, sw=0.05, Pe=50.0, m=8, b=2, seed=0, p0=0.01, T=600, mc=1000),
}

def _e9_job(name):
    c = E9_CASES[name]; Nx, Ny = GRIDS[c["n"]]
    mod = Model(Nx, Ny, Pe=c["Pe"], sw=c["sw"], m=c["m"], seed=c["seed"])
    r = run_alg1(mod, c["b"], T=c["T"], p0=c["p0"], mc=c["mc"], mc_seed=11, mc_mode="state")
    mcr = r["mc"]; se = np.sqrt(0.95 * 0.05 / c["mc"])
    row = dict(case=name, **c, gamma_max=float(r["gamma"].max()), gamma_end=float(r["gamma"][-1]),
               rho_c_max=float((r["rho_tr"] / r["gamma"]).max()), se_nominal=float(se), sec=r["sec"])
    for f in ("mean", "unobs", "worst"):
        row[f"{f}_c_emp_min"] = float(mcr[f"{f}_c"].min())
        row[f"{f}_tr_emp_min"] = float(mcr[f"{f}_tr"].min())
        row[f"{f}_maxdiff_c"] = float(np.abs(mcr[f"{f}_c"] - mcr[f"{f}_pred_c"]).max())
        row[f"{f}_maxdiff_tr"] = float(np.abs(mcr[f"{f}_tr"] - mcr[f"{f}_pred_tr"]).max())
        row[f"{f}_c_emp_mean"] = float(mcr[f"{f}_c"].mean())
        row[f"{f}_c_below_3se"] = int((mcr[f"{f}_c"] < 0.95 - 3 * se).sum())
    return row, name, r

def e9(which="all"):
    from multiprocessing import Pool
    names = list(E9_CASES) if which == "all" else which.split(",")
    rows, save = [], {}
    with Pool(2) as p:
        for row, name, r in p.imap_unordered(_e9_job, names):
            rows.append(row)
            for k, v in r["mc"].items(): save[f"{name}__{k}"] = v
            save[f"{name}__gamma"] = r["gamma"]
            print("  e9 done", name, f"{row['sec']:.0f}s", flush=True)
    tag = "" if which == "all" else "_" + which.replace(",", "_")
    keys = list(rows[0])
    with open(os.path.join(OUT, f"e9_montecarlo{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(x) for x in rows]
    np.savez_compressed(os.path.join(OUT, f"e9_series{tag}.npz"), **save)

# ----------------------------------------------------------------------------
# E10: additional sensor layouts at n = 1200 (T = 600)  -> rows like e1x
# E11: other model classes at n = 600 (T = 600): correlated Q; heterogeneous diffusion + shear + correlated Q
# ----------------------------------------------------------------------------
def _e11_job(a):
    variant, ell, m, b, seed = a
    mod = Model(30, 20, Pe=50.0, sw=0.05, m=m, seed=seed, variant=variant, ell=ell)
    base = dict(variant=variant, ell=ell, m=m, b=b, seed=seed, n=600, sw=0.05, Pe=50.0)
    if not mod.stable(): return dict(**base, status="unstable")
    r = run_alg1(mod, b, T=600, diag=True)
    g = r["gamma"]
    if r["diverged_at"] >= 0:
        return dict(**base, status="diverged", diverged_at=r["diverged_at"],
                    rho_tr_max=float(r["rho_tr"].max()), rho_c_max=float((r["rho_tr"] / r["gamma"]).max()),
                    rho_tr_last=float(r["rho_tr"][-1]), covw_tr_min=float(r["cov_worst_tr"].min()))
    return dict(**base, status="ok",
                gamma_101_150=float(g[100:150].mean()), gamma_301_350=float(g[300:350].mean()),
                gamma_501_550=float(g[500:550].mean()), **summarize(r))

def e11():
    jobs = (list(itertools.product(["corrq"], [0.5, 1.0], [8, 32], [1, 2, 3, 4], [0, 1]))
            + list(itertools.product(["corrq"], [1.0], [8, 32], [5, 6], [0, 1]))
            + list(itertools.product(["het"], [0.0], [8, 32], [1, 2, 3, 4], [0, 1])))
    run_pool(_e11_job, jobs, "e11_model_classes.csv")

def run_pool(fn, jobs, fname, procs=2):
    from multiprocessing import Pool
    rows = []
    with Pool(procs) as p:
        for i, r in enumerate(p.imap_unordered(fn, jobs)):
            if r: rows.append(r)
            print(f"  [{fname}] {i + 1}/{len(jobs)}", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("n", "sw", "Pe", "m", "b", "seed", "status"), k))
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"wrote {fname} ({len(rows)} rows)")

# ----------------------------------------------------------------------------
# E3: value of the certificate (steady and diffuse start) + Monte Carlo check
# ----------------------------------------------------------------------------
def _e3_job(a):
    b, p0, tag = a
    mod = Model(20, 12, Pe=50.0, sw=0.05, m=8, seed=0)
    r = run_alg1(mod, b, T=150, p0=p0, mc=2000, mc_seed=7)
    return tag, b, p0, r

def e3():
    from multiprocessing import Pool
    jobs = [(1, 0.01, "steady"), (2, 0.01, "steady"), (1, 1.0, "diffuse"), (2, 1.0, "diffuse")]
    save = {}
    with Pool(2) as p:
        for tag, b, p0, r in p.imap_unordered(_e3_job, jobs):
            key = f"{tag}_b{b}"
            for k, v in r.items():
                if isinstance(v, dict):
                    for kk, vv in v.items(): save[f"{key}__{k}__{kk}"] = vv
                elif isinstance(v, np.ndarray): save[f"{key}__{k}"] = v
            print("  e3 done", key, f"{r['sec']:.0f}s", flush=True)
    np.savez_compressed(os.path.join(OUT, "e3_series.npz"), **save)
    print("wrote e3_series.npz")

# ----------------------------------------------------------------------------
# E4: decay of the truncation error with b (Assumption 2 evidence)
# ----------------------------------------------------------------------------
def _e4_job(a):
    n, sw = a
    Nx, Ny = GRIDS[n]
    mod = Model(Nx, Ny, Pe=50.0, sw=sw, m=8, seed=0)
    P = 0.01 * np.eye(n)
    for _ in range(120):                                       # exact KF to steady state
        Pp = sym(mod.F @ P @ mod.F.T + mod.Q)
        Sp = inv_from_chol(chol(Pp))
        P = inv_from_chol(chol(Sp + mod.HRH))
    Lp = chol(Pp); rows = []
    for b in range(1, 7):
        D = Sp - Sp * (mod.dist <= b)
        rel = np.max(np.abs(sla.eigh(sym(Lp.T @ D @ Lp), eigvals_only=True)))
        rows.append(dict(n=n, sw=sw, b=b, Dinf=np.abs(D).sum(1).max(), Dinf_rel=np.abs(D).sum(1).max() / lmin(Sp),
                         rel2=rel, pd=bool(lmin(Sp * (mod.dist <= b)) > 0)))
    return rows

def e4():
    from multiprocessing import Pool
    jobs = list(itertools.product([240, 600, 1200], [0.2, 0.05, 0.01]))
    rows = []
    with Pool(2) as p:
        for r in p.imap_unordered(_e4_job, jobs): rows += r; print("  e4", r[0]["n"], r[0]["sw"], flush=True)
    with open(os.path.join(OUT, "e4_decay.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    print("wrote e4_decay.csv")

# ----------------------------------------------------------------------------
# E5: cost study (sparse update + randomised Lanczos certificate) and Lanczos accuracy
# ----------------------------------------------------------------------------
def grid_for(nt):
    Ny = max(2, int(round(math.sqrt(nt / 2)))); Nx = max(2, int(round(nt / Ny))); return Nx, Ny

def banded_S(Nx, Ny, b):
    """Representative SPD matrix with the pattern N_b (cost study: values irrelevant)."""
    n = Nx * Ny
    offs = [(dx, dy) for dx in range(-b, b + 1) for dy in range(-b, b + 1) if 0 < abs(dx) + abs(dy) <= b]
    ix, iy = np.arange(n) % Nx, np.arange(n) // Nx
    rows, cols = [], []
    for dx, dy in offs:
        jx, jy = ix + dx, iy + dy; ok = (jx >= 0) & (jx < Nx) & (jy >= 0) & (jy < Ny)
        rows.append(np.arange(n)[ok]); cols.append((jy * Nx + jx)[ok])
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((-0.15 * np.ones(len(r)), (r, c)), shape=(n, n)); A = 0.5 * (A + A.T)
    return (A + sp.diags(np.abs(A).sum(1).A.ravel() + 1.0)).tocsc()

class SpChol:
    """Sparse LDL' of an SPD matrix via SuperLU in symmetric mode (no pivoting):
    S = P' L D L' P; exposes products with the Cholesky-type factor L_S = P' L D^{1/2}."""
    def __init__(self, S):
        from scipy.sparse.linalg import splu
        self.lu = splu(S, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True, Equil=False))
        assert np.all(self.lu.perm_r == self.lu.perm_c)
        self.p = self.lu.perm_c
        self.L = self.lu.L.tocsr(); self.d = self.lu.U.diagonal()
        assert np.all(self.d > 0)
        self.sd = np.sqrt(self.d); self.n = S.shape[0]
        self.pinv = np.empty_like(self.p); self.pinv[self.p] = np.arange(self.n)
        self.p, self.pinv = self.pinv, self.p      # SuperLU: S[perm_c][:, perm_c] = L D L'  (fixed convention)
    def LS(self, v):      # L_S v = P' L D^{1/2} v
        w = self.L @ (self.sd * v); out = np.empty_like(w); out[self.p] = w; return out
    def LSt(self, v):     # L_S' v = D^{1/2} L' P v
        return self.sd * (self.L.T @ v[self.p])
    def nnz(self): return self.lu.L.nnz + self.lu.U.nnz

def e5():
    from scipy.sparse.linalg import splu
    b = 2; rng = np.random.default_rng(0); rows = []
    eps_list = {0.05: None, 0.01: None}
    for nt in [1000, 3000, 10000, 30000, 100000]:
        Nx, Ny = grid_for(nt); n = Nx * Ny; m = max(8, n // 100)
        mod_F = (sp.identity(n) + 0.01 * (sp.kron(sp.identity(Ny), lap1d(Nx)) + sp.kron(lap1d(Ny), sp.identity(Nx)))
                 - 0.5 * sp.kron(sp.identity(Ny), upwind1d(Nx))).tocsr()
        S = banded_S(Nx, Ny, b)
        sens = rng.choice(n, m, replace=False); hrh = np.zeros(n); hrh[sens] = 4.0
        Y = (S + sp.diags(hrh)).tocsc()
        q = 0.0025; gam_prev = 1.05
        res = dict(n=n, Nx=Nx, Ny=Ny, b=b, m=m)
        t0 = time.perf_counter(); chS = SpChol(S); res["t_factor_S"] = time.perf_counter() - t0
        t0 = time.perf_counter(); luY = splu(Y, permc_spec="MMD_AT_PLUS_A"); res["t_factor_Y"] = time.perf_counter() - t0
        t0 = time.perf_counter(); luY.solve(rng.standard_normal(n)); res["t_mean_solve"] = time.perf_counter() - t0
        res["nnz_LY"] = luY.L.nnz + luY.U.nnz; res["nnz_LS"] = chS.nnz()
        Gmv = lambda v: chS.LSt(gam_prev * (mod_F @ luY.solve(mod_F.T @ chS.LS(v))) + q * chS.LS(v))
        T_h, pfail = 1000, 1e-6
        for eps in (0.05, 0.01, 0.002):                       # Prop. 1(ii), horizon T_h
            k = math.ceil((math.log(1.648 * math.sqrt(n) * T_h / pfail) / math.sqrt(eps) + 1) / 2)
            t0 = time.perf_counter(); th = lanczos_max(Gmv, n, k, rng)
            res[f"k_eps{eps}"] = k; res[f"t_lanczos_eps{eps}"] = time.perf_counter() - t0
        # Prop. 1(iii), all-time guarantee, iteration count at t = T_h, eps = 0.002
        k = math.ceil((1 + math.log(1.648 * math.sqrt(n) * math.pi ** 2 * T_h ** 2 / (6 * pfail)) / math.sqrt(0.002)) / 2)
        t0 = time.perf_counter(); th = lanczos_max(Gmv, n, k, rng)
        res["k_all_eps0.002"] = k; res["t_lanczos_all_eps0.002"] = time.perf_counter() - t0
        if n <= 3100:                                          # dense KF step for contrast
            Pd = np.eye(n) + 0.01 * rng.standard_normal((n, n)); Pd = Pd @ Pd.T
            t0 = time.perf_counter(); Fd = mod_F.toarray(); Pp = Fd @ Pd @ Fd.T + q * np.eye(n)
            np.linalg.inv(np.linalg.inv(Pp) + np.diag(hrh)); res["t_dense_kf"] = time.perf_counter() - t0
        else:
            res["t_dense_kf"] = float("nan")
        res["mem_dense_MB"] = n * n * 8 / 1e6
        rows.append(res); print("  e5", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    with open(os.path.join(OUT, "e5_cost.csv"), "w", newline="") as f:
        keys = list(rows[0]); [keys.append(k) for r in rows for k in r if k not in keys]
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    # Lanczos accuracy on the actual Algorithm-1 operators G_t (n = 240, 600)
    acc = []
    for n_ in (240, 600):
        Nx, Ny = GRIDS[n_]
        r = run_alg1(Model(Nx, Ny, Pe=50.0, sw=0.05, m=8, seed=0), 2, T=60, lanczos_k=[5, 10, 20, 40])
        for k, ratios in r["lanczos"].items():
            for eps in (0.05, 0.01):
                acc.append(dict(n=n_, k=k, eps=eps, min_ratio=float(ratios.min()),
                                frac_upper=float(np.mean(ratios / (1 - eps) >= 1.0)),
                                median_ratio=float(np.median(ratios))))
        print("  e5 lanczos accuracy", n_, flush=True)
    with open(os.path.join(OUT, "e5_lanczos_accuracy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(acc[0])); w.writeheader(); [w.writerow(x) for x in acc]
    print("wrote e5_cost.csv, e5_lanczos_accuracy.csv")

# ----------------------------------------------------------------------------
# E13: sharp (optimal one-step) certificate vs the minimal certificate of Algorithm 1
def _e13_job(a):
    n_, m, b, seed = a
    Nx, Ny = GRIDS[n_]
    r = run_alg1(Model(Nx, Ny, Pe=50.0, sw=0.05, m=m, seed=seed), b, T=600, cert_mode="sharp")
    out = dict(n=n_, m=m, b=b, seed=seed, cert="sharp", diverged_at=r.get("diverged_at", -1))
    out.update(summarize(r)); g = r["gamma"]
    for a_, z in ((101, 150), (301, 350), (501, 550)): out[f"gamma_{a_}_{z}"] = float(np.mean(g[a_ - 1:z]))
    out["frac_below1"] = float(np.mean(g < 1))
    return out

def e13(which="small"):
    jobs = [(n_, m, b, 0) for n_ in (240, 600) for m in (8, 32) for b in (1, 2, 3, 4)] if which == "small" \
        else [(1200, 8, 2, 0), (1200, 8, 3, 0)]  # "big" or "big2"
    run_pool(_e13_job, jobs, f"e13_sharp_{which}.csv" if which != "big2" else "e13_sharp_big.csv", procs=1)

# ----------------------------------------------------------------------------
# E15: predict-and-verify certificate in closed loop
def _e15_job(a):
    n_, Pe, m, b, seed = a
    Nx, Ny = GRIDS[n_]
    r = run_alg1(Model(Nx, Ny, Pe=Pe, sw=0.05, m=m, seed=seed), b, T=600, cert_mode="pv")
    g = r["gamma"]; pv = r["pv"]
    u_ = np.array(pv["under"])
    out = dict(n=n_, Pe=Pe, m=m, b=b, seed=seed, fact_per_step=pv["fact"] / pv["steps"],
               under_min=float(u_.min()), under_med=float(np.median(u_)),
               gamma_end=float(g[-1]), gmin_end=pv["gmin"],
               ratio_end=float(g[-1]) / pv["gmin"], gamma_ss=float(np.mean(g[-50:])))
    out.update({k: v for k, v in summarize(r).items() if k in ("rho_c_max", "covw_c_min", "rho_tr_ss")})
    return out

def e15():
    jobs = [(n_, 50.0, m, b, 0) for n_ in (240, 600) for m in (8, 32) for b in (2, 3, 4)] + [(600, 10.0, 8, 3, 0)]
    run_pool(_e15_job, jobs, "e15_predict_verify.csv", procs=1)

# ----------------------------------------------------------------------------
# E6: low process noise -- gamma_t as diagnostic
# ----------------------------------------------------------------------------
def _e6_job(b):
    mod = Model(20, 12, Pe=10.0, sw=0.01, m=8, seed=0)
    return b, run_alg1(mod, b, T=150, diag=True)

def e6():
    from multiprocessing import Pool
    save = {}
    with Pool(2) as p:
        for b, r in p.imap_unordered(_e6_job, [1, 2, 3, 4]):
            for k, v in r.items():
                if isinstance(v, np.ndarray): save[f"b{b}__{k}"] = v
            print("  e6 b", b, summarize(r), flush=True)
    np.savez_compressed(os.path.join(OUT, "e6_series.npz"), **save)
    print("wrote e6_series.npz")

# ----------------------------------------------------------------------------
# E4w: decay of the truncation error along the SPARSE filter's own trajectory
#      (Assumption 2 concerns the working prior P_{t|t-1}, which depends on b)
# ----------------------------------------------------------------------------
def _e4w_job(a):
    n, sw, b = a
    Nx, Ny = GRIDS[n]
    mod = Model(Nx, Ny, Pe=50.0, sw=sw, m=8, seed=0)
    mask = mod.dist <= b; I = np.eye(n); Yinv = 0.01 * I
    for _ in range(100):                                       # working recursion of Algorithm 1
        Pbar = sym(mod.F @ Yinv @ mod.F.T + mod.Q)
        Lp = chol(Pbar); Sp = inv_from_chol(Lp); B = Sp * mask
        lmB = lmin(B); mu = 1e-8 * np.mean(np.diag(B)); s = max(0.0, mu - lmB)
        Yinv = inv_from_chol(chol(B + s * I + mod.HRH))
    D = Sp - B
    rel2 = float(np.max(np.abs(sla.eigh(sym(Lp.T @ D @ Lp), eigvals_only=True))))
    Dinf = float(np.abs(D).sum(1).max())
    return dict(n=n, sw=sw, b=b, Dinf=Dinf, Dinf_rel=Dinf / float(np.abs(Sp).sum(1).max()),
                rel2=rel2, shift_last=s, pd_trunc=bool(lmB > 0))

def e4w():
    from multiprocessing import Pool
    jobs = [(n, sw, b) for n in (240, 600) for sw in (0.2, 0.05, 0.01) for b in range(1, 7)]
    jobs += [(1200, 0.05, b) for b in range(1, 7)]
    rows = []
    with Pool(2) as p:
        for r in p.imap_unordered(_e4w_job, jobs): rows.append(r); print("  e4w", r["n"], r["sw"], r["b"], flush=True)
    rows.sort(key=lambda r: (r["n"], -r["sw"], r["b"]))
    with open(os.path.join(OUT, "e4w_decay_working.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    print("wrote e4w_decay_working.csv")

# ----------------------------------------------------------------------------
# Theorem 2 check: bound gamma* vs observed certificate (n = 240)
# ----------------------------------------------------------------------------
def _thm2_job(a):
    b, p0 = a
    r = run_alg1(Model(20, 12, Pe=50.0, sw=0.05, m=8, seed=0), b, T=150, p0=p0, diag=True)
    u, eta, g = r["u"][-50:], r["eta"][-50:], r["gamma"]
    ub, el = float(u.max()), float(eta.min()); rho = ub * (1 - el)
    gstar = max(1.0, ub * el / (1 - rho)) if rho < 1 else float("inf")
    return dict(b=b, p0=p0, u_max=ub, eta_min=el, rho=rho, gamma_star=gstar,
                gamma_obs_ss=float(g[-50:].mean()), gamma_end=float(g[-1]), gamma_max=float(g.max()))

def thm2():
    from multiprocessing import Pool
    with Pool(2) as p: rows = list(p.map(_thm2_job, [(1, 0.01), (2, 0.01), (3, 0.01), (4, 0.01), (2, 1.0)]))
    with open(os.path.join(OUT, "thm2_bound_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

# ----------------------------------------------------------------------------
# E7: closed-loop randomised Lanczos certificate (Proposition 1) vs dense certificate
# ----------------------------------------------------------------------------
def kw_k(n, eps, T, pfail):
    return math.ceil((math.log(1.648 * math.sqrt(n) * T / pfail) / math.sqrt(eps) + 1) / 2)

def _e7_job(a):
    n, b, eps = a
    Nx, Ny = GRIDS[n]; T = 400
    mod = Model(Nx, Ny, Pe=50.0, sw=0.05, m=8, seed=0)
    if eps == 0:
        r = run_alg1(mod, b, T=T); k = 0
    else:
        k = min(n, kw_k(n, eps, T, 1e-6)); r = run_alg1(mod, b, T=T, cert_eps=eps, cert_k=k)
    g = r["gamma"]; rc = r["rho_tr"] / g; kap = r["kappa"]
    return dict(n=n, b=b, eps=eps, k=k, gamma_ss=float(g[-100:].mean()), gamma_max=float(g.max()),
                gamma_t200=float(g[199]), gamma_end=float(g[-1]), kappa_med=float(np.median(kap)),
                kappa_min=float(kap.min()), rc_max=float(rc.max()), sdc_ss=float(r["sd_ratio_c"][-100:].mean()),
                covw_c_min=float(r["cov_worst_c"].min()))

def e7():
    from multiprocessing import Pool
    jobs = [(n, b, e) for n in (240, 600) for b in (1, 2, 3, 4) for e in (0, 0.05, 0.01, 0.002)]
    rows = []
    with Pool(2) as p:
        for r in p.imap_unordered(_e7_job, jobs): rows.append(r); print("  e7", r["n"], r["b"], r["eps"], flush=True)
    rows.sort(key=lambda r: (r["n"], r["b"], r["eps"]))
    with open(os.path.join(OUT, "e7_randomised_closed_loop.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    print("wrote e7_randomised_closed_loop.csv")

# ----------------------------------------------------------------------------
# E8: conservative (downward-shift) construction vs truncation + certificate
# ----------------------------------------------------------------------------
def _e8_job(a):
    n, b, smode = a
    Nx, Ny = GRIDS[n]
    r = run_alg1(Model(Nx, Ny, Pe=50.0, sw=0.05, m=8, seed=0), b, T=150, smode=smode)
    return dict(n=n, b=b, smode=smode, infeasible_steps=int(np.sum(r["infeasible"])),
                gamma_ss=float(r["gamma"][-50:].mean()), sdc_ss=float(r["sd_ratio_c"][-50:].mean()),
                rmse_ss=float(r["rmse_ratio"][-50:].mean()), rc_max=float((r["rho_tr"] / r["gamma"]).max()),
                covw_c_min=float(r["cov_worst_c"].min()))

def e8():
    from multiprocessing import Pool
    jobs = [(n, b, m) for n in (240, 600) for b in (1, 2, 3, 4) for m in ("up", "down")]
    with Pool(2) as p: rows = list(p.map(_e8_job, jobs))
    with open(os.path.join(OUT, "e8_conservative_baseline.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        mod = Model(20, 12, Pe=50.0, sw=0.05, m=8, seed=0)
        r = run_alg1(mod, 2, T=60, diag=True, mc=500)
        print({k: round(v, 4) for k, v in summarize(r, 20).items()})
        print({k: round(float(np.mean(v[-20:])), 4) for k, v in r["mc"].items()})
    elif mode == "e1": e1(sys.argv[2] if len(sys.argv) > 2 else "240")
    elif mode == "e3": e3()
    elif mode == "e4": e4()
    elif mode == "e5": e5()
    elif mode == "e6": e6()
    elif mode == "e4w": e4w()
    elif mode == "thm2": thm2()
    elif mode == "e7": e7()
    elif mode == "e1x": e1x(sys.argv[2] if len(sys.argv) > 2 else "1200")
    elif mode == "e8": e8()
    elif mode == "e9": e9(sys.argv[2] if len(sys.argv) > 2 else "all")
    elif mode == "e11": e11()
    elif mode == "e15": e15()
    elif mode == "e13": e13(sys.argv[2] if len(sys.argv) > 2 else "small")
