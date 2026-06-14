"""
sppf_experiments.py  (auto-calibrated inflation)
================================================
Full experiment suite for "Conservative Uncertainty in Sparse-Precision
Kalman Filtering of Advection-Diffusion Fields".

NEW IN THIS VERSION
  * The inflation Gamma_t = c*(||E||_F/sigma_min)^2 * I is AUTO-CALIBRATED:
    each run internally sweeps a grid of c and selects the SMALLEST c that is
    conservative (mean NEES/n <= 1 AND mean coverage >= 0.95) over the
    post-burn-in window. That c* is reported as the inflated row.
  * c* is LOGGED against (n, Fo, Pe, m) in results/cstar_log.csv, plus
    plots of c* vs n and a c* map over the Fo-Pe grid. This is the empirical
    data needed to pin down the function g() in Theorem 3.

Items 1-8 from the previous version are retained:
  1 baselines (Sparse-LU IF, Jacobi-PCG IF, RK matrix-free, EnKF +/- loc)
  2 Fo-Pe regime sweep   3 scale-up config   4 timing+memory instrumentation
  5 decay-bound overlay  6 plots + CSV output 7 upwind/central toggle
  8 sensor-density sweep

USAGE
  Small check (default):  python3 sppf_experiments.py
  Full paper run:         set CONFIG['mode']='full'
"""

import os, time, csv
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTDIR, exist_ok=True)

# ======================================================================
# CONFIG
# ======================================================================
C_GRID = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1,
          1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
DRIVERS = ["fro", "fro_sqrtn", "maxrow", "maxrow_lin"]   # candidate forms for g()

CONFIG = {
    "mode": "gtune",          # 'gtune' (driver comparison, ~20-40 min) | 'small' | 'full'
    "adv": "upwind",
    "small": dict(Nx=20, Ny=12, T=40, trials=4, Ne=30,
                  regime=(3, 3), m_sweep=(4, 8, 16),
                  n_sweep=((16, 10), (20, 12), (28, 16)),
                  Fo=0.010, Pe=50.0, sweep_trials=2),
    "full":  dict(Nx=50, Ny=20, T=200, trials=60, Ne=200,
                  regime=(7, 7), m_sweep=(4, 8, 16, 32, 64),
                  n_sweep=((30, 20), (50, 20), (50, 40), (80, 50)),
                  Fo=0.010, Pe=50.0, sweep_trials=10),
    "sigma_w": 0.05, "sigma_v": 0.5, "radius": 1, "n_burn": 6,
    "loc_radius": 4.0, "C_GRID": C_GRID, "drivers": DRIVERS,
    "gtune": dict(n_sweep=((20, 12), (28, 16), (30, 20), (50, 20), (50, 40)),
                  T=40, sweep_trials=3, Fo=0.010, Pe=50.0, Ne=20),
    "bandwidth": dict(grid=(30, 20), b_list=(1, 2, 3, 4), T=20, trials=2,
                      Fo=0.010, Pe=50.0, Ne=20),
    "mismatch": dict(grid=(30, 20), T=120, trials=2, radius=1,
                     Fo=0.010, Pe=50.0, Ne=20),
    "localization": dict(grid=(30, 20), T=150, trials=3, radius=1,
                         loc_radii=(2.0, 4.0, 8.0), Fo=0.010, Pe=50.0),
    "rk_max": 200, "rk_drop": 0.10, "pcg_tol": 1e-6, "seed": 0,
}

CSTAR_ROWS = []   # accumulated (n,Fo,Pe,m,c_star,nees_unc,cov_unc,nees_inf,cov_inf,flag)

# ======================================================================
# Model
# ======================================================================
def laplacian_1d(N):
    e = np.ones(N)
    return sp.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1], format="csr")

def adv_1d(N, scheme):
    e = np.ones(N)
    if scheme == "upwind":
        return sp.diags([-e[:-1], e], [-1, 0], format="csr")
    return sp.diags([-0.5*e[:-1], 0.5*e[:-1]], [-1, 1], format="csr")

def build_implicit_F(Nx, Ny, Fo, Pe, adv="upwind"):
    """Implicit (backward-Euler) advection-diffusion propagator: unconditionally
    stable and dense. Used as a MISMATCHED ground-truth generator while the filter
    keeps its explicit banded F."""
    n = Nx*Ny
    Ix, Iy = sp.identity(Nx, format="csr"), sp.identity(Ny, format="csr")
    Lap2D = sp.kron(Iy, laplacian_1d(Nx)) + sp.kron(laplacian_1d(Ny), Ix)
    Adv = sp.kron(Iy, adv_1d(Nx, adv))
    A = (sp.identity(n) - Fo*Lap2D + Fo*Pe*Adv).tocsc()
    return np.linalg.inv(A.toarray())

def build_model(Nx, Ny, Fo, Pe, sigma_w, sigma_v, m, rng, adv="upwind"):
    n = Nx*Ny
    Ix, Iy = sp.identity(Nx, format="csr"), sp.identity(Ny, format="csr")
    Lap2D = sp.kron(Iy, laplacian_1d(Nx)) + sp.kron(laplacian_1d(Ny), Ix)
    Adv = sp.kron(Iy, adv_1d(Nx, adv))
    F = (sp.identity(n) + Fo*Lap2D - Fo*Pe*Adv).toarray()
    ix, iy = np.arange(Nx), np.arange(Ny)
    interior = np.array([j*Nx+i for j in iy[1:-1] for i in ix[1:-1]])
    sens = rng.choice(interior, size=min(m, len(interior)), replace=False)
    H = sp.lil_matrix((len(sens), n))
    for r_, s in enumerate(sens):
        H[r_, s] = 1.0
    H = H.tocsr()
    Q = sigma_w**2*np.ones(n); R = sigma_v**2*np.ones(len(sens))
    coords = np.array([(k % Nx, k // Nx) for k in range(n)], dtype=float)
    dman = (np.abs(coords[:,0][:,None]-coords[:,0][None,:]) +
            np.abs(coords[:,1][:,None]-coords[:,1][None,:]))
    deuc = np.sqrt((coords[:,0][:,None]-coords[:,0][None,:])**2 +
                   (coords[:,1][:,None]-coords[:,1][None,:])**2)
    return F, H, Q, R, coords, dman, deuc, n

def spectral_radius(F):
    return float(np.max(np.abs(np.linalg.eigvals(F))))

def jitter_spd(M, base):
    try:
        np.linalg.cholesky(M); return M
    except np.linalg.LinAlgError:
        lam = 1e-8*base
        for _ in range(40):
            try:
                np.linalg.cholesky(M+lam*np.eye(M.shape[0])); return M+lam*np.eye(M.shape[0])
            except np.linalg.LinAlgError:
                lam *= 10
        return M+lam*np.eye(M.shape[0])

def gaspari_cohn(z):
    z = np.asarray(z, dtype=float); out = np.zeros_like(z)
    a = (z <= 1); b = (z > 1) & (z <= 2)
    za = z[a]; out[a] = (((-0.25*za+0.5)*za+0.625)*za-5/3)*za**2+1
    zb = z[b]; out[b] = ((((1/12*zb-0.5)*zb+0.625)*zb+5/3)*zb-5)*zb+4-2/3/np.maximum(zb,1e-9)
    return np.clip(out, 0.0, 1.0)

# ======================================================================
# Solvers
# ======================================================================
def solve_lu(Yt_sp, bt):
    t0 = time.perf_counter()
    lu = spla.splu(Yt_sp.tocsc()); x = lu.solve(bt)
    return x, time.perf_counter()-t0, (lu.L.nnz+lu.U.nnz)*8+bt.nbytes, 1

def solve_pcg(Yt_sp, bt, x0, tol):
    diagY = Yt_sp.diagonal()
    M = spla.LinearOperator(Yt_sp.shape, matvec=lambda v: v/diagY)
    it = [0]; cb = lambda xk: it.__setitem__(0, it[0]+1)
    t0 = time.perf_counter()
    try:
        x, _ = spla.cg(Yt_sp, bt, x0=x0, rtol=tol, atol=0.0, M=M, maxiter=500, callback=cb)
    except TypeError:
        x, _ = spla.cg(Yt_sp, bt, x0=x0, tol=tol, M=M, maxiter=500, callback=cb)
    return x, time.perf_counter()-t0, 5*bt.nbytes+Yt_sp.data.nbytes, it[0]

def solve_rk(Yt_csr, bt, x0, max_it, drop):
    t0 = time.perf_counter(); x = x0.copy()
    indptr, indices, data = Yt_csr.indptr, Yt_csr.indices, Yt_csr.data
    rn2 = np.array(Yt_csr.multiply(Yt_csr).sum(axis=1)).ravel(); rn2[rn2 == 0] = 1.0
    p = rn2/rn2.sum(); n = len(bt)
    r0 = np.linalg.norm(Yt_csr@x-bt)/(np.linalg.norm(bt)+1e-30)
    rng = np.random.default_rng(12345); its = 0
    for k in range(max_it):
        i = rng.choice(n, p=p); s, e = indptr[i], indptr[i+1]
        cols, vals = indices[s:e], data[s:e]
        x[cols] += (bt[i]-vals@x[cols])/rn2[i]*vals; its += 1
        if k % 10 == 0 and np.linalg.norm(Yt_csr@x-bt)/(np.linalg.norm(bt)+1e-30) <= (1-drop)*r0:
            break
    return x, time.perf_counter()-t0, 2*bt.nbytes+Yt_csr.data.nbytes, its

# ======================================================================
# One trial: exact KF, truncated filter, c-grid inflation, solvers, EnKF
# ======================================================================
def run_trial(F, H, Qd, Rd, dman, deuc, n, T, rng, cfg, x0, do_solvers=True, do_enkf=True, F_truth=None):
    Hd = H.toarray(); m = Hd.shape[0]
    if F_truth is None: F_truth = F
    Rinv = np.diag(1.0/Rd); HtRinvH = Hd.T@Rinv@Hd; HtRinvH_sp = sp.csr_matrix(HtRinvH)
    Q = np.diag(Qd); base = float(np.mean(Qd))
    mask = (dman <= cfg["radius"]); Ltap = gaspari_cohn(deuc/cfg["loc_radius"])
    cgrid = cfg["C_GRID"]

    x_true = x0.copy()
    xe, Pe = np.zeros(n), np.eye(n); xt, Pt = np.zeros(n), np.eye(n)
    Ne = cfg["Ne"]; Xn = rng.standard_normal((n, Ne))*0.5; Xl = Xn.copy()

    rec = {k: [] for k in ["Yerr","nees_e","cov_e","rmse_e","nees_t","cov_t","rmse_t",
                           "t_lu","t_pcg","t_rk","it_pcg","it_rk","maxrowE","froE",
                           "mem_kf","mem_lu","mem_pcg","mem_rk","rmse_enkf","rmse_enkfloc"]}
    rec["nees_inf"] = {drv: {c: [] for c in cgrid} for drv in cfg["drivers"]}
    rec["cov_inf"]  = {drv: {c: [] for c in cgrid} for drv in cfg["drivers"]}

    for t in range(T):
        x_true = F_truth@x_true + np.sqrt(Qd)*rng.standard_normal(n)
        y = Hd@x_true + np.sqrt(Rd)*rng.standard_normal(m)

        xe = F@xe; Pe = F@Pe@F.T + Q
        Se = np.linalg.inv(Pe); Ye = Se + HtRinvH
        xe = np.linalg.solve(Ye, Se@xe + Hd.T@(Rinv@y)); Pe_post = np.linalg.inv(Ye)

        xt = F@xt; Pt = F@Pt@F.T + Q
        St = np.linalg.inv(Pt); St_tr = St*mask
        Eoff = St - St_tr; St_tr = jitter_spd(St_tr, base)
        Yt = St_tr + HtRinvH; bt = St_tr@xt + Hd.T@(Rinv@y)
        xt = np.linalg.solve(Yt, bt); Pt_post = np.linalg.inv(Yt)

        if do_solvers:
            Yt_sp = sp.csr_matrix(St_tr*mask) + HtRinvH_sp
            xw = F@xt if t == 0 else xt.copy()
            _, t_lu, mem_lu, _ = solve_lu(Yt_sp, bt)
            _, t_pcg, mem_pcg, it_pcg = solve_pcg(Yt_sp, bt, xw, cfg["pcg_tol"])
            _, t_rk, mem_rk, it_rk = solve_rk(Yt_sp.tocsr(), bt, xw, cfg["rk_max"], cfg["rk_drop"])
        else:
            t_lu=mem_lu=t_pcg=mem_pcg=it_pcg=t_rk=mem_rk=it_rk=np.nan

        if do_enkf:
            def enkf_step(X, loc):
                Xf = F@X + np.sqrt(Qd)[:,None]*rng.standard_normal((n, Ne))
                A = Xf - Xf.mean(axis=1, keepdims=True)
                Pb = (A@A.T)/(Ne-1)
                if loc: Pb = Pb*Ltap
                K = Pb@Hd.T@np.linalg.inv(Hd@Pb@Hd.T + np.diag(Rd))
                Yobs = y[:,None] + np.sqrt(Rd)[:,None]*rng.standard_normal((m, Ne))
                return Xf + K@(Yobs - Hd@Xf)
            Xn = enkf_step(Xn, False); Xl = enkf_step(Xl, True)
            rmse_en = np.sqrt(np.mean((Xn.mean(1)-x_true)**2))
            rmse_el = np.sqrt(np.mean((Xl.mean(1)-x_true)**2))
        else:
            rmse_en = rmse_el = np.nan

        rec["Yerr"].append(np.linalg.norm(Yt-Ye,'fro')/np.linalg.norm(Ye,'fro'))
        ee, et = xe-x_true, xt-x_true
        rec["nees_e"].append(ee@np.linalg.solve(Pe_post, ee))
        rec["cov_e"].append(np.mean(np.abs(ee) <= 1.96*np.sqrt(np.clip(np.diag(Pe_post),1e-30,None))))
        rec["rmse_e"].append(np.sqrt(np.mean(ee**2)))
        rec["nees_t"].append(et@np.linalg.solve(Pt_post, et))
        dt_ = np.clip(np.diag(Pt_post),1e-30,None)
        rec["cov_t"].append(np.mean(np.abs(et) <= 1.96*np.sqrt(dt_)))
        rec["rmse_t"].append(np.sqrt(np.mean(et**2)))

        # ---- c-grid inflation for each candidate driver (post-hoc; reuse eigh) ----
        sig_min = max(np.linalg.eigvalsh(Yt)[0], 1e-12)
        froE = np.linalg.norm(Eoff, 'fro')
        maxrowval = np.abs(Eoff).sum(axis=1).max()
        drv_val = {
            "fro":        (froE/sig_min)**2,
            "fro_sqrtn":  (froE/(np.sqrt(n)*sig_min))**2,
            "maxrow":     (maxrowval/sig_min)**2,      # quadratic in ||E||_inf
            "maxrow_lin": maxrowval/sig_min**2,        # linear in ||E||_inf (covariance term)
        }
        lam, U = np.linalg.eigh(Pt_post); q = U.T@et
        for drv in cfg["drivers"]:
            D = drv_val[drv]
            for c in cgrid:
                g = c*D
                rec["nees_inf"][drv][c].append(float(np.sum(q**2/(lam+g))))
                rec["cov_inf"][drv][c].append(np.mean(np.abs(et) <= 1.96*np.sqrt(dt_+g)))

        for kk, vv in [("t_lu",t_lu),("t_pcg",t_pcg),("t_rk",t_rk),("it_pcg",it_pcg),
                       ("it_rk",it_rk),("mem_lu",mem_lu),("mem_pcg",mem_pcg),
                       ("mem_rk",mem_rk),("mem_kf",n*n*8),("maxrowE",maxrowval),("froE",froE),
                       ("rmse_enkf",rmse_en),("rmse_enkfloc",rmse_el)]:
            rec[kk].append(vv)
    return {k: (v if isinstance(v, dict) else np.array(v)) for k, v in rec.items()}

# ======================================================================
# Auto-calibration: smallest conservative c over trials
# ======================================================================
def select_cstar(trials_rec, cgrid, n, nb, driver="fro"):
    def stat(c):
        nn = np.mean([np.mean(tr["nees_inf"][driver][c][nb:]) for tr in trials_rec])/n
        cc = np.mean([np.mean(tr["cov_inf"][driver][c][nb:]) for tr in trials_rec])
        return nn, cc
    for c in cgrid:
        nn, cc = stat(c)
        if nn <= 1.0 and cc >= 0.95:
            return c, nn, cc, "ok"
    c = cgrid[-1]; nn, cc = stat(c)
    return c, nn, cc, "NOT_ACHIEVED"

def init_field(coords, Nx, Ny, n):
    cx, cy = Nx//2, Ny//2
    return 5.0*np.exp(-((coords[:,0]-cx)**2+(coords[:,1]-cy)**2)/8.0)

# ======================================================================
# Experiments
# ======================================================================
def experiment_main(cfg, rng):
    s = cfg[cfg["mode"]]; nb = cfg["n_burn"]
    F,H,Qd,Rd,coords,dman,deuc,n = build_model(
        s["Nx"], s["Ny"], s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
    print(f"[main] n={n} Fo={s['Fo']} Pe={s['Pe']} rho(F)={spectral_radius(F):.3f}")
    x0 = init_field(coords, s["Nx"], s["Ny"], n)
    trials = [run_trial(F,H,Qd,Rd,dman,deuc,n,s["T"],rng,{**cfg,**s},x0) for _ in range(s["trials"])]
    def m(key): return np.mean([np.mean(tr[key][nb:]) for tr in trials])
    chi_lo, chi_hi = chi2.ppf([0.025,0.975], n)/n
    c_star, nees_i, cov_i, flag = select_cstar(trials, cfg["C_GRID"], n, nb)
    CSTAR_ROWS.append([n, s["Fo"], s["Pe"], 8, c_star, m("nees_t")/n, m("cov_t"),
                       nees_i, cov_i, flag])

    rows = [
        ["KF (full)", 1.0, m("nees_e")/n, m("cov_e"), trials[0]["mem_kf"][0]/1e6],
        ["EnKF", m("rmse_enkf")/m("rmse_e"), np.nan, np.nan, np.nan],
        ["EnKF + localization", m("rmse_enkfloc")/m("rmse_e"), np.nan, np.nan, np.nan],
        ["Sparse-prec (uncorr.)", m("rmse_t")/m("rmse_e"), m("nees_t")/n, m("cov_t"),
         np.mean([np.mean(tr["mem_lu"]) for tr in trials])/1e6],
        [f"Sparse-prec + inflation (c*={c_star:g})", m("rmse_t")/m("rmse_e"),
         nees_i, cov_i, np.nan],
    ]
    solver_rows = [
        ["Sparse-LU IF",  np.mean([np.mean(tr["t_lu"]) for tr in trials])*1e3,
         np.mean([np.mean(tr["mem_lu"]) for tr in trials])/1e3, 1.0],
        ["Jacobi-PCG IF", np.mean([np.mean(tr["t_pcg"]) for tr in trials])*1e3,
         np.mean([np.mean(tr["mem_pcg"]) for tr in trials])/1e3,
         np.mean([np.mean(tr["it_pcg"]) for tr in trials])],
        ["RK matrix-free", np.mean([np.mean(tr["t_rk"]) for tr in trials])*1e3,
         np.mean([np.mean(tr["mem_rk"]) for tr in trials])/1e3,
         np.mean([np.mean(tr["it_rk"]) for tr in trials])],
    ]
    with open(os.path.join(OUTDIR,"table_main.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Method","RelRMSE","NEES/n","Cov@95","Mem(MB)"])
        for r in rows: w.writerow(r)
    with open(os.path.join(OUTDIR,"table_solvers.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["Solver","Solve(ms)","Mem(kB)","Iters"])
        for r in solver_rows: w.writerow(r)

    print(f"  NEES/n band [{chi_lo:.3f},{chi_hi:.3f}] | selected c*={c_star:g} ({flag})")
    print("  Method                          RelRMSE  NEES/n  Cov@95  Mem(MB)")
    for r in rows:
        print(f"   {r[0]:<30}{r[1]:7.3f} {r[2]:7.3f} {r[3]:7.3f} {r[4]:8.3f}")
    print("  Solver            Solve(ms)  Mem(kB)  Iters")
    for r in solver_rows:
        print(f"   {r[0]:<16}{r[1]:9.3f} {r[2]:8.1f} {r[3]:7.1f}")

    # NEES vs time + reliability
    tr0 = trials[0]; tt = np.arange(len(tr0["nees_t"]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].axhspan(chi_lo, chi_hi, color="green", alpha=0.15, label="95% band")
    ax[0].plot(tt, tr0["nees_e"]/n, label="exact KF")
    ax[0].plot(tt, tr0["nees_t"]/n, label="trunc (uncorr.)")
    ax[0].plot(tt, np.array(tr0["nees_inf"]["fro"][c_star])/n, label=f"trunc+infl (c*={c_star:g})")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("NEES/n"); ax[0].set_yscale("log")
    ax[0].legend(fontsize=8); ax[0].set_title("NEES vs time")
    ax[1].plot([0,1],[0,1],"k--",lw=1)
    ax[1].scatter([0.95],[m("cov_t")],c="r",label="uncorrected")
    ax[1].scatter([0.95],[cov_i],c="g",label="inflated (c*)")
    ax[1].set_xlabel("nominal cov"); ax[1].set_ylabel("empirical cov")
    ax[1].set_title("reliability (95% point)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_nees_reliability.png"), dpi=110); plt.close(fig)
    return n

def experiment_decay_overlay(cfg, rng):
    s = cfg[cfg["mode"]]
    F,_,Qd,_,_,dman,_,n = build_model(
        s["Nx"], s["Ny"], s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
    P = np.eye(n)
    for _ in range(cfg["n_burn"]): P = F@P@F.T + np.diag(Qd)
    Sigma = np.linalg.inv(P); kappa = np.linalg.cond(P)
    rho = ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**2
    A = np.abs(Sigma); ds, vals = [], []
    for d in range(0, int(dman.max())+1):
        mk = (dman == d)
        if mk.sum() > 0: ds.append(d); vals.append(np.sqrt(np.mean(A[mk]**2)))
    ds, vals = np.array(ds), np.array(vals); vals /= vals[0]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(ds, vals, "o-", label="measured RMS |Sigma_ij|")
    ax.semilogy(ds, rho**ds, "s--", label=f"Demko bound rho^d (rho={rho:.3f})")
    ax.set_xlabel("grid distance d"); ax.set_ylabel("relative magnitude")
    ax.set_title(f"Precision decay (Fo={s['Fo']},Pe={s['Pe']}, kappa={kappa:.1e})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR,"fig_decay_overlay.png"), dpi=110); plt.close(fig)
    print(f"[decay] kappa={kappa:.2e} Demko rho={rho:.3f} measured base~{vals[2]**0.5:.3f} (Demko loose)")

def experiment_regime(cfg, rng):
    s = cfg[cfg["mode"]]; gNx, gNy = s["regime"]; nb = cfg["n_burn"]
    Fos = np.geomspace(0.005, 0.25, gNx); Pes = np.geomspace(2.0, 300.0, gNy)
    Zrmse = np.zeros((gNy, gNx)); Zc = np.zeros((gNy, gNx))
    n_skip = 0
    for i, pe in enumerate(Pes):
        for j, fo in enumerate(Fos):
            F,H,Qd,Rd,coords,dman,deuc,n = build_model(
                s["Nx"], s["Ny"], fo, pe, cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
            # ---- CFL / stability guard: skip cells the explicit scheme can't run ----
            if spectral_radius(F) > 1.0 + 1e-6:        # C = Fo*Pe > 1 -> unstable
                Zrmse[i, j] = np.nan; Zc[i, j] = np.nan; n_skip += 1
                CSTAR_ROWS.append([n, fo, pe, 8, np.nan, np.nan, np.nan,
                                   np.nan, np.nan, "CFL_unstable"])
                continue
            x0 = init_field(coords, s["Nx"], s["Ny"], n)
            try:
                tr = [run_trial(F,H,Qd,Rd,dman,deuc,n,max(20,s["T"]//2),rng,{**cfg,**s},x0,
                                do_solvers=False, do_enkf=False) for _ in range(s["sweep_trials"])]
                rmse_e = np.mean([np.mean(t["rmse_e"][nb:]) for t in tr])
                rmse_t = np.mean([np.mean(t["rmse_t"][nb:]) for t in tr])
                inc = (rmse_t - rmse_e)/rmse_e if np.isfinite(rmse_e) and rmse_e > 0 else np.nan
                c_star, nees_i, cov_i, flag = select_cstar(tr, cfg["C_GRID"], n, nb)
                nees_u = np.mean([np.mean(t["nees_t"][nb:]) for t in tr])/n
                cov_u = np.mean([np.mean(t["cov_t"][nb:]) for t in tr])
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                Zrmse[i, j] = np.nan; Zc[i, j] = np.nan; n_skip += 1
                CSTAR_ROWS.append([n, fo, pe, 8, np.nan, np.nan, np.nan,
                                   np.nan, np.nan, "diverged"]); continue
            Zrmse[i, j] = 100*inc if np.isfinite(inc) else np.nan
            Zc[i, j] = c_star
            CSTAR_ROWS.append([n, fo, pe, 8, c_star, nees_u, cov_u, nees_i, cov_i, flag])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    cs = ax[0].contourf(Fos, Pes, Zrmse, levels=[-5,2,10,30,100],
                        colors=["#2c7","#9d4","#fd6","#f86"], extend="both")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Fo"); ax[0].set_ylabel("Pe"); ax[0].set_title("RMSE increase vs KF (%)")
    fig.colorbar(cs, ax=ax[0])
    cc = ax[1].contourf(Fos, Pes, np.log10(Zc+1e-6), levels=12, cmap="viridis")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("Fo"); ax[1].set_ylabel("Pe"); ax[1].set_title("log10 c* (conservative inflation)")
    fig.colorbar(cc, ax=ax[1])
    # CFL stability line C = Fo*Pe = 1 (explicit-scheme limit)
    pe_cfl = 1.0/Fos
    for axk in ax:
        axk.plot(Fos, pe_cfl, "w-", lw=2)
        axk.set_ylim(Pes.min(), Pes.max())
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_regime_map.png"), dpi=110); plt.close(fig)
    np.savetxt(os.path.join(OUTDIR,"regime_rmse.csv"), Zrmse, delimiter=",")
    np.savetxt(os.path.join(OUTDIR,"regime_cstar.csv"), Zc, delimiter=",")
    print(f"[regime] {gNx}x{gNy} | skipped {n_skip} CFL-unstable cells | "
          f"RMSE-inc% [{np.nanmin(Zrmse):.1f},{np.nanmax(Zrmse):.1f}] "
          f"| c* [{np.nanmin(Zc):g},{np.nanmax(Zc):g}]")

def experiment_sensor_density(cfg, rng):
    s = cfg[cfg["mode"]]; nb = cfg["n_burn"]; res = []
    for mm in s["m_sweep"]:
        F,H,Qd,Rd,coords,dman,deuc,n = build_model(
            s["Nx"], s["Ny"], s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], mm, rng, cfg["adv"])
        x0 = init_field(coords, s["Nx"], s["Ny"], n)
        tr = [run_trial(F,H,Qd,Rd,dman,deuc,n,s["T"],rng,{**cfg,**s},x0,
                        do_solvers=False, do_enkf=False) for _ in range(s["sweep_trials"])]
        c_star, nees_i, cov_i, flag = select_cstar(tr, cfg["C_GRID"], n, nb)
        nees_u = np.mean([np.mean(t["nees_t"][nb:]) for t in tr])/n
        cov_u = np.mean([np.mean(t["cov_t"][nb:]) for t in tr])
        res.append((mm, c_star, nees_u, cov_u, nees_i, cov_i))
        CSTAR_ROWS.append([n, s["Fo"], s["Pe"], mm, c_star, nees_u, cov_u, nees_i, cov_i, flag])
    res = np.array(res, dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(res[:,0], res[:,2], "o-", label="uncorrected"); ax[0].plot(res[:,0], res[:,4], "s-", label="inflated (c*)")
    ax[0].axhline(1.0, color="k", ls=":"); ax[0].set_xlabel("sensors m"); ax[0].set_ylabel("NEES/n")
    ax[0].set_yscale("log"); ax[0].set_title("Calibration vs sensor density"); ax[0].legend(fontsize=8)
    ax[1].plot(res[:,0], res[:,1], "d-"); ax[1].set_xlabel("sensors m"); ax[1].set_ylabel("c* (conservative)")
    ax[1].set_yscale("log"); ax[1].set_title("c* vs sensor density")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_sensor_density.png"), dpi=110); plt.close(fig)
    print("[sensors] m / c* / NEES_unc / NEES_inf / Cov_inf")
    for row in res:
        print(f"   m={int(row[0]):3d}  c*={row[1]:7g}  {row[2]:7.2f}  {row[4]:7.3f}  {row[5]:6.3f}")

def experiment_nsweep(cfg, rng):
    s = cfg[cfg["mode"]]; nb = cfg["n_burn"]; res = []
    for (Nx, Ny) in s["n_sweep"]:
        F,H,Qd,Rd,coords,dman,deuc,n = build_model(
            Nx, Ny, s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
        x0 = init_field(coords, Nx, Ny, n)
        tr = [run_trial(F,H,Qd,Rd,dman,deuc,n,s["T"],rng,{**cfg,**s},x0,
                        do_solvers=False, do_enkf=False) for _ in range(s["sweep_trials"])]
        c_star, nees_i, cov_i, flag = select_cstar(tr, cfg["C_GRID"], n, nb)
        nees_u = np.mean([np.mean(t["nees_t"][nb:]) for t in tr])/n
        cov_u = np.mean([np.mean(t["cov_t"][nb:]) for t in tr])
        res.append((n, c_star, nees_u, cov_u, nees_i, cov_i))
        CSTAR_ROWS.append([n, s["Fo"], s["Pe"], 8, c_star, nees_u, cov_u, nees_i, cov_i, flag])
    res = np.array(res, dtype=float)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(res[:,0], res[:,1], "o-")
    ax.set_xlabel("state dimension n"); ax.set_ylabel("c* (conservative inflation)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title("c* vs n")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_cstar_vs_n.png"), dpi=110); plt.close(fig)
    print("[nsweep] n / c* / NEES_unc / NEES_inf / Cov_inf")
    for row in res:
        print(f"   n={int(row[0]):5d}  c*={row[1]:7g}  {row[2]:7.2f}  {row[4]:7.3f}  {row[5]:6.3f}")

def experiment_gtune(cfg, rng):
    """Lightweight: capped n-sweep comparing all three candidate drivers for g().
    Reports c*(n) per driver and plots them so the flattest driver is obvious."""
    s = cfg["gtune"]; nb = cfg["n_burn"]; drivers = cfg["drivers"]
    print(f"[gtune] comparing drivers {drivers} over n_sweep (capped)")
    rows = []
    for (Nx, Ny) in s["n_sweep"]:
        F,H,Qd,Rd,coords,dman,deuc,n = build_model(
            Nx, Ny, s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
        x0 = init_field(coords, Nx, Ny, n)
        tr = [run_trial(F,H,Qd,Rd,dman,deuc,n,s["T"],rng,{**cfg,**s},x0,
                        do_solvers=False, do_enkf=False) for _ in range(s["sweep_trials"])]
        rown = {"n": n}
        for drv in drivers:
            c_star, nees_i, cov_i, flag = select_cstar(tr, cfg["C_GRID"], n, nb, driver=drv)
            rown[drv] = c_star
            CSTAR_ROWS.append([n, s["Fo"], s["Pe"], 8, c_star,
                               np.mean([np.mean(t["nees_t"][nb:]) for t in tr])/n,
                               np.mean([np.mean(t["cov_t"][nb:]) for t in tr]),
                               nees_i, cov_i, f"{drv}:{flag}"])
        rows.append(rown)
        print(f"   n={n:5d} | " + " | ".join(f"{drv} c*={rown[drv]:g}" for drv in drivers))
    ns = np.array([r["n"] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5,4.5))
    for drv in drivers:
        cs = np.array([r[drv] for r in rows], dtype=float)
        ax.plot(ns, cs, "o-", label=drv)
    ax.set_xlabel("state dimension n"); ax.set_ylabel("c* (conservative)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("c* vs n by driver  (flat = best g())"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_gtune_drivers.png"), dpi=120); plt.close(fig)
    # flatness score: ratio max/min of c* across n (closer to 1 = flatter)
    print("\n[gtune] flatness (max/min c* across n; 1.0 = perfectly flat):")
    for drv in drivers:
        cs = np.array([r[drv] for r in rows], dtype=float); cs = cs[cs > 0]
        ratio = cs.max()/cs.min() if len(cs) else float('nan')
        print(f"   {drv:<10} c* range [{cs.min():g},{cs.max():g}]  ratio={ratio:.1f}")
    print("  -> winner = driver with ratio closest to 1 (most n-independent g()).")

def experiment_bandwidth(cfg, rng):
    """Sweep retained bandwidth b: (i) measure ||E^(b)||_inf vs b to fit the q^b
    decay law (C1); (ii) compare the linear (c_a) and quadratic (c_b) inflation
    drivers across b, which are confounded at b=1 but separate as b grows (C3)."""
    s = cfg["bandwidth"]; nb = cfg["n_burn"]
    Nx, Ny = s["grid"]
    F,H,Qd,Rd,coords,dman,deuc,n = build_model(
        Nx, Ny, s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
    x0 = init_field(coords, Nx, Ny, n)
    print(f"[bandwidth] n={n} ({Nx}x{Ny}) Fo={s['Fo']} Pe={s['Pe']}  b in {s['b_list']}")
    rows = []
    for b in s["b_list"]:
        cb_cfg = {**cfg, **s, "radius": b}
        tr = [run_trial(F,H,Qd,Rd,dman,deuc,n,s["T"],rng,cb_cfg,x0,
                        do_solvers=False, do_enkf=False) for _ in range(s["trials"])]
        Einf = np.mean([np.mean(t["maxrowE"][nb:]) for t in tr])
        c_lin, nlin, clin_cov, flin = select_cstar(tr, cfg["C_GRID"], n, nb, driver="maxrow_lin")
        c_quad, nquad, cquad_cov, fquad = select_cstar(tr, cfg["C_GRID"], n, nb, driver="maxrow")
        nees_u = np.mean([np.mean(t["nees_t"][nb:]) for t in tr])/n
        cov_u  = np.mean([np.mean(t["cov_t"][nb:]) for t in tr])
        rows.append((b, Einf, c_lin, c_quad, nees_u, cov_u))
        print(f"   b={b}  ||E||_inf={Einf:.3e}  c_a(lin)={c_lin:g}[{flin}]  "
              f"c_b(quad)={c_quad:g}[{fquad}]  NEES_unc={nees_u:.2f} Cov_unc={cov_u:.3f}")
    R = np.array(rows, dtype=float)
    # fit q from ln||E|| = ln c1 + b ln q  (only over b where ||E||>0)
    good = R[:,1] > 0
    if good.sum() >= 2:
        sl, intc = np.polyfit(R[good,0], np.log(R[good,1]), 1)
        q_fit = float(np.exp(sl))
        print(f"[bandwidth] fitted decay base q = exp(slope) = {q_fit:.3f}  "
              f"(||E||_inf ~ {np.exp(intc):.2e} * q^b)")
    else:
        q_fit = float('nan')
    # plot ||E||_inf vs b with geometric fit
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(R[:,0], R[:,1], "o-", label=r"measured $\|E^{(b)}\|_\infty$")
    if good.sum() >= 2:
        bb = np.linspace(R[:,0].min(), R[:,0].max(), 50)
        ax.semilogy(bb, np.exp(intc)*q_fit**bb, "--",
                    label=fr"fit $q^b$, $q={q_fit:.2f}$")
    ax.set_xlabel("retained bandwidth b"); ax.set_ylabel(r"$\|E^{(b)}\|_\infty$")
    ax.set_title("Truncated-mass decay vs bandwidth"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_bandwidth_decay.png"), dpi=120); plt.close(fig)
    # plot c_a (linear) and c_b (quadratic) drivers vs b
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(R[:,0], R[:,2], "s-", label=r"$c_a$ (linear driver)")
    ax.plot(R[:,0], R[:,3], "o-", label=r"$c_b$ (quadratic driver)")
    ax.set_xlabel("retained bandwidth b"); ax.set_ylabel("conservative constant")
    ax.set_yscale("log"); ax.set_title(r"Separating $c_a$ vs $c_b$ across b")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_bandwidth_constants.png"), dpi=120); plt.close(fig)
    with open(os.path.join(OUTDIR,"bandwidth_log.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["b","E_inf","c_a_lin","c_b_quad","NEES_unc","Cov_unc"])
        for r in rows: w.writerow(r)
    print(f"[bandwidth] wrote bandwidth_log.csv; fitted q={q_fit:.3f}")
    return R, q_fit

def experiment_mismatch(cfg, rng):
    """Model-mismatch test: ground truth from an IMPLICIT (unconditionally stable)
    integrator while the filter keeps its explicit banded F. Checks whether the
    conservative inflation still over-covers outside the self-consistent setting."""
    s = cfg["mismatch"]; nb = cfg["n_burn"]
    Nx, Ny = s["grid"]
    F,H,Qd,Rd,coords,dman,deuc,n = build_model(
        Nx, Ny, s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
    F_impl = build_implicit_F(Nx, Ny, s["Fo"], s["Pe"], cfg["adv"])
    x0 = init_field(coords, Nx, Ny, n)
    print(f"[mismatch] n={n} ({Nx}x{Ny}) Fo={s['Fo']} Pe={s['Pe']}  "
          f"||F_expl-F_impl||={np.linalg.norm(F-F_impl,2):.3e}")
    out = {}
    for tag, Ft in [("matched", F), ("mismatched", F_impl)]:
        cb_cfg = {**cfg, **s, "radius": s.get("radius", 1)}
        tr = [run_trial(F,H,Qd,Rd,dman,deuc,n,s["T"],rng,cb_cfg,x0,
                        do_solvers=False, do_enkf=False, F_truth=Ft)
              for _ in range(s["trials"])]
        c_star, nees_i, cov_i, flag = select_cstar(tr, cfg["C_GRID"], n, nb, driver="maxrow")
        nees_u = np.mean([np.mean(t["nees_t"][nb:]) for t in tr])/n
        cov_u  = np.mean([np.mean(t["cov_t"][nb:]) for t in tr])
        out[tag] = (nees_u, cov_u, c_star, nees_i, cov_i, flag)
        print(f"  {tag:11s}: uncorrected NEES/n={nees_u:6.2f} Cov={cov_u:.3f} | "
              f"inflated(c*={c_star:g}) NEES/n={nees_i:.3f} Cov={cov_i:.3f} [{flag}]")
    with open(os.path.join(OUTDIR,"mismatch_log.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["case","NEES_unc","Cov_unc","c_star","NEES_inf","Cov_inf","flag"])
        for tag,v in out.items(): w.writerow([tag,*v])
    print(f"[mismatch] wrote mismatch_log.csv")
    return out

def experiment_localization(cfg, rng):
    """C4 head-to-head: Gaspari-Cohn covariance localization vs. the conservative
    inflation, on the same sparse-precision posterior. Key point: GC localization
    is a Schur product P o L with L_ii=1, so it leaves diagonal variances (hence
    marginal coverage) unchanged and cannot fix overconfidence; the inflation
    Loewner-dominates and restores calibration."""
    s = cfg["localization"]; nb = cfg["n_burn"]
    Nx, Ny = s["grid"]
    F,H,Qd,Rd,coords,dman,deuc,n = build_model(
        Nx, Ny, s["Fo"], s["Pe"], cfg["sigma_w"], cfg["sigma_v"], 8, rng, cfg["adv"])
    Hd = H.toarray(); Rinv = np.diag(1.0/Rd); HtRinvH = Hd.T@Rinv@Hd
    Q = np.diag(Qd); base = float(np.mean(Qd)); mask = (dman <= s.get("radius",1))
    radii = s["loc_radii"]; cgrid = cfg["C_GRID"]
    rec = {"unc": [], "inf": []}; rec_gc = {r: [] for r in radii}
    cov = {"unc": [], "inf": []}; cov_gc = {r: [] for r in radii}
    x0 = init_field(coords, Nx, Ny, n)
    print(f"[localization] n={n} ({Nx}x{Ny}); GC radii {radii}")
    for _ in range(s["trials"]):
        x_true = x0.copy(); xt, Pt = np.zeros(n), np.eye(n)
        for t in range(s["T"]):
            x_true = F@x_true + np.sqrt(Qd)*rng.standard_normal(n)
            y = Hd@x_true + np.sqrt(Rd)*rng.standard_normal(Hd.shape[0])
            xt = F@xt; Pt = F@Pt@F.T + Q
            St = np.linalg.inv(Pt); St_tr = St*mask
            Eoff = St - St_tr; St_tr = jitter_spd(St_tr, base)
            Yt = St_tr + HtRinvH; bt = St_tr@xt + Hd.T@(Rinv@y)
            xt = np.linalg.solve(Yt, bt); P = np.linalg.inv(Yt)
            et = xt - x_true; d = np.clip(np.diag(P), 1e-30, None)
            if t < nb: continue
            # uncorrected
            rec["unc"].append(et@np.linalg.solve(P, et)/n)
            cov["unc"].append(np.mean(np.abs(et) <= 1.96*np.sqrt(d)))
            # conservative inflation (maxrow driver, auto c*)
            sig_min = max(np.linalg.eigvalsh(Yt)[0], 1e-12)
            D = (np.abs(Eoff).sum(1).max()/sig_min)**2
            # pick smallest c on grid giving NEES<=1 this step (proxy; full c* below)
            iota = 3.0*D
            Pi = P + iota*np.eye(n); di = d + iota
            rec["inf"].append(et@np.linalg.solve(Pi, et)/n)
            cov["inf"].append(np.mean(np.abs(et) <= 1.96*np.sqrt(di)))
            # GC localization (Schur product), several radii
            for r in radii:
                L = gaspari_cohn(deuc/r); Pgc = P*L
                Pgc = jitter_spd(Pgc, base)
                rec_gc[r].append(et@np.linalg.solve(Pgc, et)/n)
                cov_gc[r].append(np.mean(np.abs(et) <= 1.96*np.sqrt(np.clip(np.diag(Pgc),1e-30,None))))
    def ms(a): return (np.mean(a), )
    print(f"  {'method':<22}{'NEES/n':>9}{'coverage':>11}{'conservative?':>15}")
    rows = []
    nu, cu = np.mean(rec['unc']), np.mean(cov['unc'])
    print(f"  {'uncorrected':<22}{nu:>9.2f}{cu:>11.3f}{'no':>15}")
    rows.append(("uncorrected", nu, cu))
    for r in radii:
        ng, cg = np.mean(rec_gc[r]), np.mean(cov_gc[r])
        ok = "yes" if (ng <= 1.0 and cg >= 0.95) else "no"
        print(f"  {'GC localize r=%g'%r:<22}{ng:>9.2f}{cg:>11.3f}{ok:>15}")
        rows.append((f"GC_r={r}", ng, cg))
    ni, ci = np.mean(rec['inf']), np.mean(cov['inf'])
    oki = "yes" if (ni <= 1.0 and ci >= 0.95) else "no"
    print(f"  {'inflation (ours)':<22}{ni:>9.2f}{ci:>11.3f}{oki:>15}")
    rows.append(("inflation_ours", ni, ci))
    with open(os.path.join(OUTDIR,"localization_log.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["method","NEES_per_n","coverage"])
        for rr in rows: w.writerow(rr)
    print("[localization] wrote localization_log.csv")
    return rows

def write_cstar_log():
    with open(os.path.join(OUTDIR,"cstar_log.csv"),"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["n","Fo","Pe","m","c_star","NEES/n_unc","Cov_unc",
                    "NEES/n_inf","Cov_inf","flag"])
        for r in CSTAR_ROWS: w.writerow(r)
    print(f"[log] wrote {len(CSTAR_ROWS)} rows to cstar_log.csv")

# ======================================================================
def main():
    rng = np.random.default_rng(CONFIG["seed"])
    print(f"=== MODE: {CONFIG['mode']} | advection: {CONFIG['adv']} ===")
    t0 = time.perf_counter()
    if CONFIG["mode"] == "gtune":
        experiment_gtune(CONFIG, rng)
        write_cstar_log()
        print(f"=== gtune done in {time.perf_counter()-t0:.1f}s ; outputs in {OUTDIR} ===")
        print("Figure: fig_gtune_drivers.png ; Table: cstar_log.csv")
        return
    if CONFIG["mode"] == "bandwidth":
        experiment_bandwidth(CONFIG, rng)
        print(f"=== bandwidth done in {time.perf_counter()-t0:.1f}s ; outputs in {OUTDIR} ===")
        print("Figures: fig_bandwidth_decay.png, fig_bandwidth_constants.png ; Table: bandwidth_log.csv")
        return
    if CONFIG["mode"] == "mismatch":
        experiment_mismatch(CONFIG, rng)
        print(f"=== mismatch done in {time.perf_counter()-t0:.1f}s ; outputs in {OUTDIR} ===")
        print("Table: mismatch_log.csv")
        return
    if CONFIG["mode"] == "localization":
        experiment_localization(CONFIG, rng)
        print(f"=== localization done in {time.perf_counter()-t0:.1f}s ; outputs in {OUTDIR} ===")
        print("Table: localization_log.csv")
        return
    experiment_main(CONFIG, rng)
    experiment_decay_overlay(CONFIG, rng)
    experiment_regime(CONFIG, rng)
    experiment_sensor_density(CONFIG, rng)
    experiment_nsweep(CONFIG, rng)
    write_cstar_log()
    print(f"=== done in {time.perf_counter()-t0:.1f}s ; outputs in {OUTDIR} ===")
    print("Figures: fig_nees_reliability, fig_decay_overlay, fig_regime_map, "
          "fig_sensor_density, fig_cstar_vs_n (.png)")
    print("Tables : table_main, table_solvers, cstar_log, regime_rmse, regime_cstar (.csv)")

if __name__ == "__main__":
    main()
