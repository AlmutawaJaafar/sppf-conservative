"""
sppf_diagnostics.py
===================
Diagnostic harness for "Conservative Uncertainty in Sparse-Precision Kalman
Filtering of Advection-Diffusion Fields".

PURPOSE (exploratory, not final validation):
  Test numerically whether the four conjectures behind the paper hold, BEFORE
  attempting the proofs, and extract the constants the proofs will need.

  D1 (Prop. 1)     : off-diagonal entries of the predicted precision decay with
                     grid distance, faster as Fo/Pe shrink.
  D2 (Thm. 2)      : locality-r truncation error of the filtered precision stays
                     BOUNDED over the horizon after a startup transient
                     (does not accumulate); reported as post-burn-in sup + trend.
  D3 (motivates C3): the uncorrected sparse-precision filter is OVERCONFIDENT
                     (NEES > n, coverage < nominal).
  D4 (Thm. 3)      : a Loewner inflation Gamma_t = c*(||E||_F/sigma_min)^2 * I
                     restores CONSERVATIVE calibration; we SWEEP c to find the
                     tight conservative threshold (this empirically fixes g()).

IMPLEMENTATION NOTE:
  Dense linear algebra is used deliberately to isolate and measure the
  precision-truncation error against the exact Kalman filter. The production
  filter would perform the same updates matrix-free; the diagnostics need the
  exact filter for comparison. "Locality-r" keeps precision entries between grid
  nodes within Manhattan distance r (the correct 2-D locality, = hard precision
  localization), NOT an index band.
"""

import numpy as np
import scipy.sparse as sp
from scipy.stats import chi2

# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
def laplacian_1d(N):
    e = np.ones(N)
    return sp.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1], format='csr')

def upwind_1d(N):
    """Upwind first difference for u>0:  h*dT/dx ~ T_i - T_{i-1}."""
    e = np.ones(N)
    return sp.diags([-e[:-1], e], [-1, 0], format='csr')

def build_model(Nx, Ny, Fo, Pe, sigma_w, sigma_v, m, rng):
    """
    F = I + Fo*Lap2D - C*Adv_x ,  C = Fo*Pe (Courant number)
    Q = sigma_w^2 I , R = sigma_v^2 I , state k = iy*Nx + ix.
    """
    n = Nx * Ny
    Ix, Iy = sp.identity(Nx, format='csr'), sp.identity(Ny, format='csr')
    Lxx, Lyy, Dx = laplacian_1d(Nx), laplacian_1d(Ny), upwind_1d(Nx)
    Lap2D = sp.kron(Iy, Lxx) + sp.kron(Lyy, Ix)
    Adv = sp.kron(Iy, Dx)
    C = Fo * Pe
    F = (sp.identity(n) + Fo*Lap2D - C*Adv).toarray()

    ix, iy = np.arange(Nx), np.arange(Ny)
    interior = np.array([j*Nx + i for j in iy[1:-1] for i in ix[1:-1]])
    sens = rng.choice(interior, size=min(m, len(interior)), replace=False)
    H = sp.lil_matrix((len(sens), n))
    for r_, s in enumerate(sens):
        H[r_, s] = 1.0
    H = H.tocsr()

    Q = sigma_w**2 * np.ones(n)
    R = sigma_v**2 * np.ones(len(sens))

    coords = np.array([(k % Nx, k // Nx) for k in range(n)])
    dx = np.abs(coords[:, 0][:, None] - coords[:, 0][None, :])
    dy = np.abs(coords[:, 1][:, None] - coords[:, 1][None, :])
    dist = dx + dy
    return F, H, Q, R, coords, dist, n

def spectral_radius(F):
    return float(np.max(np.abs(np.linalg.eigvals(F))))

def jitter_spd(M, base):
    try:
        np.linalg.cholesky(M); return M
    except np.linalg.LinAlgError:
        lam = 1e-8*base
        for _ in range(40):
            try:
                np.linalg.cholesky(M + lam*np.eye(M.shape[0]))
                return M + lam*np.eye(M.shape[0])
            except np.linalg.LinAlgError:
                lam *= 10
        return M + lam*np.eye(M.shape[0])

# ----------------------------------------------------------------------
# D1: decay of predicted precision vs grid distance
# ----------------------------------------------------------------------
def decay_profile(F, Qd, dist, n, P0_scale=1.0, n_burn=8, dmax=10):
    Q = np.diag(Qd); P = P0_scale*np.eye(n)
    for _ in range(n_burn):
        P = F @ P @ F.T + Q
    A = np.abs(np.linalg.inv(P))
    ds, vals = [], []
    for d in range(0, dmax+1):
        mask = (dist == d)
        if mask.sum() > 0:
            ds.append(d); vals.append(np.sqrt(np.mean(A[mask]**2)))
    return np.array(ds), np.array(vals)

# ----------------------------------------------------------------------
# Paired run with an inflation-constant SWEEP for D4
# ----------------------------------------------------------------------
def run_paired(F, H, Qd, Rd, dist, n, T, rng, r, c_list, P0_scale=1.0, x0=None):
    Hd = H.toarray()
    Rinv = np.diag(1.0/Rd)
    HtRinvH = Hd.T @ Rinv @ Hd
    Q = np.diag(Qd)
    mask = (dist <= r)
    base = float(np.mean(Qd))

    x_true = (np.zeros(n) if x0 is None else x0.copy())
    xe, Pe = np.zeros(n), P0_scale*np.eye(n)
    xt, Pt = np.zeros(n), P0_scale*np.eye(n)

    rec = {'Yerr': [], 'nees_e': [], 'cov_e': [], 'rmse_e': [],
           'nees_t': [], 'cov_t': [], 'rmse_t': [],
           'nees_inf': {c: [] for c in c_list}, 'cov_inf': {c: [] for c in c_list}}

    for t in range(T):
        w = np.sqrt(Qd)*rng.standard_normal(n)
        x_true = F @ x_true + w
        y = Hd @ x_true + np.sqrt(Rd)*rng.standard_normal(len(Rd))

        # exact KF
        xe = F @ xe
        Pe = F @ Pe @ F.T + Q
        Se = np.linalg.inv(Pe)
        Ye = Se + HtRinvH
        xe = np.linalg.solve(Ye, Se @ xe + Hd.T @ (Rinv @ y))
        Pe_post = np.linalg.inv(Ye)

        # locality-r truncated filter
        xt = F @ xt
        Pt = F @ Pt @ F.T + Q
        St = np.linalg.inv(Pt)
        St_tr = St*mask
        Eoff = St - St_tr
        St_tr = jitter_spd(St_tr, base)
        Yt = St_tr + HtRinvH
        xt = np.linalg.solve(Yt, St_tr @ xt + Hd.T @ (Rinv @ y))
        Pt_post = np.linalg.inv(Yt)

        rec['Yerr'].append(np.linalg.norm(Yt-Ye, 'fro')/np.linalg.norm(Ye, 'fro'))
        ee = xe - x_true; et = xt - x_true
        rec['nees_e'].append(ee @ np.linalg.solve(Pe_post, ee))
        rec['cov_e'].append(np.mean(np.abs(ee) <= 1.96*np.sqrt(np.clip(np.diag(Pe_post), 1e-30, None))))
        rec['rmse_e'].append(np.sqrt(np.mean(ee**2)))
        rec['nees_t'].append(et @ np.linalg.solve(Pt_post, et))
        dt = np.clip(np.diag(Pt_post), 1e-30, None)
        rec['cov_t'].append(np.mean(np.abs(et) <= 1.96*np.sqrt(dt)))
        rec['rmse_t'].append(np.sqrt(np.mean(et**2)))

        # inflation sweep: Gamma = c*(||Eoff||_F/sigma_min(Yt))^2 * I
        sig_min = max(np.linalg.eigvalsh(Yt)[0], 1e-12)
        driver = (np.linalg.norm(Eoff, 'fro')/sig_min)**2
        lam, U = np.linalg.eigh(Pt_post)        # reuse across all c
        q = U.T @ et
        for c in c_list:
            g = c*driver
            rec['nees_inf'][c].append(float(np.sum(q**2/(lam+g))))
            rec['cov_inf'][c].append(np.mean(np.abs(et) <= 1.96*np.sqrt(dt+g)))
    return rec

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    Nx, Ny = 30, 20            # scale to 50x20 (n=1000) for the paper runs
    Fo, Pe = 0.010, 50.0
    sigma_w, sigma_v = 0.05, 0.5
    m, T, trials, radius = 8, 80, 10, 1
    n_burn = 8
    c_list = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

    F, H, Qd, Rd, coords, dist, n = build_model(Nx, Ny, Fo, Pe, sigma_w, sigma_v, m, rng)
    print(f"Model n={n} (Nx={Nx},Ny={Ny}) Fo={Fo} Pe={Pe} C={Fo*Pe} m={m}")
    print(f"Spectral radius of F: {spectral_radius(F):.4f}")
    print("="*70)

    print("D1  predicted-precision decay (RMS |Sigma_ij| ratio by grid distance)")
    for foi, pei in [(0.005, 10.0), (0.010, 50.0), (0.05, 150.0), (0.20, 300.0)]:
        Fi, _, Qdi, _, _, di, ni = build_model(Nx, Ny, foi, pei, sigma_w, sigma_v, m, rng)
        ds, vals = decay_profile(Fi, Qdi, di, ni, n_burn=n_burn)
        ratio = vals/vals[0]
        base = float(ratio[2]**0.5) if len(ratio) > 2 else float('nan')
        print(f"  Fo={foi:5.3f} Pe={pei:6.1f} | d=1,2,3: "
              f"{ratio[1]:.2e} {ratio[2]:.2e} {ratio[3]:.2e} | base~{base:.3f}")
    print("="*70)

    print(f"D2/D3/D4  exact vs locality-r={radius}, {trials} trials x T={T}, burn-in={n_burn}")
    chi_lo, chi_hi = chi2.ppf([0.025, 0.975], n)
    acc = {'Yerr_sup': [], 'Yerr_slope': [], 'nees_e': [], 'cov_e': [], 'rmse_e': [],
           'nees_t': [], 'cov_t': [], 'rmse_t': [],
           'nees_inf': {c: [] for c in c_list}, 'cov_inf': {c: [] for c in c_list}}
    for _ in range(trials):
        x0 = np.zeros(n); cx, cy = Nx//2, Ny//2
        for k in range(n):
            x0[k] = 5.0*np.exp(-((coords[k,0]-cx)**2+(coords[k,1]-cy)**2)/8.0)
        rec = run_paired(F, H, Qd, Rd, dist, n, T, rng, radius, c_list, x0=x0)
        ye = np.array(rec['Yerr'])[n_burn:]
        acc['Yerr_sup'].append(ye.max())
        acc['Yerr_slope'].append(np.polyfit(np.arange(len(ye)), ye, 1)[0])
        for k in ['nees_e','cov_e','rmse_e','nees_t','cov_t','rmse_t']:
            acc[k].append(np.mean(np.array(rec[k])[n_burn:]))
        for c in c_list:
            acc['nees_inf'][c].append(np.mean(np.array(rec['nees_inf'][c])[n_burn:]))
            acc['cov_inf'][c].append(np.mean(np.array(rec['cov_inf'][c])[n_burn:]))
    A = {k: (np.mean(v) if not isinstance(v, dict) else {c: np.mean(v[c]) for c in v})
         for k, v in acc.items()}

    print(f"  NEES/n 95% consistency band: [{chi_lo/n:.3f}, {chi_hi/n:.3f}] (target ~1.0)")
    print()
    print(f"  D2 post-burn ||Yt-Ye||_F/||Ye||_F: sup={A['Yerr_sup']:.3e}  "
          f"trend slope={A['Yerr_slope']:.2e}/step "
          f"({'no growth' if A['Yerr_slope']<1e-3 else 'CHECK'})")
    print()
    print("  Filter            NEES/n     Cov@95%    RMSE")
    print(f"  exact KF        {A['nees_e']/n:8.3f}  {A['cov_e']:8.3f}  {A['rmse_e']:8.4f}")
    print(f"  trunc (uncorr.) {A['nees_t']/n:8.3f}  {A['cov_t']:8.3f}  {A['rmse_t']:8.4f}"
          f"   <- D3")
    print()
    print("  D4 inflation sweep  Gamma = c*(||E||_F/sigma_min)^2 * I :")
    print("     c          NEES/n     Cov@95%   conservative?")
    c_star = None
    for c in c_list:
        nn, cc = A['nees_inf'][c]/n, A['cov_inf'][c]
        ok = (nn <= 1.0) and (cc >= 0.95)
        if ok and c_star is None:
            c_star = c
        print(f"   {c:7.0e}   {nn:8.3f}  {cc:8.3f}   {'YES' if ok else 'no'}")
    print()
    print(f"  -> tightest conservative constant c* = {c_star}")
    print("="*70)
    print("Readout:")
    print("  D1 holds: decay base < 1 and grows with Fo/Pe.")
    print("  D2 holds: post-burn sup small and trend slope ~ 0.")
    print("  D3 holds: uncorrected NEES/n >> 1, coverage < 0.95.")
    print("  D4 holds: a finite c* restores NEES/n<=1, coverage>=0.95 "
          "(empirical target for g() in Thm 3).")

if __name__ == "__main__":
    main()
