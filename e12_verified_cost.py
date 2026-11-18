"""E12: cost of the estimate-then-verify certificate (cheap Lanczos + one sparse LDL' of the LMI block)
against the randomised certificate at eps=0.002, in the same harness as e5 (b=2, m=n/100)."""
import time, math, csv, sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from sppf_certified import grid_for, lap1d, upwind1d, banded_S, SpChol, lanczos_max

def ldl_posdef(B):
    """Symmetric-mode SuperLU (no pivoting) of a symmetric matrix; True iff all pivots > 0 (B > 0)."""
    # symmetric diagonal scaling D B D, D = diag(B)^{-1/2}: preserves the inertia (Sylvester) and keeps
    # blocks of very different size (e.g. Y/gamma with gamma large) from being lost in rounding
    dg = np.abs(B.diagonal()); dg[dg == 0] = 1.0; D = sp.diags(1.0 / np.sqrt(dg)); B = D @ B @ D
    try:
        lu = splu(B.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True, Equil=False))
    except RuntimeError:
        return False, 0
    if not np.all(lu.perm_r == lu.perm_c): return False, 0
    d = lu.U.diagonal(); return bool(np.all(d > 0)), lu.L.nnz + lu.U.nnz

rows = []
sizes = [int(a) for a in sys.argv[1:]] or [1000, 3000, 10000, 30000, 100000]
for nt in sizes:
    rng = np.random.default_rng(0); b = 2
    Nx, Ny = grid_for(nt); n = Nx * Ny; m = max(8, n // 100)
    F = (sp.identity(n) + 0.01 * (sp.kron(sp.identity(Ny), lap1d(Nx)) + sp.kron(lap1d(Ny), sp.identity(Nx)))
         - 0.5 * sp.kron(sp.identity(Ny), upwind1d(Nx))).tocsr()
    S = banded_S(Nx, Ny, b).tocsr(); sens = rng.choice(n, m, replace=False); hrh = np.zeros(n); hrh[sens] = 4.0
    Y = (S + sp.diags(hrh)).tocsc(); q = 0.0025; gp = 1.05
    t0 = time.perf_counter(); luY = splu(Y, permc_spec="MMD_AT_PLUS_A"); chS = SpChol(S); t_fac = time.perf_counter() - t0
    Gmv = lambda v: chS.LSt(gp * (F @ luY.solve(F.T @ chS.LS(v))) + q * chS.LS(v))
    # reference: eps = 0.002 randomised certificate (k from Prop. 1(ii), T_h = 1000, p = 1e-6)
    k_ref = math.ceil((math.log(1.648 * math.sqrt(n) * 1000 / 1e-6) / math.sqrt(0.002) + 1) / 2)
    t0 = time.perf_counter(); th_ref = lanczos_max(Gmv, n, k_ref, np.random.default_rng(1)); t_ref = time.perf_counter() - t0
    # estimate: cheap Lanczos
    k_est = 40
    t0 = time.perf_counter(); th = lanczos_max(Gmv, n, k_est, np.random.default_rng(2)); t_est = time.perf_counter() - t0
    # verify (Woodbury form, Q = qI):  lmax(Pc S) < g  <=>  Pc^{-1} > S/g  <=>
    #   B(g) = [[Z, F'Q^{-1}], [Q^{-1}F, Q^{-1} - S/g]] > 0,   Z = Y_{t-1}/gamma_{t-1} + F'Q^{-1}F
    Bblk = lambda g: sp.bmat([[Y / gp + (F.T @ F) / q, F.T / q], [F / q, sp.identity(n) / q - S / g]], format="csc")
    tau = 1e-3; tries = 0; t_ver = 0.0
    while True:
        g = th * (1 + tau * 2 ** tries)
        t0 = time.perf_counter(); ok, nnzB = ldl_posdef(Bblk(g)); t_ver += time.perf_counter() - t0; tries += 1
        if ok or tries > 8: break
    # resolution check: just below the reference value the test must fail
    t0 = time.perf_counter()
    ok_below, _ = ldl_posdef(Bblk(th_ref * (1 - 1e-3)))
    r = dict(n=n, m=m, t_factor_YS=t_fac, k_ref=k_ref, t_lanczos_ref=t_ref, gam_ref=th_ref,
             k_est=k_est, t_lanczos_est=t_est, gam_est=th, gam_verified=g, verify_tries=tries, t_verify=t_ver,
             nnz_factor_block=nnzB, verified_over_ref=g / th_ref, fails_below_ref=not ok_below)
    rows.append(r); print(r, flush=True)
with open("results/e12_verified_cost.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
