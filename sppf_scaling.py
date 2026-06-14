"""
sppf_scaling.py  --  scalability / end-to-end timing of the sparse-precision filter.

Purpose: answer the "scale mismatch" critique (motivation cites n=1e3-1e5 but the
calibration experiments are n<=4000) by timing the *cost* of the per-step linear
algebra the calibrated sparse filter actually performs, on the true band-b grid
sparsity pattern, for n up to 1e5 -- and contrasting with the dense Kalman filter,
which is O(n^3) time / O(n^2) memory and cannot reach that range.

The per-step cost of the sparse-precision filter is dominated by:
  (1) factorizing the band-b information matrix Y = Sigma^(b) + H^T R^-1 H,
  (2) one solve for the mean,
  (3) recovering the band-b covariance -- here represented by k Hutchinson
      probe-solves for diag(Y^-1) (a selected-inverse would be the same O(nnz(L))).
We measure wall-clock time/step and factor memory (nnz of L) vs n, and fit the
empirical scaling exponent. Numerical *values* are irrelevant to cost; the band-b
grid sparsity pattern (which determines fill-in) is what is timed, so we build a
representative diagonally-dominant SPD band-b matrix on each grid.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
import time, os, csv, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "results"; os.makedirs(OUTDIR, exist_ok=True)
RNG = np.random.default_rng(0)

def factor_grids(n_target):
    """near-square 2D grid with Nx*Ny ~ n_target."""
    Ny = max(2, int(round(math.sqrt(n_target/2))))
    Nx = max(2, int(round(n_target/Ny)))
    return Nx, Ny

def build_band_precision(Nx, Ny, b, m):
    """Representative SPD band-b information matrix on the Nx x Ny grid:
    off-diagonal -w for grid (Manhattan) distance <= b, diagonal made dominant,
    plus a measurement boost on m random sensor nodes (H^T R^-1 H)."""
    n = Nx*Ny
    xs, ys = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    xs = xs.ravel(); ys = ys.ravel()
    rows, cols, vals = [], [], []
    # neighbor offsets within Manhattan radius b
    offs = [(dx,dy) for dx in range(-b,b+1) for dy in range(-b,b+1)
            if 0 < abs(dx)+abs(dy) <= b]
    idx = lambda ix,iy: ix*Ny+iy
    for k in range(n):
        ix, iy = xs[k], ys[k]
        for dx,dy in offs:
            jx, jy = ix+dx, iy+dy
            if 0 <= jx < Nx and 0 <= jy < Ny:
                rows.append(k); cols.append(idx(jx,jy)); vals.append(-0.15)
    A = sp.csr_matrix((vals,(rows,cols)), shape=(n,n))
    A = 0.5*(A + A.T)                                   # symmetric
    deg = np.abs(A).sum(axis=1).A.ravel()
    diag = deg + 1.0                                    # strict diagonal dominance -> SPD
    sensors = RNG.choice(n, size=min(m,n), replace=False)
    meas = np.zeros(n); meas[sensors] += 4.0            # H^T R^-1 H (diagonal, m sensors)
    Y = A + sp.diags(diag + meas)
    return Y.tocsc(), n

def time_sparse_step(Y, k_probe=8, reps=2):
    n = Y.shape[0]
    best = np.inf; nnzL = np.nan
    for _ in range(reps):
        t0 = time.perf_counter()
        lu = splu(Y, permc_spec="MMD_AT_PLUS_A")        # factorization (dominant cost)
        rhs = RNG.standard_normal(n)
        _ = lu.solve(rhs)                               # mean solve
        for _ in range(k_probe):                        # covariance diag via Hutchinson probes
            z = RNG.choice([-1.0, 1.0], size=n)
            lu.solve(z)
        dt = time.perf_counter() - t0
        best = min(best, dt)
        nnzL = lu.L.nnz + lu.U.nnz
    return best, nnzL

def time_dense_step(n, reps=2):
    best = np.inf
    for _ in range(reps):
        M = np.eye(n) + 0.01*RNG.standard_normal((n,n)); M = M@M.T  # SPD
        t0 = time.perf_counter()
        Minv = np.linalg.inv(M)                         # dense KF covariance update
        _ = Minv @ RNG.standard_normal(n)
        best = min(best, time.perf_counter()-t0)
    mem = n*n*8
    return best, mem

def main():
    b = 2
    n_targets = [1000, 3000, 10000, 30000, 100000]
    dense_cap = 4000
    rows = []
    print(f"=== scaling experiment (band b={b}) ===")
    for nt in n_targets:
        Nx, Ny = factor_grids(nt)
        m = max(8, (Nx*Ny)//100)                        # ~1% sensors
        Y, n = build_band_precision(Nx, Ny, b, m)
        ts, nnzL = time_sparse_step(Y)
        mem_sparse = nnzL*8
        if n <= dense_cap:
            td, mem_dense = time_dense_step(n)
        else:
            td, mem_dense = np.nan, n*n*8.0             # dense infeasible; memory is analytic
        rows.append((n, Nx, Ny, ts, mem_sparse, td, mem_dense))
        print(f"  n={n:6d} ({Nx}x{Ny}) m={m:4d} | sparse {ts*1e3:8.2f} ms, "
              f"{mem_sparse/1e6:8.2f} MB (factor) | dense "
              f"{'%.1f ms'%(td*1e3) if td==td else '   --  '}, "
              f"{mem_dense/1e6:9.1f} MB")
    R = np.array([(r[0], r[3], r[4], r[5], r[6]) for r in rows], dtype=float)
    # fit sparse time exponent
    sl, ic = np.polyfit(np.log(R[:,0]), np.log(R[:,1]), 1)
    slm, icm = np.polyfit(np.log(R[:,0]), np.log(R[:,2]), 1)
    print(f"\n[scaling] sparse time ~ n^{sl:.2f} ; sparse factor memory ~ n^{slm:.2f}")
    print(f"[scaling] dense is n^3 time / n^2 memory (shown where feasible)")

    # time-vs-n
    fig, ax = plt.subplots(figsize=(6,4.2))
    ax.loglog(R[:,0], R[:,1]*1e3, "o-", label=fr"sparse-precision (fit $n^{{{sl:.2f}}}$)")
    dmask = R[:,3]==R[:,3]
    ax.loglog(R[dmask,0], R[dmask,3]*1e3, "s--", label="dense KF (measured)")
    nn = np.array([R[0,0], R[-1,0]])
    ax.loglog(nn, R[0,3]*1e3*(nn/R[0,0])**3, ":", color="gray", label=r"$n^3$ ref")
    ax.set_xlabel("state dimension n"); ax.set_ylabel("per-step wall time (ms)")
    ax.set_title(f"Per-step cost vs n (band b={b})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_scaling_time.png"), dpi=120); plt.close(fig)
    # memory-vs-n
    fig, ax = plt.subplots(figsize=(6,4.2))
    ax.loglog(R[:,0], R[:,2]/1e6, "o-", label=fr"sparse factor (fit $n^{{{slm:.2f}}}$)")
    ax.loglog(R[:,0], R[:,4]/1e6, "s--", label=r"dense KF ($n^2$)")
    ax.set_xlabel("state dimension n"); ax.set_ylabel("memory (MB)")
    ax.set_title("Memory vs n"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR,"fig_scaling_mem.png"), dpi=120); plt.close(fig)

    with open(os.path.join(OUTDIR,"scaling_log.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["n","Nx","Ny","t_sparse_s","mem_sparse_B","t_dense_s","mem_dense_B"])
        for r in rows: w.writerow(r)
    print(f"[scaling] wrote scaling_log.csv, fig_scaling_time.png, fig_scaling_mem.png")
    return R, sl, slm

if __name__ == "__main__":
    main()
