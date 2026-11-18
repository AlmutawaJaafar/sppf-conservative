"""E25: Algorithm 2 (offline design) run matrix-free at n = 10^4, compared with the closed-loop run of E20.

The working filter of E20 (Woodbury + probing S_t, same model, sensors and seed) is run without data to t = T; it is
data-independent, so S_t, Y_t coincide with those of the closed-loop run. From S = S_T, Y = Y_T, Yp = Y_{T-1}:
  * nu_(1) = lmax(F Yp^{-1} F' S), the growth rate of the basic certificate (pencil (A, S), A = S F Yp^{-1} F' S);
  * nu#_(1..21), eigenvalues of the pencil (A, Y) (Corollary 3; Suppl. Proposition S2);
  * the steady state g* of the sharp certificate, smallest fixed point of g -> lmax(g A + N, Y), N = q S^2 + H'R^{-1}H
    (bisection), and its price relative to the uncorrected filter, sqrt(g*) - 1;
  * optionally, the steady state of the rank-p map with S, Y frozen (iteration), and its price
    sqrt(g + |Z|_F^2 / tr Y^{-1}) - 1 (trace by 64 Rademacher probes, as in E20).
All eigenvalue problems are solved by ARPACK on matrix-free operators with sparse factorisations; nothing is dense.
usage: python e25_offline_large.py b T [p1,p2 [safe]]  ->  results/e25_offline_large_b<b>[_safe].json
(safe: the safety factor of the closed-loop low-rank certificate of E20, 1 + 1e-3, applied in the rank-p map)
"""
import sys, time, json, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh, LinearOperator
sys.path.insert(0, ".")
from sppf_certified import lap1d, upwind1d
from e20_large import ldl_posdef

def main(b=3, T=300, ranks=(), safe=1.0, Nx=100, Ny=100, Fo=0.01, Pe=50.0, sw=0.05, sv=0.5, p0=0.01, seed=0, maxit=400):
    t_all = time.perf_counter()
    rng = np.random.default_rng(seed); n = Nx * Ny; q = sw ** 2
    Ix, Iy = sp.identity(Nx), sp.identity(Ny)
    F = (sp.identity(n) + Fo * (sp.kron(Iy, lap1d(Nx)) + sp.kron(lap1d(Ny), Ix)) - Fo * Pe * sp.kron(Iy, upwind1d(Nx))).tocsr()
    FT = F.T.tocsr(); FtF = (FT @ F).tocsr()
    x = np.arange(n) % Nx; y = np.arange(n) // Nx
    interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]
    m = max(8, n // 100); sens = np.sort(rng.choice(interior, m, replace=False))
    hrh = np.zeros(n); hrh[sens] = 1 / sv ** 2; J = sp.diags(hrh)
    s = 2 * b + 3; col = (x % s) + s * (y % s); ncol = s * s
    Pm = sp.csr_matrix((np.ones(n), (np.arange(n), col)), shape=(n, ncol)).toarray()
    offs = [(dx, dy) for dx in range(-b, b + 1) for dy in range(-b, b + 1) if abs(dx) + abs(dy) <= b]
    Y = sp.identity(n, format="csc") / p0
    for t in range(1, T + 1):                                    # data-free working filter (identical to E20)
        luZ = splu((Y + FtF / q).tocsc(), permc_spec="MMD_AT_PLUS_A")
        W = Pm / q - F @ luZ.solve(np.asarray(FT @ Pm)) / q ** 2
        rows, cols, vals = [], [], []
        for dx, dy in offs:
            ok = (x + dx >= 0) & (x + dx < Nx) & (y + dy >= 0) & (y + dy < Ny); i = np.where(ok)[0]; j = i + dx + dy * Nx
            rows.append(j); cols.append(i); vals.append(W[j, col[i]])
        Sraw = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        S = ((Sraw + Sraw.T) / 2).tocsc(); mu = 1e-8 * S.diagonal().mean(); shift = 0.0
        while not ldl_posdef(S - mu * sp.identity(n)):
            shift = max(2 * shift, mu); S = (S + shift * sp.identity(n)).tocsc()
        Yp, Y = Y, (S + J).tocsc()
    t_run = time.perf_counter() - t_all; t0 = time.perf_counter()
    luYp, luY, luS = (splu(M.tocsc(), permc_spec="MMD_AT_PLUS_A") for M in (Yp, Y, S))
    Amv = lambda v: S @ (F @ luYp.solve(FT @ (S @ v)))
    Nmv = lambda v: q * (S @ (S @ v)) + hrh * v
    op = lambda f: LinearOperator((n, n), matvec=f, dtype=float)
    MiY, MiS = op(luY.solve), op(luS.solve)
    top = lambda f, k, M, Mi: np.sort(eigsh(op(f), k=k, M=M, Minv=Mi, which="LA", tol=1e-9, ncv=max(2 * k + 20, 40))[0])[::-1]
    nu = float(top(Amv, 1, S, MiS)[0])
    nus = top(Amv, 21, Y, MiY)
    c = float(top(Nmv, 1, Y, MiY)[0])
    out = dict(n=n, b=b, T=T, safe=safe, nu_1=nu, nus=nus.tolist(), c=c, k_above=int((nus >= 1).sum()))
    if nus[0] < 1:
        lo, hi = 0.0, max(1.0, c / (1 - nus[0])) * 1.0001
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if top(lambda v: mid * Amv(v) + Nmv(v), 1, Y, MiY)[0] > mid: lo = mid
            else: hi = mid
        out["g_sharp_pred"] = float(hi); out["price_sharp_pred_vs_unc"] = float(np.sqrt(hi) - 1)
    else:
        out["g_sharp_pred"] = None; out["price_sharp_pred_vs_unc"] = None
    out["t_offline_eig"] = time.perf_counter() - t0
    Rp = np.random.default_rng(7).choice([-1.0, 1.0], size=(n, 64)); trYi = float(np.mean(np.einsum("ij,ij->j", Rp, luY.solve(Rp))))
    for p in ranks:                                              # steady state of the rank-p map, S and Y frozen
        t1 = time.perf_counter(); g, Z, g_old = 1.0, np.zeros((n, p)), np.inf
        for it in range(maxit):
            def Xmv(v, g=g, Z=Z):
                u = S @ v; w = g * (F @ luY.solve(FT @ u)) + q * u + F @ (Z @ (Z.T @ (FT @ u)))
                return S @ w + hrh * v
            lam, V = eigsh(op(Xmv), k=p + 1, M=Y, Minv=MiY, which="LA", tol=1e-9, ncv=max(2 * p + 20, 40))
            o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]; V = V / np.sqrt(np.einsum("ij,ij->j", V, Y @ V))
            g = float(safe * lam[p]); Z = V[:, :p] * np.sqrt(np.maximum(safe * lam[:p] - g, 0))
            if abs(g - g_old) <= 1e-7 * g: break
            g_old = g
        out[f"g_r{p}_pred"] = g; out[f"it_r{p}"] = it + 1; out[f"t_r{p}"] = time.perf_counter() - t1
        out[f"price_r{p}_pred_vs_unc"] = float(np.sqrt(g + np.sum(Z ** 2) / trYi) - 1)
        print(out, flush=True)
    out["t_run"] = t_run; out["sec"] = time.perf_counter() - t_all
    print(out, flush=True)
    json.dump(out, open(f"results/e25_offline_large_b{b}" + ("_safe" if safe != 1.0 else "") + ".json", "w"))
    return out

if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), tuple(int(v) for v in a[2].split(",")) if len(a) > 2 else (), float(a[3]) if len(a) > 3 else 1.0)
