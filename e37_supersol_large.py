"""E37: a verified rank-p super-solution at n = 10^4 (Algorithm 2, second return; Corollary 2).

Runs the data-free working filter of E20/E25 (Woodbury + probing S_t) to t = T, freezes S, Y, iterates the rank-p map
of Corollary 3 to its fixed point Psi = g Y^{-1} + Z Z' (no safety factor inside the iteration), scales it by c and
tests L(c Psi) < c Psi strictly with the bordered matrix of Supplement Sec. S2 (Y_{t-1} = Y_t = Y,
(g_{t-1}, Z_{t-1}) = (g_t, Z_t) = (c g, sqrt(c) Z)): it holds iff the matrix has exactly p negative and no zero
eigenvalues. c = 1.001 is the scaling of Algorithm 2; c < 1 is a control that must fail. Also reports the
steady-state price bound of the fixed shape relative to the uncorrected filter, sqrt(c (g + |Z|_F^2 / tr Y^{-1})) - 1.
usage: python e37_supersol_large.py [b T p]  ->  results/e37_supersol_large_b<b>.json
"""
import sys, time, json, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh, LinearOperator
sys.path.insert(0, ".")
from sppf_certified import lap1d, upwind1d
from e20_large import ldl_posdef
from e23_verify_lowrank_large import inertia_negatives, bordered

def main(b=3, T=300, p=20, cs=(1.001, 1.0001, 0.999), Nx=100, Ny=100, Fo=0.01, Pe=50.0, sw=0.05, sv=0.5, p0=0.01, seed=0, maxit=600):
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
    t_run = time.perf_counter() - t_all
    luY = splu(Y.tocsc(), permc_spec="MMD_AT_PLUS_A")
    op = lambda f: LinearOperator((n, n), matvec=f, dtype=float); MiY = op(luY.solve)
    Rp = np.random.default_rng(7).choice([-1.0, 1.0], size=(n, 64)); trYi = float(np.mean(np.einsum("ij,ij->j", Rp, luY.solve(Rp))))
    t1 = time.perf_counter(); g, Z, g_old = 1.0, np.zeros((n, p)), np.inf
    for it in range(maxit):
        def Xmv(v, g=g, Z=Z):
            u = S @ v; w = g * (F @ luY.solve(FT @ u)) + q * u + F @ (Z @ (Z.T @ (FT @ u)))
            return S @ w + hrh * v
        lam, V = eigsh(op(Xmv), k=p + 1, M=Y, Minv=MiY, which="LA", tol=1e-10, ncv=max(2 * p + 20, 40))
        o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]; V = V / np.sqrt(np.einsum("ij,ij->j", V, Y @ V))
        g = float(lam[p]); Z = V[:, :p] * np.sqrt(np.maximum(lam[:p] - g, 0))
        if it % 25 == 0: print('it', it, g, time.perf_counter() - t1, flush=True)
        if abs(g - g_old) <= 1e-9 * g: break
        g_old = g
    out = dict(n=n, b=b, T=T, p=p, g=g, it=it + 1, t_fix=time.perf_counter() - t1, t_run=t_run, tests={})
    for c in cs:
        t2 = time.perf_counter()
        B = bordered(Y, c * g, np.sqrt(c) * Z, F, FT, S, q, hrh, Y, c * g, np.sqrt(c) * Z)
        neg, piv = inertia_negatives(B)
        out["tests"][str(c)] = dict(negatives=neg, min_abs_pivot=piv, holds=(neg == p), sec=time.perf_counter() - t2)
        print(c, out["tests"][str(c)], flush=True)
    out["price_bound_vs_unc"] = float(np.sqrt(cs[0] * (g + np.sum(Z ** 2) / trYi)) - 1)
    out["sec"] = time.perf_counter() - t_all
    print(out, flush=True)
    json.dump(out, open(f"results/e37_supersol_large_b{b}.json", "w"))
    return out

if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 3, int(a[1]) if len(a) > 1 else 300, int(a[2]) if len(a) > 2 else 20)
