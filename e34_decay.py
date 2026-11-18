"""E34: the decay theorem (Supplement Theorem S1, summarised in Section 4.2) evaluated along the working filter.

For A_t = Y_{t-1} + F'Q^{-1}F, D_A = diag(A_t), X_t = I - D_A^{-1/2} A_t D_A^{-1/2}, L_t = Q^{-1} F D_A^{-1/2} and the weighted
norm ||Z||_a = max(max_i sum_j |Z_ij| e^{a d(i,j)}, same over columns), d = Manhattan grid distance, the theorem gives,
whenever omega_t(a) = ||X_t||_a < 1 and b >= 2 r_F (Q diagonal):
    ||D_t||_inf <= beta_t(a) e^{-a(b+1)},   beta_t(a) = ||L_t||_a^2 omega_t(a) / (1 - omega_t(a)),
with D_t = P^{-1}_{t|t-1} - B_b(P^{-1}_{t|t-1}). Everything on the right is sparse and data-free.
  dense : n = 240, 600 (dense reference filter): bound (minimised over a) against the exact ||D_t||_inf at t = 150,
          Pe in {10, 50, 90}, b = 2..4, m = 8, layout 0; also u_t, eta_t and the explicit bandwidth b0 of part (iii)
  large : n = 10^4 (fully sparse filter of E20, data-free): omega_t(a) and the bound at t = 300 for b = 1, 2, 3
usage: python e34_decay.py dense|large  ->  results/e34_decay_<mode>.json
"""
import sys, json, numpy as np, scipy.sparse as sp
sys.path.insert(0, ".")
from sppf_certified import lap1d, upwind1d
AL = np.round(np.arange(0.05, 2.51, 0.05), 2)

def wnorm_dense(Z, dist, a):
    W = np.abs(Z) * np.exp(a * dist); return max(W.sum(1).max(), W.sum(0).max())

def wnorm_sparse(Z, x, y, a):
    Z = Z.tocoo(); d = np.abs(x[Z.row] - x[Z.col]) + np.abs(y[Z.row] - y[Z.col]); w = np.abs(Z.data) * np.exp(a * d)
    n = Z.shape[0]; return max(np.bincount(Z.row, w, n).max(), np.bincount(Z.col, w, n).max())

def best_bound(wn, X, L, b):
    best = (np.inf, None, None)
    for a in AL:
        om = wn(X, a)
        if om >= 1: break
        v = wn(L, a) ** 2 * om / (1 - om) * np.exp(-a * (b + 1))
        if v < best[0]: best = (v, float(a), float(om))
    return best

def model(Nx, Ny, Pe, Fo=0.01):
    n = Nx * Ny; Ix, Iy = sp.identity(Nx), sp.identity(Ny)
    F = (sp.identity(n) + Fo * (sp.kron(Iy, lap1d(Nx)) + sp.kron(lap1d(Ny), Ix)) - Fo * Pe * sp.kron(Iy, upwind1d(Nx))).tocsr()
    return n, F, np.arange(n) % Nx, np.arange(n) // Nx

def dense(Nx, Ny, b, Pe, sw=0.05, m=8, T=150, sv=0.5, p0=0.01, seed=0):
    rng = np.random.default_rng(seed); n, Fs, x, y = model(Nx, Ny, Pe); F = Fs.toarray(); q = sw ** 2
    dist = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
    interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]; sens = rng.choice(interior, m, replace=False)
    J = np.zeros(n); J[sens] = 1 / sv ** 2; Y = np.eye(n) / p0; G = F.T @ F / q
    for t in range(1, T + 1):
        A = Y + G; Pp = F @ np.linalg.solve(Y, F.T) + q * np.eye(n); Pi = np.linalg.inv(Pp)
        D = np.where(dist > b, Pi, 0); S = Pi - D
        if t == T:
            dA = np.sqrt(np.diag(A)); X = np.eye(n) - A / dA[:, None] / dA[None, :]; L = (F / q) / dA[None, :]
            bound, a, om = best_bound(lambda Z, al: wnorm_dense(Z, dist, al), X, L, b)
            Dinf = np.abs(D).sum(1).max(); Pn = np.linalg.eigvalsh(Pp).max(); eta = 1 / np.linalg.eigvalsh(Pp / q).max()
            ev, V = np.linalg.eigh(Pp); Ph = V @ np.diag(np.sqrt(ev)) @ V.T; u = np.linalg.eigvalsh(Ph @ S @ Ph).max()
            return dict(n=n, Pe=Pe, b=b, Dinf=float(Dinf), bound=float(bound), ratio=float(bound / Dinf), alpha=a, omega=om,
                        omega0=float(wnorm_dense(X, dist, 0.0)), Pnorm=float(Pn), eta=float(eta), u=float(u),
                        eps_bound=float(Pn * bound), eps_true=float(Pn * Dinf))
        Y = S + np.diag(J)

def large(b, T=300, Nx=100, Ny=100, Pe=50.0, sw=0.05, sv=0.5, p0=0.01, seed=0):
    from scipy.sparse.linalg import splu
    from e20_large import ldl_posdef
    rng = np.random.default_rng(seed); n, F, x, y = model(Nx, Ny, Pe); q = sw ** 2; FT = F.T.tocsr(); FtF = (FT @ F).tocsr()
    interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]
    m = max(8, n // 100); sens = np.sort(rng.choice(interior, m, replace=False)); hrh = np.zeros(n); hrh[sens] = 1 / sv ** 2
    s = 2 * b + 3; col = (x % s) + s * (y % s); Pm = sp.csr_matrix((np.ones(n), (np.arange(n), col)), shape=(n, s * s)).toarray()
    offs = [(dx, dy) for dx in range(-b, b + 1) for dy in range(-b, b + 1) if abs(dx) + abs(dy) <= b]
    Y = sp.identity(n, format="csc") / p0
    for t in range(1, T + 1):
        A = (Y + FtF / q).tocsr()
        if t == T:
            dA = np.sqrt(A.diagonal()); Di = sp.diags(1 / dA)
            X = (sp.identity(n) - Di @ A @ Di).tocsr(); L = (F / q @ Di).tocsr()
            bound, a, om = best_bound(lambda Z, al: wnorm_sparse(Z, x, y, al), X, L, b)
            return dict(n=n, b=b, T=T, bound=float(bound), alpha=a, omega=om, omega0=float(wnorm_sparse(X, x, y, 0.0)))
        luZ = splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A"); W = Pm / q - F @ luZ.solve(np.asarray(FT @ Pm)) / q ** 2
        rows, cols, vals = [], [], []
        for dx, dy in offs:
            ok = (x + dx >= 0) & (x + dx < Nx) & (y + dy >= 0) & (y + dy < Ny); i = np.where(ok)[0]; j = i + dx + dy * Nx
            rows.append(j); cols.append(i); vals.append(W[j, col[i]])
        Sr = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        S = ((Sr + Sr.T) / 2).tocsc(); mu = 1e-8 * S.diagonal().mean(); shift = 0.0
        while not ldl_posdef(S - mu * sp.identity(n)):
            shift = max(2 * shift, mu); S = (S + shift * sp.identity(n)).tocsc()
        Y = (S + sp.diags(hrh)).tocsc()

if __name__ == "__main__" and (len(sys.argv) < 2 or sys.argv[1] in ("dense", "large")):
    mode = sys.argv[1] if len(sys.argv) > 1 else "dense"; out = []
    if mode == "dense":
        for (Nx, Ny) in ((20, 12), (30, 20)):
            for Pe in (10.0, 50.0, 90.0):
                for b in (2, 3, 4):
                    r = dense(Nx, Ny, b, Pe); out.append(r); print(r, flush=True)
    else:
        for b in (1, 2, 3):
            r = large(b); out.append(r); print(r, flush=True)
    json.dump(out, open(f"results/e34_decay_{mode}.json", "w"))

def selfconsistent_b0(Nx, Ny, Pe, bmax=30, sw=0.05, m=8, T=150, sv=0.5, p0=0.01, seed=0):
    """Smallest b for which part (iii) holds along the trajectory run with that same b (one-step check at t = T)."""
    rows = []
    for b in range(2, bmax + 1):
        rng = np.random.default_rng(seed); n, Fs, x, y = model(Nx, Ny, Pe); F = Fs.toarray(); q = sw ** 2
        dist = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
        interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]; sens = rng.choice(interior, m, replace=False)
        J = np.zeros(n); J[sens] = 1 / sv ** 2; Y = np.eye(n) / p0; G = F.T @ F / q
        for t in range(T):
            Pp = F @ np.linalg.solve(Y, F.T) + q * np.eye(n); Pi = np.linalg.inv(Pp)
            if t == T - 1: break
            Y = np.where(dist <= b, Pi, 0) + np.diag(J)
        A = Y + G; dA = np.sqrt(np.diag(A)); X = np.eye(n) - A / dA[:, None] / dA[None, :]; L = (F / q) / dA[None, :]
        bound, a, om = best_bound(lambda Z, al: wnorm_dense(Z, dist, al), X, L, b)
        Pn = np.linalg.eigvalsh(Pp).max(); eta = 1 / np.linalg.eigvalsh(Pp / q).max(); ceta = min(1, eta / (1 - eta))
        Dinf = np.abs(np.where(dist > b, Pi, 0)).sum(1).max()
        r = dict(n=n, Pe=Pe, b=b, eps_bound=float(Pn * bound), eps_true=float(Pn * Dinf), ceta=float(ceta), alpha=a, omega=om)
        rows.append(r); print(r, flush=True)
        if r["eps_bound"] < ceta: break
    return rows

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "b0":
    out = []
    for Pe in (10.0, 50.0, 90.0): out += selfconsistent_b0(30, 20, Pe)
    json.dump(out, open("results/e34_decay_b0.json", "w"))

def sharp(Nx, Ny, Pe, b, mm=5, T=150, checks=(110, 120, 130, 140, 150), sw=0.05, m=8, sv=0.5, p0=0.01, seed=0):
    """Supplement Theorem S1(iii): u_t <= 1 + lmax(-P^{1/2} D^(m) P^{1/2}) + ||P||_2 r_m at several steps, against the exact u_t;
    Theorem 2's condition sup_t u_t (1 - eta_t) < 1 evaluated with the bound and with the exact u_t."""
    import scipy.linalg as sl
    rng = np.random.default_rng(seed); n, Fs, x, y = model(Nx, Ny, Pe); F = Fs.toarray(); q = sw ** 2
    dist = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
    interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]; sens = rng.choice(interior, m, replace=False)
    J = np.zeros(n); J[sens] = 1 / sv ** 2; Y = np.eye(n) / p0; G = F.T @ F / q; rows = []
    for t in range(1, T + 1):
        Pp = F @ np.linalg.solve(Y, F.T) + q * np.eye(n); Pi = np.linalg.inv(Pp); S = np.where(dist <= b, Pi, 0)
        if t in checks:
            A = Y + G; dA = np.sqrt(np.diag(A)); X = np.eye(n) - A / dA[:, None] / dA[None, :]; L = (F / q) / dA[None, :]
            Sm = np.zeros((n, n)); Xk = np.eye(n)
            for k in range(mm): Xk = Xk @ X; Sm += Xk
            Dm = np.where(dist > b, -(L @ Sm @ L.T), 0)
            rem = np.inf
            for a in AL:
                om = wnorm_dense(X, dist, a)
                if om >= 1: break
                rem = min(rem, wnorm_dense(L, dist, a) ** 2 * om ** (mm + 1) / (1 - om) * np.exp(-a * (b + 1)))
            ev, V = np.linalg.eigh(Pp); Ph = V @ np.diag(np.sqrt(ev)) @ V.T
            u_b = 1 + max(0.0, np.linalg.eigvalsh(-Ph @ Dm @ Ph).max()) + ev.max() * rem
            u_e = sl.eigh(S, np.linalg.inv(Pp), eigvals_only=True).max(); eta = 1 / np.linalg.eigvalsh(Pp / q).max()
            rows.append((t, u_e, u_b, eta))
        Y = S + np.diag(J)
    ue = max(r[1] for r in rows); ub = max(r[2] for r in rows)
    rho_e = max(r[1] * (1 - r[3]) for r in rows); rho_b = max(r[2] * (1 - r[3]) for r in rows)
    return dict(n=n, Pe=Pe, b=b, m=mm, u_exact=float(ue), u_bound=float(ub), rho_exact=float(rho_e), rho_bound=float(rho_b),
                rel_gap=float((ub - ue) / (ue - 1)))

def cn(Nx, Ny, Pe, b, T=150, sw=0.05, m=8, sv=0.5, p0=0.01, seed=0, Fo=0.01):
    """Supplement Theorem S1(i) for Crank--Nicolson (F = A^{-1}B dense): beta = ||L||_a^2 / (1 - omega), all Neumann terms."""
    rng = np.random.default_rng(seed); n = Nx * Ny; q = sw ** 2; Ix, Iy = sp.identity(Nx), sp.identity(Ny)
    cen = sp.diags([-0.5 * np.ones(Nx - 1), 0.5 * np.ones(Nx - 1)], [-1, 1])
    Lm = (Fo * (sp.kron(Iy, lap1d(Nx)) + sp.kron(lap1d(Ny), Ix)) - Fo * Pe * sp.kron(Iy, cen)).toarray(); I = np.eye(n)
    Am = I - Lm / 2; F = np.linalg.solve(Am, I + Lm / 2)
    x = np.arange(n) % Nx; y = np.arange(n) // Nx; dist = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
    interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]; sens = rng.choice(interior, m, replace=False)
    J = np.zeros(n); J[sens] = 1 / sv ** 2; Y = np.eye(n) / p0; G = F.T @ F / q
    for t in range(T):
        Pp = F @ np.linalg.solve(Y, F.T) + q * I; Pi = np.linalg.inv(Pp)
        if t == T - 1: break
        Y = np.where(dist <= b, Pi, 0) + np.diag(J)
    A = Y + G; d = np.sqrt(np.diag(A)); X = I - A / d[:, None] / d[None, :]; L = (F / q) / d[None, :]
    Dinf = np.abs(np.where(dist > b, Pi, 0)).sum(1).max(); best = (np.inf, None, None)
    for a in AL:
        om = wnorm_dense(X, dist, a)
        if om >= 1: break
        v = wnorm_dense(L, dist, a) ** 2 / (1 - om) * np.exp(-a * (b + 1))
        if v < best[0]: best = (v, float(a), float(om))
    omA = float(wnorm_dense(I - Am / np.diag(Am)[:, None], dist, best[1]))
    return dict(n=n, Pe=Pe, b=b, Dinf=float(Dinf), bound=float(best[0]), ratio=float(best[0] / Dinf), alpha=best[1], omega=best[2], omega_A=omA)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("sharp", "cn"):
    out = []
    for Pe in (10.0, 50.0, 90.0):
        for b in ((2, 3, 4, 5) if sys.argv[1] == "sharp" else (2, 3, 4)):
            r = sharp(30, 20, Pe, b) if sys.argv[1] == "sharp" else cn(20, 12, Pe, b); out.append(r); print(r, flush=True)
    json.dump(out, open(f"results/e34_decay_{sys.argv[1]}.json", "w"))
