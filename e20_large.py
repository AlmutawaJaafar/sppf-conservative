"""E20: end-to-end closed-loop run at large n with a fully sparse pipeline and Monte Carlo validation.

Model of Section 5.1 (Fo = 0.01, upwind advection, Dirichlet, Q = sw^2 I, point sensors, R = sv^2 I).
Per step, everything is sparse:
  1. prior precision by Woodbury,  P^{-1} = I/q - F Z^{-1} F'/q^2,  Z = Y_{t-1} + F'F/q  (one sparse factorisation)
  2. S_t ~ B_b(P^{-1}) by graph-coloring probing: colour c(x,y) = (x mod s) + s (y mod s), s = 2b+3; one block solve
     with Z for all colours; entry (j,i), dist(i,j) <= b, read from column c(i). Symmetrised. This is a deterministic,
     data-independent approximation of B_b(P^{-1}); Theorem 1 holds for any such S_t and the certificate absorbs the
     probing error. If the sparse LDL' of S_t - mu I fails, a shift is added (doubled until it succeeds).
  3. Y_t = S_t + H'R^{-1}H (sparse factorisation)
  4. certificate: warm-started Lanczos (k = 10) for lmax(P^c S_t), P^c = gamma_{t-1} F Y_{t-1}^{-1} F' + Q, accepted only
     after the sparse LDL' test of the Lemma 3(b) block at theta (1 + tau 2^j), tau = 1e-3 (diagonally scaled).
  5. Monte Carlo: N independent trajectories (true state, observations, estimate); the estimate does not depend on gamma.
Validation (no exact moments at this size): coverage of nominal 95% intervals for the domain mean, the node farthest
from the sensors, and a cross-validated worst direction (leading direction of the whitened errors of one half of the
trajectories, coverage evaluated on the other half); NEES/n = mean e' C^{-1} e / n.
usage: python e20_large.py Nx Ny b T N [check]   ('check' compares S_t with the exact B_b(P^{-1}), small n only)
"""
import sys, time, json, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
sys.path.insert(0, ".")
from sppf_certified import lap1d, upwind1d, SpChol, lanczos_max

def ldl_posdef(B):
    dg = np.abs(B.diagonal()); dg[dg == 0] = 1.0; D = sp.diags(1.0 / np.sqrt(dg)); B = D @ B @ D
    try:
        lu = splu(B.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True, Equil=False))
    except RuntimeError:
        return False
    return bool(np.all(lu.perm_r == lu.perm_c) and np.all(lu.U.diagonal() > 0))

def main(Nx, Ny, b, T, N, check=False, Fo=0.01, Pe=50.0, sw=0.05, sv=0.5, p0=0.01, seed=0, ranks=()):
    from scipy.sparse.linalg import eigsh, LinearOperator
    rng = np.random.default_rng(seed); n = Nx * Ny; q = sw ** 2
    Ix, Iy = sp.identity(Nx), sp.identity(Ny)
    F = (sp.identity(n) + Fo * (sp.kron(Iy, lap1d(Nx)) + sp.kron(lap1d(Ny), Ix)) - Fo * Pe * sp.kron(Iy, upwind1d(Nx))).tocsr()
    FT = F.T.tocsr(); FtF = (FT @ F).tocsr()
    x = np.arange(n) % Nx; y = np.arange(n) // Nx
    interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]
    m = max(8, n // 100); sens = np.sort(rng.choice(interior, m, replace=False))
    hrh = np.zeros(n); hrh[sens] = 1 / sv ** 2; J = sp.diags(hrh)
    s = 2 * b + 3; col = (x % s) + s * (y % s); ncol = s * s
    Pm = sp.csr_matrix((np.ones(n), (np.arange(n), col)), shape=(n, ncol)).toarray()   # probe vectors
    offs = [(dx, dy) for dx in range(-b, b + 1) for dy in range(-b, b + 1) if abs(dx) + abs(dy) <= b]
    # node farthest from the sensors (graph distance)
    dmin = np.min(np.abs(x[:, None] - x[sens][None, :]) + np.abs(y[:, None] - y[sens][None, :]), axis=1); far = int(np.argmax(dmin))
    # Monte Carlo state
    X = np.sqrt(p0) * rng.standard_normal((n, N)); Xh = np.zeros((n, N))
    Y = sp.identity(n, format="csc") / p0; gam = 1.0; v_warm = None
    LR = {p: dict(g=p0, Z=np.zeros((n, p)), first=True) for p in ranks}    # C_0 = p0 I = p0 Y_0^{-1} ... (Y_0 = I/p0)
    rec = []; t_all = time.perf_counter()
    for t in range(1, T + 1):
        t0 = time.perf_counter()
        Z = (Y + FtF / q).tocsc(); luZ = splu(Z, permc_spec="MMD_AT_PLUS_A")
        W = Pm / q - F @ luZ.solve(np.asarray(FT @ Pm)) / q ** 2                         # P^{-1} * probes
        rows, cols, vals = [], [], []
        for dx, dy in offs:
            ok = (x + dx >= 0) & (x + dx < Nx) & (y + dy >= 0) & (y + dy < Ny); i = np.where(ok)[0]; j = i + dx + dy * Nx
            rows.append(j); cols.append(i); vals.append(W[j, col[i]])
        Sraw = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        S = ((Sraw + Sraw.T) / 2).tocsc()
        mu = 1e-8 * S.diagonal().mean(); shift = 0.0
        while not ldl_posdef(S - mu * sp.identity(n)):
            shift = max(2 * shift, mu); S = (S + shift * sp.identity(n)).tocsc()
        t_S = time.perf_counter() - t0
        if check:
            Yd = Y.toarray(); Pp = F.toarray() @ np.linalg.solve(Yd, FT.toarray()) + q * np.eye(n); Pinv = np.linalg.inv(Pp)
            dist = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :]); Sex = Pinv * (dist <= b)
            err_S = np.linalg.norm(S.toarray() - Sex, 2) / np.linalg.norm(Sex, 2)
        # certificate (needs Y_{t-1}); then update
        t0 = time.perf_counter()
        luYp = splu(Y.tocsc(), permc_spec="MMD_AT_PLUS_A"); chS = SpChol(S)
        Gmv = lambda v: chS.LSt(gam * (F @ luYp.solve(FT @ chS.LS(v))) + q * chS.LS(v))
        th, v_warm = lanczos_max(Gmv, n, 10, rng, v0=v_warm, return_vec=True)
        Blk = lambda g: sp.bmat([[Y / gam + FtF / q, FT / q], [F / q, sp.identity(n) / q - S / g]], format="csc")
        tries = 0
        while True:
            g = th * (1 + 1e-3 * 2 ** tries); tries += 1
            if ldl_posdef(Blk(g)): break
            if tries >= 12: raise RuntimeError(f"step {t}: certificate not verified after {tries} attempts")
        gam_new = max(1.0, g); t_cert = time.perf_counter() - t0
        # filter update and Monte Carlo
        t0 = time.perf_counter()
        Ynew = (S + J).tocsc(); luY = splu(Ynew, permc_spec="MMD_AT_PLUS_A")
        # low-rank certificates (Corollary 3), computed sparsely: leading eigenpairs of the pencil (X_t, Y_t)
        t_lr = time.perf_counter()
        for p, c in LR.items():
            gp_, Zp = (1.0, np.zeros((n, p))) if c["first"] else (c["g"], c["Z"])     # C_0 = P_0 = Y_0^{-1}
            def Xmv(v, gp_=gp_, Zp=Zp):
                u = S @ v; w = gp_ * (F @ luYp.solve(FT @ u)) + q * u
                if Zp.shape[1]: w = w + F @ (Zp @ (Zp.T @ (FT @ u)))
                return S @ w + hrh * v
            Xop = LinearOperator((n, n), matvec=Xmv, dtype=float)
            Mi = LinearOperator((n, n), matvec=luY.solve, dtype=float)
            lam, V = eigsh(Xop, k=p + 1, M=Ynew, Minv=Mi, which="LA", tol=1e-8, ncv=max(2 * p + 20, 40))
            o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
            V = V / np.sqrt(np.einsum("ij,ij->j", V, Ynew @ V))                       # V' Y V = I
            gnew = (1 + 1e-3) * lam[p]
            c["Znew"] = V[:, :p] * np.sqrt(np.maximum((1 + 1e-3) * lam[:p] - gnew, 0)); c["gnew"] = gnew
        t_lr = time.perf_counter() - t_lr
        X = F @ X + sw * rng.standard_normal((n, N)); obs = X[sens] + sv * rng.standard_normal((m, N))
        Xp = F @ Xh; innov = np.zeros((n, N)); innov[sens] = (obs - Xp[sens]) / sv ** 2
        Xh = Xp + luY.solve(innov); E = Xh - X; t_mc = time.perf_counter() - t0
        Y, gam = Ynew, gam_new
        for c in LR.values(): c["g"], c["Z"], c["first"] = c["gnew"], c["Znew"], False
        r = dict(t=t, gamma=gam, tries=tries, shift=shift, t_S=t_S, t_cert=t_cert, t_mc=t_mc, t_lr=t_lr)
        for p, c in LR.items(): r[f"g{p}"] = c["g"]
        if check: r["err_S"] = err_S
        if t % 10 == 0 or t == T:
            nees = np.mean(np.einsum("ij,ij->j", E, Ynew @ E)) / n
            a = np.ones(n) / n; va = a @ luY.solve(a); em = a @ E
            ef = np.zeros(n); ef[far] = 1.0; vf = luY.solve(ef)[far]
            ch = SpChol(Ynew); U = np.vstack([ch.LSt(E[:, k]) for k in range(N)]).T     # whitened errors (uncorrected)
            A, B = U[:, : N // 2], U[:, N // 2:]
            G = A.T @ A; ev, V = np.linalg.eigh(G); w = A @ V[:, -1]; w /= np.linalg.norm(w)   # leading direction of half A
            zB = w @ B                                                                     # unit variance under C = Y^{-1}
            r.update(nees_unc=nees, nees_cert=nees / gam,
                     cov_mean_unc=np.mean(np.abs(em) <= 1.96 * np.sqrt(va)), cov_mean_cert=np.mean(np.abs(em) <= 1.96 * np.sqrt(gam * va)),
                     cov_far_unc=np.mean(np.abs(E[far]) <= 1.96 * np.sqrt(vf)), cov_far_cert=np.mean(np.abs(E[far]) <= 1.96 * np.sqrt(gam * vf)),
                     cov_cv_unc=np.mean(np.abs(zB) <= 1.96), cov_cv_cert=np.mean(np.abs(zB) <= 1.96 * np.sqrt(gam)),
                     var_cv=np.mean(zB ** 2))
            for p, c in LR.items():                     # coverage of the same functionals under C = g Y^{-1} + Z Z'
                g_, Z_ = c["g"], c["Z"]
                aw = ch.LS(w)                                                                # a with a'Y^{-1}a = 1, a'e = w'L'e
                r[f"cov_cv_r{p}"] = np.mean(np.abs(zB) <= 1.96 * np.sqrt(g_ + np.sum((Z_.T @ aw) ** 2)))
                r[f"cov_mean_r{p}"] = np.mean(np.abs(em) <= 1.96 * np.sqrt(g_ * va + np.sum((Z_.T @ a) ** 2)))
                r[f"cov_far_r{p}"] = np.mean(np.abs(E[far]) <= 1.96 * np.sqrt(g_ * vf + np.sum(Z_[far] ** 2)))

            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if t == T:                                       # price relative to the uncorrected sparse filter
            Rp = np.random.default_rng(7).choice([-1.0, 1.0], size=(n, 64)); trYi = np.mean(np.einsum("ij,ij->j", Rp, luY.solve(Rp)))
            r["price_scalar_vs_unc"] = np.sqrt(gam) - 1
            for p_, c in LR.items(): r[f"price_r{p_}_vs_unc"] = np.sqrt(c["g"] + np.sum(c["Z"] ** 2) / trYi) - 1
            print({k: v for k, v in r.items() if k.startswith("price")}, flush=True)
        rec.append(r)
    out = dict(n=n, m=m, b=b, T=T, N=N, ncol=ncol, sec=time.perf_counter() - t_all, rec=rec)
    tag = f"_r{'-'.join(map(str, ranks))}" if ranks else ""
    json.dump(out, open(f"results/e20_large_n{n}_b{b}{tag}.json", "w"))
    return out

if __name__ == "__main__":
    a = sys.argv[1:]
    rk = tuple(int(v) for v in a[5].split(",")) if len(a) > 5 and a[5] not in ("check",) else ()
    main(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), check=(len(a) > 5 and a[5] == "check"), ranks=rk)
