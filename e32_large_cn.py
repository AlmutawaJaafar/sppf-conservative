"""E32: fully sparse closed loop at n = 10^4 for the Crank--Nicolson model of Section 5.4 (no numerical diffusion).

F = A^{-1} B with A = I - L/2, B = I + L/2, L = Fo*Lap - Fo*Pe*Dc (central differences), so F is dense but is applied
by one sparse solve with A. Per step:
  1. S_t ~ B_b(P^{-1}), P^{-1} = I/q - F Z^{-1} F'/q^2, Z = Y_{t-1} + F'F/q (dense): the probe solves Z X = F' Pm are done by
     block preconditioned CG (preconditioner: sparse factorisation of Y_{t-1} + Fe'Fe/q, Fe = I + L the explicit
     propagator), relative residual 1e-10; colouring probing as in E20. Deterministic and data-independent.
  2. Y_t = S_t + H'R^{-1}H.
  3. scalar certificate: warm-started Lanczos estimate of lmax(P^c S_t), P^c = gamma F Y_{t-1}^{-1} F' + qI, accepted only
     after a sparse inertia test. Lemma 2 needs F'F; with F = A^{-1}B we use instead the equivalent condition
     g A S^{-1} A' >= gamma B Y_{t-1}^{-1} B' + q A A', i.e. by the Haynsworth inertia formula
         K = [[-q A A', B, A], [B', Y_{t-1}/gamma, 0], [A', 0, -S/g]]
     has exactly n negative eigenvalues and no zero eigenvalue (sparse LDL', order 3n).
  4. low-rank certificates p = 5, 20: leading eigenpairs of the pencil by ARPACK (safety factor 1 + 1e-3), as in E20.
  5. Monte Carlo: N trajectories; coverage of the domain mean, the far node and the cross-validated direction.
usage: python e32_large_cn.py Nx Ny b T N [ranks] [Pe]   ->  results/e32_cn_n<n>_b<b>_Pe<Pe>.json
       python e32_large_cn.py 20 12 3 20 50 check         (compares S_t with the exact B_b(P^{-1}))
"""
import sys, time, json, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh, LinearOperator
sys.path.insert(0, ".")
from sppf_certified import lap1d, SpChol, lanczos_max
from e20_large import ldl_posdef

def central1d(N):
    return sp.diags([-0.5 * np.ones(N - 1), 0.5 * np.ones(N - 1)], [-1, 1])

def inertia_negatives(B):
    dg = np.abs(B.diagonal()); dg[dg == 0] = 1.0; D = sp.diags(1.0 / np.sqrt(dg)); B = (D @ B @ D).tocsc()
    try:
        lu = splu(B, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True, Equil=False))
    except RuntimeError:
        return None
    if not np.all(lu.perm_r == lu.perm_c): return None
    d = lu.U.diagonal()
    if np.min(np.abs(d)) == 0: return None
    return int((d < 0).sum())

def block_pcg(Aop, Minv, Bm, tol=1e-10, maxit=500):
    X = np.zeros_like(Bm); R = Bm.copy(); Zr = Minv(R); P = Zr.copy(); rz = np.einsum("ij,ij->j", R, Zr)
    nb = np.linalg.norm(Bm, axis=0); it = 0
    for it in range(1, maxit + 1):
        AP = Aop(P); alpha = rz / np.einsum("ij,ij->j", P, AP)
        X += P * alpha; R -= AP * alpha
        if np.all(np.linalg.norm(R, axis=0) <= tol * nb): break
        Zr = Minv(R); rz_new = np.einsum("ij,ij->j", R, Zr); P = Zr + P * (rz_new / rz); rz = rz_new
    return X, it

def main(Nx, Ny, b, T, N, ranks=(), Pe=50.0, check=False, Fo=0.01, sw=0.05, sv=0.5, p0=0.01, seed=0):
    rng = np.random.default_rng(seed); n = Nx * Ny; q = sw ** 2
    Ix, Iy = sp.identity(Nx), sp.identity(Ny)
    L = (Fo * (sp.kron(Iy, lap1d(Nx)) + sp.kron(lap1d(Ny), Ix)) - Fo * Pe * sp.kron(Iy, central1d(Nx))).tocsc()
    A = (sp.identity(n) - L / 2).tocsc(); Bp = (sp.identity(n) + L / 2).tocsr(); AT = A.T.tocsc(); BT = Bp.T.tocsr()
    luA, luAT = splu(A), splu(AT)
    Fm = lambda V: luA.solve(np.asarray(Bp @ V))                  # F V
    FTm = lambda V: BT @ luAT.solve(np.asarray(V))               # F' V
    Fe = (sp.identity(n) + L).tocsr(); FetFe = (Fe.T @ Fe).tocsr()
    x = np.arange(n) % Nx; y = np.arange(n) // Nx
    interior = np.where((x > 0) & (x < Nx - 1) & (y > 0) & (y < Ny - 1))[0]
    m = max(8, n // 100); sens = np.sort(rng.choice(interior, m, replace=False))
    hrh = np.zeros(n); hrh[sens] = 1 / sv ** 2; J = sp.diags(hrh)
    s = 2 * b + 3; col = (x % s) + s * (y % s); ncol = s * s
    Pm = sp.csr_matrix((np.ones(n), (np.arange(n), col)), shape=(n, ncol)).toarray()
    offs = [(dx, dy) for dx in range(-b, b + 1) for dy in range(-b, b + 1) if abs(dx) + abs(dy) <= b]
    dmin = np.min(np.abs(x[:, None] - x[sens][None, :]) + np.abs(y[:, None] - y[sens][None, :]), axis=1); far = int(np.argmax(dmin))
    X = np.sqrt(p0) * rng.standard_normal((n, N)); Xh = np.zeros((n, N))
    Y = sp.identity(n, format="csc") / p0; gam = 1.0; v_warm = None
    LR = {p: dict(g=p0, Z=np.zeros((n, p)), first=True) for p in ranks}
    if check:
        Ad = A.toarray(); Fd = np.linalg.solve(Ad, Bp.toarray())
    rec = []; t_all = time.perf_counter()
    for t in range(1, T + 1):
        t0 = time.perf_counter()
        luPre = splu((Y + FetFe / q).tocsc(), permc_spec="MMD_AT_PLUS_A")
        Zop = lambda V: Y @ V + FTm(Fm(V)) / q
        Xs, its = block_pcg(Zop, luPre.solve, FTm(Pm))
        W = Pm / q - Fm(Xs) / q ** 2
        rows, cols, vals = [], [], []
        for dx, dy in offs:
            ok = (x + dx >= 0) & (x + dx < Nx) & (y + dy >= 0) & (y + dy < Ny); i = np.where(ok)[0]; j = i + dx + dy * Nx
            rows.append(j); cols.append(i); vals.append(W[j, col[i]])
        Sraw = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        S = ((Sraw + Sraw.T) / 2).tocsc(); mu = 1e-8 * S.diagonal().mean(); shift = 0.0
        while not ldl_posdef(S - mu * sp.identity(n)):
            shift = max(2 * shift, mu); S = (S + shift * sp.identity(n)).tocsc()
        t_S = time.perf_counter() - t0
        if check:
            Pp = Fd @ np.linalg.solve(Y.toarray(), Fd.T) + q * np.eye(n); Pinv = np.linalg.inv(Pp)
            dist = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :]); Sex = Pinv * (dist <= b)
            err_S = np.linalg.norm(S.toarray() - Sex, 2) / np.linalg.norm(Sex, 2)
        t0 = time.perf_counter()
        luYp = splu(Y.tocsc(), permc_spec="MMD_AT_PLUS_A"); chS = SpChol(S)
        Gmv = lambda v: chS.LSt(gam * Fm(luYp.solve(FTm(chS.LS(v)))) + q * chS.LS(v))
        th, v_warm = lanczos_max(Gmv, n, 10, rng, v0=v_warm, return_vec=True)
        def K(g):
            return sp.bmat([[-q * (A @ AT), Bp, A], [BT, Y / gam, None], [AT, None, -S / g]], format="csc")
        tries = 0
        while True:
            g = max(th, 1.0) * (1 + 1e-3 * 2 ** tries); tries += 1
            if inertia_negatives(K(g)) == n: break
            if tries >= 14: raise RuntimeError(f"step {t}: certificate not verified")
        lowered = inertia_negatives(K(g / (1 + 1e-3 * 2 ** (tries - 1)) * (1 - 1e-3))) if t % 50 == 0 else None
        gam_new = max(1.0, g); t_cert = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ynew = (S + J).tocsc(); luY = splu(Ynew, permc_spec="MMD_AT_PLUS_A")
        for p, c in LR.items():
            gp_, Zp = (1.0, np.zeros((n, p))) if c["first"] else (c["g"], c["Z"])
            def Xmv(v, gp_=gp_, Zp=Zp):
                u = S @ v; ftu = FTm(u); w = gp_ * Fm(luYp.solve(ftu)) + q * u
                if Zp.shape[1]: w = w + Fm(Zp @ (Zp.T @ ftu))
                return S @ w + hrh * v
            lam, V = eigsh(LinearOperator((n, n), matvec=Xmv, dtype=float), k=p + 1, M=Ynew,
                           Minv=LinearOperator((n, n), matvec=luY.solve, dtype=float), which="LA", tol=1e-8, ncv=max(2 * p + 20, 40))
            o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]; V = V / np.sqrt(np.einsum("ij,ij->j", V, Ynew @ V))
            gnew = (1 + 1e-3) * lam[p]; c["Znew"] = V[:, :p] * np.sqrt(np.maximum((1 + 1e-3) * lam[:p] - gnew, 0)); c["gnew"] = gnew
        X = Fm(X) + sw * rng.standard_normal((n, N)); obs = X[sens] + sv * rng.standard_normal((m, N))
        Xp = Fm(Xh); innov = np.zeros((n, N)); innov[sens] = (obs - Xp[sens]) / sv ** 2
        Xh = Xp + luY.solve(innov); E = Xh - X; t_mc = time.perf_counter() - t0
        Y, gam = Ynew, gam_new
        for c in LR.values(): c["g"], c["Z"], c["first"] = c["gnew"], c["Znew"], False
        r = dict(t=t, gamma=gam, tries=tries, shift=shift, cg_its=its, t_S=t_S, t_cert=t_cert, t_rest=t_mc)
        if lowered is not None: r["lowered_negatives"] = lowered; r["lowered_rejected"] = (lowered != n)
        for p, c in LR.items(): r[f"g{p}"] = c["g"]
        if check: r["err_S"] = err_S
        if t % 10 == 0 or t == T:
            nees = np.mean(np.einsum("ij,ij->j", E, Ynew @ E)) / n
            a = np.ones(n) / n; va = a @ luY.solve(a); em = a @ E
            ef = np.zeros(n); ef[far] = 1.0; vf = luY.solve(ef)[far]
            ch = SpChol(Ynew); U = np.vstack([ch.LSt(E[:, k]) for k in range(N)]).T
            Ah, Bh = U[:, : N // 2], U[:, N // 2:]
            ev, Vv = np.linalg.eigh(Ah.T @ Ah); w = Ah @ Vv[:, -1]; w /= np.linalg.norm(w); zB = w @ Bh
            r.update(nees_unc=nees, nees_cert=nees / gam,
                     cov_mean_unc=np.mean(np.abs(em) <= 1.96 * np.sqrt(va)), cov_mean_cert=np.mean(np.abs(em) <= 1.96 * np.sqrt(gam * va)),
                     cov_far_unc=np.mean(np.abs(E[far]) <= 1.96 * np.sqrt(vf)), cov_far_cert=np.mean(np.abs(E[far]) <= 1.96 * np.sqrt(gam * vf)),
                     cov_cv_unc=np.mean(np.abs(zB) <= 1.96), cov_cv_cert=np.mean(np.abs(zB) <= 1.96 * np.sqrt(gam)), var_cv=np.mean(zB ** 2))
            for p, c in LR.items():
                g_, Z_ = c["g"], c["Z"]; aw = ch.LS(w)
                r[f"cov_cv_r{p}"] = np.mean(np.abs(zB) <= 1.96 * np.sqrt(g_ + np.sum((Z_.T @ aw) ** 2)))
                r[f"cov_mean_r{p}"] = np.mean(np.abs(em) <= 1.96 * np.sqrt(g_ * va + np.sum((Z_.T @ a) ** 2)))
                r[f"cov_far_r{p}"] = np.mean(np.abs(E[far]) <= 1.96 * np.sqrt(g_ * vf + np.sum(Z_[far] ** 2)))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if t == T:
            Rp = np.random.default_rng(7).choice([-1.0, 1.0], size=(n, 64)); trYi = np.mean(np.einsum("ij,ij->j", Rp, luY.solve(Rp)))
            r["price_scalar_vs_unc"] = np.sqrt(gam) - 1
            for p_, c in LR.items(): r[f"price_r{p_}_vs_unc"] = np.sqrt(c["g"] + np.sum(c["Z"] ** 2) / trYi) - 1
            print({k: v for k, v in r.items() if k.startswith("price")}, flush=True)
        rec.append(r)
    out = dict(n=n, m=m, b=b, T=T, N=N, Pe=Pe, ncol=ncol, sec=time.perf_counter() - t_all, rec=rec)
    if not check:
        json.dump(out, open(f"results/e32_cn_n{n}_b{b}_Pe{int(Pe)}.json", "w"))
    return out

if __name__ == "__main__":
    a = sys.argv[1:]
    chk = len(a) > 5 and a[5] == "check"
    rk = tuple(int(v) for v in a[5].split(",")) if len(a) > 5 and not chk else ()
    pe = float(a[6]) if len(a) > 6 else 50.0
    main(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), ranks=rk, Pe=pe, check=chk)
