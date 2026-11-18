"""E35: exact check of the n=10^4 runs of E20 (Table 3): the dense Kalman filter and the exact error covariance of
the sparse filter, propagated alongside the fully sparse pipeline of e20_large.py (same seed, sensors, certificates).

Exact moments: Sigma_t = A_t (F Sigma F' + Q) A_t' + K_t R K_t',  K_t = Y_t^{-1} H' R^{-1},  A_t = I - K_t H  (dense, 0.8 GB),
and the Kalman covariance P^KF_t (dense). For each certificate C (uncorrected Y^{-1}, scalar gamma Y^{-1}, rank-p
g Y^{-1} + Z Z') we report the exact worst-direction ratio  lambda_max(Sigma_t, C)  (generalised Lanczos; <= 1 means
every linear functional is covered), the exact coverage of the domain mean and of the farthest node, and the price
relative to the Kalman filter  (tr C / tr P^KF)^{1/2} - 1  with tr Y^{-1} computed exactly.
With 'sharp', the scalar certificate is the sharp one, gamma_t >= lmax(Y_t^{-1}(S_t P^c S_t + J)) (Theorem 1(a)), estimated by
Lanczos on the pencil and accepted only after the sparse LDL' test of the sharp form of Lemma 3(a),
  [[Y_{t-1}/gamma_{t-1}, F'S_t], [S_t F, g Y_t - J - q S_t^2]] >= 0  (order 2n; up to 12 attempts g = theta (1 + 1e-3 2^j)).
usage: python e35_exact_large.py Nx Ny b T N [ranks] [sharp]   (resumes from results/e35_state_*.pkl after an interruption; dense moments in float32, ~1.5 GB RAM; ~20 s per step at n=10^4 plus ~2 min per checkpoint, every 50 steps)
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

def main(Nx, Ny, b, T, N, check=False, every=50, Fo=0.01, Pe=50.0, sw=0.05, sv=0.5, p0=0.01, seed=0, ranks=(), sharp=False):
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
    from scipy.stats import norm
    f32 = np.float32                                      # dense moments in single precision (0.4 GB each)
    Sig = (p0 * np.eye(n)).astype(f32); PKF = (p0 * np.eye(n)).astype(f32); F32 = F.astype(f32)
    def propagate(Mat):                                   # F Mat F' + q I, Mat symmetric
        A_ = np.asarray(F32 @ Mat); Mat = np.asarray(F32 @ A_.T); del A_
        Mat[np.diag_indices(n)] += f32(q); return Mat
    def gen_lmax(Sg, Cmv, Cinv):                         # largest eigenvalue of the pencil (Sigma, C)
        So = LinearOperator((n, n), matvec=lambda v: (Sg @ v.astype(f32)).astype(float), dtype=float)
        lam = eigsh(So, k=1, M=LinearOperator((n, n), matvec=Cmv, dtype=float),
                    Minv=LinearOperator((n, n), matvec=Cinv, dtype=float), which="LA", tol=1e-4, ncv=40, return_eigenvectors=False)
        return float(lam[0])
    def ecols(idx):                                      # columns idx of the identity, without forming I
        idx = np.asarray(idx); Z = np.zeros((n, len(idx))); Z[idx, np.arange(len(idx))] = 1.0; return Z
    cov = lambda rho: float(2 * norm.cdf(1.96 / np.sqrt(rho)) - 1)
    rec = []; t_all = time.perf_counter(); t_start = 1
    import pickle, os
    ckf = f"results/e35_state_n{n}_b{b}{'_sharp' if sharp else ''}.pkl"                # resume after an interruption (state saved every checkpoint)
    if os.path.exists(ckf):
        stt = pickle.load(open(ckf, "rb"))
        Sig, PKF = np.load(ckf + ".sig.npy"), np.load(ckf + ".pkf.npy")
        Y, gam, v_warm, LR, X, Xh, rec, t_start = stt["Y"], stt["gam"], stt["v_warm"], stt["LR"], stt["X"], stt["Xh"], stt["rec"], stt["t"] + 1
        rng.bit_generator.state = stt["rng"]; print(f"resumed at t={t_start}", flush=True)
    for t in range(t_start, T + 1):
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
            Sd = S.toarray(); Yd1 = Sd + np.diag(hrh); Pc = gam * F.toarray() @ np.linalg.solve(Yd, FT.toarray()) + q * np.eye(n)
            g_sharp_dense = float(np.max(np.linalg.eigvals(np.linalg.solve(Yd1, Sd @ Pc @ Sd + np.diag(hrh))).real))
        # certificate (needs Y_{t-1}); then update
        t0 = time.perf_counter()
        luYp = splu(Y.tocsc(), permc_spec="MMD_AT_PLUS_A"); chS = SpChol(S)
        if sharp:                                        # sharp certificate: pencil (S P^c S + J, Y_t), sharp Lemma 3(a)
            Yn_ = (S + J).tocsc(); luYn_ = splu(Yn_, permc_spec="MMD_AT_PLUS_A")
            Xs = LinearOperator((n, n), matvec=lambda v: S @ (gam * (F @ luYp.solve(FT @ (S @ v))) + q * (S @ v)) + hrh * v, dtype=float)
            lam_, V_ = eigsh(Xs, k=1, M=Yn_, Minv=LinearOperator((n, n), matvec=luYn_.solve, dtype=float), which="LA",
                             tol=1e-6, ncv=40, v0=v_warm)
            th, v_warm = float(lam_[0]), V_[:, 0]
            S2 = (S @ S).tocsc(); SF = (S @ F).tocsc()
            Blk = lambda g: sp.bmat([[Y / gam, SF.T], [SF, g * Yn_ - J - q * S2]], format="csc")
        else:
            Gmv = lambda v: chS.LSt(gam * (F @ luYp.solve(FT @ chS.LS(v))) + q * chS.LS(v))
            th, v_warm = lanczos_max(Gmv, n, 10, rng, v0=v_warm, return_vec=True)
            Blk = lambda g: sp.bmat([[Y / gam + FtF / q, FT / q], [F / q, sp.identity(n) / q - S / g]], format="csc")
        tries = 0
        while True:
            g = th * (1 + 1e-3 * 2 ** tries); tries += 1
            if ldl_posdef(Blk(g)): break
            if tries >= 12: raise RuntimeError(f"step {t}: certificate not verified after {tries} attempts")
        gam_new = g if sharp else max(1.0, g); t_cert = time.perf_counter() - t0
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
        t0 = time.perf_counter()
        Sig = propagate(Sig); PKF = propagate(PKF)
        Kg = luY.solve(ecols(sens)) / sv ** 2                                          # sparse-filter gain (n x m)
        M1 = Sig[:, sens].astype(float); G = Sig[np.ix_(sens, sens)].astype(float) + sv ** 2 * np.eye(m)
        a_ = (0.5 * Kg @ G - M1).astype(f32); K32 = Kg.astype(f32)
        Sig += a_ @ K32.T; Sig += K32 @ a_.T; del a_, K32, M1                                   # A Sig A' + K R K'
        HP = PKF[sens].astype(float); Sinn = HP[:, sens] + sv ** 2 * np.eye(m)
        PKF -= (HP.T @ np.linalg.solve(Sinn, HP)).astype(f32); del HP                          # Kalman update
        t_dense = time.perf_counter() - t0
        Y, gam = Ynew, gam_new
        for c in LR.values(): c["g"], c["Z"], c["first"] = c["gnew"], c["Znew"], False
        r = dict(t=t, gamma=gam, tries=tries, shift=shift, t_S=t_S, t_cert=t_cert, t_mc=t_mc, t_lr=t_lr)
        for p, c in LR.items(): r[f"g{p}"] = c["g"]
        if check: r["err_S"] = err_S; r["g_sharp_dense"] = g_sharp_dense; r["th"] = th
        if t % every == 0 or t == T:
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

            # exact moments
            t0 = time.perf_counter(); r["t_dense"] = t_dense
            Yinv_mv = luY.solve; Ymv = lambda v: Ynew @ v
            rho_u = gen_lmax(Sig, Yinv_mv, Ymv); r["t_eig1"] = time.perf_counter() - t0
            r.update(ex_ratio_unc=rho_u, ex_ratio_cert=rho_u / gam, ex_wcov_unc=cov(rho_u), ex_wcov_cert=cov(max(rho_u / gam, 1e-12)))
            vm = float(a.astype(f32) @ (Sig @ a.astype(f32))); vF = float(Sig[far, far])
            r.update(ex_cov_mean_unc=cov(vm / va), ex_cov_mean_cert=cov(vm / (gam * va)),
                     ex_cov_far_unc=cov(vF / vf), ex_cov_far_cert=cov(vF / (gam * vf)),
                     ex_subopt=float(np.sqrt(np.trace(Sig, dtype=float) / np.trace(PKF, dtype=float)) - 1),
                     ex_cov_mean_kf=cov(vm / float(a.astype(f32) @ (PKF @ a.astype(f32)))))
            for p, c in LR.items():
                g_, Z_ = c["g"], c["Z"]
                YZ = Ynew @ Z_; Wm = np.linalg.inv(g_ * np.eye(p) + Z_.T @ YZ)
                Cmv = lambda v, g_=g_, Z_=Z_: g_ * luY.solve(v) + Z_ @ (Z_.T @ v)
                Cinv = lambda v, g_=g_, YZ=YZ, Wm=Wm: (Ynew @ v - YZ @ (Wm @ (YZ.T @ v))) / g_
                rho = gen_lmax(Sig, Cmv, Cinv)
                r[f"ex_ratio_r{p}"] = rho; r[f"ex_wcov_r{p}"] = cov(rho)
                r[f"ex_cov_mean_r{p}"] = cov(vm / (g_ * va + np.sum((Z_.T @ a) ** 2)))
                r[f"ex_cov_far_r{p}"] = cov(vF / (g_ * vf + np.sum(Z_[far] ** 2)))
            r["t_eigs"] = time.perf_counter() - t0
            if t % 50 == 0 or t == T:                    # exact traces, prices against the Kalman filter
                trYi = sum(np.trace(luY.solve(ecols(np.arange(k, min(k + 500, n))))[k:k + 500]) for k in range(0, n, 500))
                trK = float(np.trace(PKF, dtype=float)); r.update(trYinv=float(trYi), trPKF=trK, trSig=float(np.trace(Sig, dtype=float)))
                r["ex_price_unc"] = float(np.sqrt(trYi / trK) - 1); r["ex_price_cert"] = float(np.sqrt(gam * trYi / trK) - 1)
                for p, c in LR.items(): r[f"ex_price_r{p}"] = float(np.sqrt((c["g"] * trYi + np.sum(c["Z"] ** 2)) / trK) - 1)
            r["t_exact"] = time.perf_counter() - t0
            json.dump(dict(n=n, m=m, b=b, T=T, N=N, partial=True, rec=rec + [r]), open(f"results/e35_exact_n{n}_b{b}{'_sharp' if sharp else ''}_partial.json", "w"), default=float)
            np.save(ckf + ".sig.npy", Sig); np.save(ckf + ".pkf.npy", PKF)
            pickle.dump(dict(Y=Y, gam=gam, v_warm=v_warm, LR=LR, X=X, Xh=Xh, rec=rec + [r], t=t, rng=rng.bit_generator.state), open(ckf + ".tmp", "wb"))
            os.replace(ckf + ".tmp", ckf)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if t == T:                                       # price relative to the uncorrected sparse filter
            Rp = np.random.default_rng(7).choice([-1.0, 1.0], size=(n, 64)); trYi = np.mean(np.einsum("ij,ij->j", Rp, luY.solve(Rp)))
            r["price_scalar_vs_unc"] = np.sqrt(gam) - 1
            for p_, c in LR.items(): r[f"price_r{p_}_vs_unc"] = np.sqrt(c["g"] + np.sum(c["Z"] ** 2) / trYi) - 1
            print({k: v for k, v in r.items() if k.startswith("price")}, flush=True)
        rec.append(r)
    out = dict(n=n, m=m, b=b, T=T, N=N, ncol=ncol, sec=time.perf_counter() - t_all, rec=rec)
    tag = (f"_r{'-'.join(map(str, ranks))}" if ranks else "") + ("_sharp" if sharp else "")
    json.dump(out, open(f"results/e35_exact_n{n}_b{b}{tag}.json", "w"), default=float)
    for ext in ("", ".sig.npy", ".pkf.npy"):
        if os.path.exists(ckf + ext): os.remove(ckf + ext)
    return out

if __name__ == "__main__":
    a = sys.argv[1:]
    opt = a[5:]; rk = next((tuple(int(v) for v in o.split(",")) for o in opt if o[0].isdigit()), ())
    main(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), check=("check" in opt), ranks=rk, sharp=("sharp" in opt))
