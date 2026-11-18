"""E23: the bordered-matrix verification of Section 4.5 for low-rank certificates at n = 10^4.

Runs the fully sparse filter of E20 (Woodbury + probing S_t) with the low-rank certificates p = 5 and 20 computed by
ARPACK (safety factor 1 + 1e-3), and at selected steps verifies each candidate (g_t, Z_t) by the sparse test:
the certificate satisfies (8) strictly iff the bordered matrix
    [[Y_{t-1}/g_{t-1}, 0, F'S, 0], [0, I_p, Z_{t-1}'F'S, 0], [SF, SFZ_{t-1}, Xi, YZ], [0, 0, (YZ)', -I_p]],
    Xi = g Y - S Q S - H'R^{-1}H,
has exactly p negative and no zero eigenvalues. The inertia is read from a sparse LDL' factorisation (SuperLU in
symmetric mode, no pivoting, symmetric permutation checked) after symmetric diagonal scaling. As a control, the
same test is applied with g lowered by 0.1%, which must fail because g is the smallest admissible scalar.
usage: python e23_verify_lowrank_large.py [Nx Ny b T]  ->  results/e23_verify_lowrank_large.json
       python e23_verify_lowrank_large.py every  ->  rank 20 at every step, b = 3, T = 300 (control every 50 steps)
                                              ->  results/e23_verify_lowrank_large_every_b3.json
"""
import sys, time, json, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu, eigsh, LinearOperator
sys.path.insert(0, ".")
from sppf_certified import lap1d, upwind1d
from e20_large import ldl_posdef

def inertia_negatives(B):
    dg = np.abs(B.diagonal()); dg[dg == 0] = 1.0; D = sp.diags(1.0 / np.sqrt(dg)); B = (D @ B @ D).tocsc()
    lu = splu(B, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True, Equil=False))
    if not np.all(lu.perm_r == lu.perm_c): return None, None
    d = lu.U.diagonal(); return int((d < 0).sum()), float(np.min(np.abs(d)))

def bordered(Yp, gp, Zp, F, FT, S, q, hrh, Y, g, Z):
    n, p = S.shape[0], Z.shape[1]
    Xi = (g * Y - q * (S @ S) - sp.diags(hrh)).tocsr()
    SF = (S @ F).tocsr(); SFZ = SF @ Zp; YZ = Y @ Z
    return sp.bmat([[Yp / gp, None, (FT @ S), None],
                    [None, sp.identity(p), sp.csr_matrix(SFZ.T), None],
                    [SF, sp.csr_matrix(SFZ), Xi, sp.csr_matrix(YZ)],
                    [None, None, sp.csr_matrix(YZ.T), -sp.identity(p)]], format="csc")

def main(Nx=100, Ny=100, b=3, T=60, Fo=0.01, Pe=50.0, sw=0.05, sv=0.5, p0=0.01, seed=0, ranks=(5, 20), check_every=20, control_every=None, tag=None):
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
    LR = {p: dict(g=1.0, Z=np.zeros((n, p))) for p in ranks}
    out = []
    for t in range(1, T + 1):
        Z_ = (Y + FtF / q).tocsc(); luZ = splu(Z_, permc_spec="MMD_AT_PLUS_A")
        W = Pm / q - F @ luZ.solve(np.asarray(FT @ Pm)) / q ** 2
        rows, cols, vals = [], [], []
        for dx, dy in offs:
            ok = (x + dx >= 0) & (x + dx < Nx) & (y + dy >= 0) & (y + dy < Ny); i = np.where(ok)[0]; j = i + dx + dy * Nx
            rows.append(j); cols.append(i); vals.append(W[j, col[i]])
        Sraw = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        S = ((Sraw + Sraw.T) / 2).tocsc(); mu = 1e-8 * S.diagonal().mean(); shift = 0.0
        while not ldl_posdef(S - mu * sp.identity(n)):
            shift = max(2 * shift, mu); S = (S + shift * sp.identity(n)).tocsc()
        Ynew = (S + J).tocsc(); luY = splu(Ynew, permc_spec="MMD_AT_PLUS_A"); luYp = splu(Y.tocsc(), permc_spec="MMD_AT_PLUS_A")
        for p, c in LR.items():
            gp_, Zp = c["g"], c["Z"]
            def Xmv(v, gp_=gp_, Zp=Zp):
                u = S @ v; w = gp_ * (F @ luYp.solve(FT @ u)) + q * u
                if Zp.shape[1]: w = w + F @ (Zp @ (Zp.T @ (FT @ u)))
                return S @ w + hrh * v
            lam, V = eigsh(LinearOperator((n, n), matvec=Xmv, dtype=float), k=p + 1, M=Ynew,
                           Minv=LinearOperator((n, n), matvec=luY.solve, dtype=float), which="LA", tol=1e-8, ncv=max(2 * p + 20, 40))
            o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]; V = V / np.sqrt(np.einsum("ij,ij->j", V, Ynew @ V))
            g = (1 + 1e-3) * lam[p]; Z = V[:, :p] * np.sqrt(np.maximum((1 + 1e-3) * lam[:p] - g, 0))
            if t % check_every == 0:
                t0 = time.perf_counter(); B = bordered(Y, gp_, Zp, F, FT, S, q, hrh, Ynew, g, Z); neg, piv = inertia_negatives(B)
                t_test = time.perf_counter() - t0
                if control_every is None or t % control_every == 0:
                    B2 = bordered(Y, gp_, Zp, F, FT, S, q, hrh, Ynew, g * (1 - 1e-3), Z); neg2, _ = inertia_negatives(B2)
                else:
                    neg2 = None
                rec = dict(t=t, p=p, g=g, negatives=neg, passed=(neg == p), min_abs_pivot=piv, negatives_lowered=neg2,
                           rejected_lowered=(None if neg2 is None else neg2 != p), t_test=t_test, size=B.shape[0], nnz=B.nnz)
                print(rec, flush=True); out.append(rec)
            c["gnew"], c["Znew"] = g, Z
        for c in LR.values(): c["g"], c["Z"] = c["gnew"], c["Znew"]
        Y = Ynew
    name = tag if tag else ('' if T <= 40 else '_T' + str(T))
    json.dump(out, open(f"results/e23_verify_lowrank_large{name}.json", "w"))
    return out

if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "every":      # rank 20 verified at every step of the headline design (b = 3, T = 300)
        main(100, 100, 3, 300, ranks=(20,), check_every=1, control_every=50, tag="_every_b3")
    elif a: main(int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    else: main()
