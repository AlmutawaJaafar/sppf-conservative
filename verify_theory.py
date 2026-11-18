"""Numerical checks of the LMI characterisation, the sharp certificate, the stability bound, the bordered test,\nRemark 2 (oracle factor) and Suppl. Corollary S1 (a rank-p certificate is never above the sharp one)."""
import numpy as np, scipy.linalg as sla
from sppf_certified import Model, sym, chol, inv_from_chol, lmax, lmin
rng = np.random.default_rng(0)
for (Nx, Ny, b, m, steps) in [(20, 12, 1, 8, 40), (20, 12, 2, 8, 200), (30, 20, 2, 32, 80)]:
    mod = Model(Nx, Ny, m=m); n = mod.n; F, Q = mod.F, mod.Q; J = mod.HRH; I = np.eye(n)
    mask = mod.dist <= b; Yinv = 4.0 * I; gam = 1.0; gs = 1.0; Yinv_s = 4.0 * I
    M = 4.0 * I; Ms = 4.0 * I; worst = {}
    for t in range(steps):
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        S = S + max(0, 1e-8 * np.mean(np.diag(S)) - lmin(S)) * I
        Y = S + J; Yi = inv_from_chol(chol(Y))
        Pc = gam * (Pbar - Q) + Q
        lam = lmax(chol(S).T @ Pc @ chol(S))
        # (1A) LMI: W = Y_{t-1}/gam_{t-1};  B(g) = [[W, F'S],[SF, gS - SQS]] >= 0  <=>  lam <= g
        # balanced by the congruence diag(sqrt(gam) I, I/sqrt(g)), which leaves the sign unchanged
        # but avoids mixing entries of size 1/gam and g (unbalanced, the test fails in rounding for gam ~ 1e5)
        Yprev = inv_from_chol(chol(Yinv))
        def Bmin(g, sharp=False):
            c = np.sqrt(gam / g)
            low = (Y - (J + S @ Q @ S) / g) if sharp else (S - S @ Q @ S / g)
            Bm = np.block([[Yprev, c * F.T @ S], [c * S @ F, low]]); return lmin(Bm) / np.abs(Bm).sum(1).max()
        a1 = Bmin(lam * (1 + 1e-7)) >= -1e-12 and Bmin(lam * (1 - 1e-4)) < 0
        # (1B) sharp: g# = lmax(Y^{-1}(S Pc S + J)) <= max(1, lam); LMI with gY - J - SQS
        X = sym(S @ Pc @ S + J); gsh = sla.eigh(X, Y, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]
        a2 = gsh <= max(1, lam) * (1 + 1e-10)
        a3 = Bmin(gsh * (1 + 1e-7), True) >= -1e-12 and Bmin(gsh * (1 - 1e-4), True) < 0
        # optimality: with M_prior = Pc, the Joseph moment equals Y^{-1} X Y^{-1}; lmax(Y * that) = g#
        A = Yi @ S; K = Yi @ mod.H.T / mod.sv ** 2
        Mj = A @ Pc @ A.T + mod.sv ** 2 * K @ K.T
        a4 = abs(sla.eigh(sym(Mj), Yi, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0] / gsh - 1) < 1e-8
        gam = max(1.0, (1 + 1e-10) * lam)
        Mp = sym(F @ M @ F.T + Q); M = sym(A @ Mp @ A.T + mod.sv ** 2 * K @ K.T)
        # (1C) tr M <= gam tr(Y^{-1}) <= n gam / lmin(S)
        a5 = np.trace(M) <= gam * np.trace(Yi) * (1 + 1e-9) <= n * gam / lmin(S) * (1 + 1e-9)
        Yinv = Yi
        for k, v in dict(lmi=a1, sharp_le=a2, sharp_lmi=a3, sharp_opt=a4, trace=a5).items():
            worst[k] = worst.get(k, True) and bool(v)
    print(n, b, m, steps, worst, "gam", round(gam, 4), "last g#", round(gsh, 4))

# ---------------------------------------------------------------------------------------------
# (2) Section 4.5: bordered-matrix verification of a scalar-plus-rank-p certificate (Corollary 3).
# The certificate (g, Z) satisfies (8) strictly iff the bordered matrix has exactly p negative and
# no zero eigenvalues. Checked (a) against a direct eigenvalue test on random dense cases with
# g_{t-1} up to 1e5, after the diagonal scaling D B D used in the paper, and (b) along an actual
# rank-5 certificate of the filter: (g_t, Z_t) must pass, and (g_t (1 - 1e-3), Z_t) must fail,
# because g_t is the smallest scalar admitting any rank-p correction.
def bordered_negatives(Yp, gp, Zp, F, S, Q, J, Y, g, Z):
    n, p = S.shape[0], Z.shape[1]; O = np.zeros
    W = Y @ Z; K = g * Y - S @ Q @ S - J
    B = np.block([[Yp / gp, O((n, p)), F.T @ S, O((n, p))],
                  [O((p, n)), np.eye(p), Zp.T @ F.T @ S, O((p, p))],
                  [S @ F, S @ F @ Zp, K, W],
                  [O((p, n)), O((p, p)), W.T, -np.eye(p)]])
    d = 1 / np.sqrt(np.abs(np.diag(B))); B = d[:, None] * B * d[None, :]
    ev = np.linalg.eigvalsh(sym(B)); return int((ev < 0).sum()), np.abs(ev).min()

rng = np.random.default_rng(1); nn, pp, ok, N = 30, 4, 0, 400
def spd(k): A = rng.standard_normal((k, k)); return A @ A.T / k + 0.1 * np.eye(k)
for _ in range(N):
    Yp, S, Q = spd(nn), spd(nn), spd(nn); J = np.diag(rng.random(nn)); Y = S + J
    F = rng.standard_normal((nn, nn)) / np.sqrt(nn); gp = 10 ** rng.uniform(0, 5)
    Zp = rng.standard_normal((nn, pp)); Z = rng.standard_normal((nn, pp)) * rng.uniform(0, 3)
    base = S @ Q @ S + J + gp * S @ F @ np.linalg.solve(Yp, F.T @ S) + (S @ F @ Zp) @ (S @ F @ Zp).T - (Y @ Z) @ (Y @ Z).T
    g = np.max(np.linalg.eigvals(np.linalg.solve(Y, base)).real) * rng.choice([0.98, 1.02])
    T = g * Y - base
    ok += (np.linalg.eigvalsh(sym(T)).min() > 0) == (bordered_negatives(Yp, gp, Zp, F, S, Q, J, Y, g, Z)[0] == pp)
print(f"bordered test vs direct check, random cases: {ok}/{N} agree")

mod = Model(20, 12, Pe=50.0, sw=0.05, m=8, seed=0); F, Q, J = mod.F, mod.Q, mod.HRH
n = mod.n; I = np.eye(n); mask = mod.dist <= 2; p = 5
Yinv = 0.01 * I; g = 0.0; Z = None; passed = failed_low = checked = 0
for t in range(120):
    Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
    S = S + max(0, 1e-8 * np.mean(np.diag(S)) - lmin(S)) * I; Y = S + J; Yi = inv_from_chol(chol(Y))
    Cprev = 0.01 * I if t == 0 else g * Yinv + Z @ Z.T
    X = sym(S @ sym(F @ Cprev @ F.T + Q) @ S + J); lam, V = sla.eigh(X, Y); lam, V = lam[::-1], V[:, ::-1]
    gnew = lam[p]; Znew = V[:, :p] * np.sqrt(np.maximum(lam[:p] - gnew, 0))
    if t >= 1 and t % 10 == 9:
        Yp = inv_from_chol(chol(Yinv)); checked += 1
        passed += bordered_negatives(Yp, g, Z, F, S, Q, J, Y, gnew * (1 + 1e-7), Znew)[0] == p
        failed_low += bordered_negatives(Yp, g, Z, F, S, Q, J, Y, gnew * (1 - 1e-3), Znew)[0] != p
    g, Z, Yinv = gnew, Znew, Yi
print(f"bordered test along a rank-5 certificate (n=240, b=2): pass {passed}/{checked}, reject lowered g {failed_low}/{checked}")


# Proposition 2 (oracle) and Suppl. Corollary S1 (rank p never above sharp) along a run (n=240, b=1, Pe=50)
mod = Model(20, 12, Pe=50.0, sw=0.05, m=8, seed=0); F, Q, J, H, sv = mod.F, mod.Q, mod.HRH, mod.H, mod.sv
n = mod.n; I = np.eye(n); mask = mod.dist <= 1; P0 = 0.01 * I
Yinv = P0.copy(); gs = 1.0; C5 = P0.copy(); M = P0.copy(); ok_or = ok_rk = True
for t in range(200):
    Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
    S = S + max(0, 1e-8 * np.mean(np.diag(S)) - lmin(S)) * I; Y = S + J; Yi = inv_from_chol(chol(Y))
    gs = sla.eigh(sym(S @ (gs * (F @ Yinv @ F.T) + Q) @ S + J), Y, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]
    X = sym(S @ sym(F @ C5 @ F.T + Q) @ S + J); lam, V = sla.eigh(X, Y); lam, V = lam[::-1], V[:, ::-1]
    C5 = sym(lam[5] * Yi + (V[:, :5] * (lam[:5] - lam[5])) @ V[:, :5].T)
    A = Yi @ S; K = Yi @ H.T / sv ** 2; M = sym(A @ sym(F @ M @ F.T + Q) @ A.T + sv ** 2 * K @ K.T)   # M_0 = P_0
    oracle = sla.eigh(M, Yi, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]
    ok_or &= gs >= oracle * (1 - 1e-9)
    ok_rk &= np.linalg.eigvalsh(sym(gs * Yi - C5)).min() >= -1e-9 * gs * np.abs(Yi).max()
    Yinv = Yi
print(f"Proposition 2 (sharp >= oracle factor) along 200 steps: {bool(ok_or)}; Suppl. Corollary S1 (rank-5 <= sharp): {bool(ok_rk)}")
