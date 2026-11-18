"""E28: exact miss and false-alarm rates of per-node threshold alarms (Remark "Threshold alarms").

Truth x_t = F x_{t-1} + w_t; working filter error e_t = A_t(F e_{t-1} + w_t) - K_t v_t, A_t = Y_t^{-1} S_t.
Exact joint second moments (no simulation):
  Sigma_t = F Sigma_{t-1} F' + Q,  Xi_t = E[x_t e_t'] = (F Xi_{t-1} F' + Q) A_t',  M_t = E[e_t e_t'],
  so for a functional y = a'x and yhat = a'xhat: Var(yhat) = a'(Sigma - Xi - Xi' + M)a, Cov(y, yhat) = a'(Sigma - Xi)a.
Alarm at node j: yhat_j + 1.645 sigma_j > ell_j, ell_j = 2 sd(x_j) above the mean (exceedance probability 2.3%).
Reported per method (uncorrected, sharp, rank 20, exact Kalman filter): P(y > yhat + 1.645 sigma) (guaranteed <= 5% for a
conservative sigma), conditional miss rate P(no alarm | exceedance) and false-alarm rate P(alarm | no exceedance),
from the bivariate normal distribution function.
usage: python e28_alarm.py  ->  results/e28_alarm.csv
"""
import sys, numpy as np, pandas as pd, scipy.linalg as sla
from scipy.stats import norm, multivariate_normal as mvn
sys.path.insert(0, ".")
from sppf_certified import Model, GRIDS, sym, chol, inv_from_chol

def top_pencil(X, Y, k):
    n = Y.shape[0]; w, V = sla.eigh(X, Y, subset_by_index=[n - k, n - 1]); return w[::-1], V[:, ::-1]

def rates(sy2, sh2, cyh, sig, ell, z=1.645):
    tau = ell - z * sig
    F2 = mvn(mean=[0, 0], cov=[[sh2, cyh], [cyh, sy2]]).cdf([tau, ell])      # P(yhat <= tau, y <= ell)
    pex = norm.sf(ell / np.sqrt(sy2)); pno = norm.cdf(tau / np.sqrt(sh2))
    return dict(miss=(pno - F2) / pex, fa=(norm.cdf(ell / np.sqrt(sy2)) - F2) / (1 - pex),
                above=norm.sf(z * sig / np.sqrt(sy2 + sh2 - 2 * cyh)))

pd.set_option("display.width", 200)

def main(n=600, sw=0.05, Pe=10.0, m=8, b=1, seed=0, p0=0.01, T=600, p=20, kappa=2.0):
    Nx, Ny = GRIDS[n]; mod = Model(Nx, Ny, Pe=Pe, sw=sw, m=m, seed=seed); F, Q, J = mod.F, mod.Q, mod.HRH
    I = np.eye(n); mask = mod.dist <= b
    Yinv = p0 * I; Sig = p0 * I; Xi = p0 * I; M = p0 * I; gam = 1.0; C = p0 * I; Pk = p0 * I; Xik = p0 * I
    for t in range(T):
        Pbar = sym(F @ Yinv @ F.T + Q); S = inv_from_chol(chol(Pbar)) * mask
        S = S + max(0.0, 1e-8 * np.mean(np.diag(S)) - np.linalg.eigvalsh(S)[0]) * I
        Y = S + J; Yprev = Yinv; Yinv = inv_from_chol(chol(Y)); A = Yinv @ S
        Sig = sym(F @ Sig @ F.T + Q); Xi = (F @ Xi @ F.T + Q) @ A.T; M = sym(A @ (F @ M @ F.T + Q) @ A.T + Yinv @ J @ Yinv)
        gam = sla.eigh(sym(S @ (gam * (F @ Yprev @ F.T) + Q) @ S + J), Y, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0]
        lam, V = top_pencil(sym(S @ sym(F @ C @ F.T + Q) @ S + J), Y, p + 1); g = lam[p]
        C = sym(g * Yinv + (V[:, :p] * (lam[:p] - g)) @ V[:, :p].T)
        Ppk = sym(F @ Pk @ F.T + Q); Pn = inv_from_chol(chol(inv_from_chol(chol(Ppk)) + J))
        Xik = (F @ Xik @ F.T + Q) @ (Pn @ inv_from_chol(chol(Ppk))).T; Pk = Pn
    rows = []
    for j in range(n):
        sy2 = Sig[j, j]; ell = kappa * np.sqrt(sy2); info = 1 - Pk[j, j] / sy2
        for meth, s2, sh2, cyh in (("uncorrected", Yinv[j, j], Sig[j, j] - 2 * Xi[j, j] + M[j, j], Sig[j, j] - Xi[j, j]),
                                   ("sharp", gam * Yinv[j, j], Sig[j, j] - 2 * Xi[j, j] + M[j, j], Sig[j, j] - Xi[j, j]),
                                   (f"rank{p}", C[j, j], Sig[j, j] - 2 * Xi[j, j] + M[j, j], Sig[j, j] - Xi[j, j]),
                                   ("exact KF", Pk[j, j], Sig[j, j] - 2 * Xik[j, j] + Pk[j, j], Sig[j, j] - Xik[j, j])):
            rows.append(dict(node=j, method=meth, info=info, **rates(sy2, sh2, cyh, np.sqrt(s2), ell)))
    df = pd.DataFrame(rows); df.to_csv("results/e28_alarm.csv", index=False)
    inf = df[df["info"] >= 0.2]
    print(f"informed nodes (exact-filter variance reduction >= 20%): {inf.node.nunique()} of {n}")
    print(pd.concat({"P(y>bound) max, all": df.groupby("method")["above"].max(),
                     "miss, informed": inf.groupby("method")["miss"].mean(),
                     "false alarm, informed": inf.groupby("method")["fa"].mean(),
                     "false alarm, all": df.groupby("method")["fa"].mean()}, axis=1).round(4))
if __name__ == "__main__":
    main()
