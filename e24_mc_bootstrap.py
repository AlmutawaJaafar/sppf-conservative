"""E24: is the number of certified Monte Carlo coverage estimates below 0.95 (Suppl. Table S7) consistent with sampling
error once temporal correlation is accounted for?  The empirical coverages at consecutive steps use the same
trajectories, so their sampling errors are correlated in time, which inflates the variance of the count.
For every configuration and functional the standardised deviation z_t = (empirical - exact)/sqrt(c(1-c)/N) is
fitted by an AR(1) model (lag-1 correlation rho); 20000 series with the same rho are simulated, mapped back
through the exact coverage c_t (rounded to the grid k/N of attainable empirical values), and the number of steps
below 0.95 is counted. The result is compared with the observed count and with the independent-binomial
expectation.  usage: python e24_mc_bootstrap.py  ->  stdout"""
import numpy as np
from scipy.stats import binom
d = np.load("results/e9_series.npz")
cfg = {"lownoise": 2000, "diffuse": 2000, "b1large": 2000, "growing": 2000, "growing1200": 1000}
rng = np.random.default_rng(0); B = 20000
obs = 0; mu = 0.0; var = 0.0; cnt = np.zeros(B); rhos = []
for c, N in cfg.items():
    for q in ("worst", "mean", "unobs"):
        emp, ex = d[f"{c}__{q}_c"], np.clip(d[f"{c}__{q}_pred_c"], 1e-6, 1 - 1e-6)
        obs += int((emp < 0.95 - 1e-12).sum())
        p = binom.cdf(np.ceil(0.95 * N) - 1, N, ex); mu += p.sum(); var += (p * (1 - p)).sum()
        sd = np.sqrt(ex * (1 - ex) / N); z = (emp - ex) / sd
        rho = float(np.corrcoef(z[:-1], z[1:])[0, 1]); rhos.append(rho); T = len(z)
        e = rng.standard_normal((B, T)); zz = np.empty_like(e); zz[:, 0] = e[:, 0]
        for t in range(1, T): zz[:, t] = rho * zz[:, t - 1] + np.sqrt(1 - rho ** 2) * e[:, t]
        sim = np.round((ex + sd * zz) * N) / N
        cnt += (sim < 0.95 - 1e-12).sum(1)
print(f"observed below 0.95: {obs}; independent binomial: expected {mu:.1f}, sd {np.sqrt(var):.1f}")
print(f"lag-1 correlation of standardised deviations: {np.min(rhos):.2f}-{np.max(rhos):.2f} (median {np.median(rhos):.2f})")
lo, hi = np.percentile(cnt, [2.5, 97.5])
print(f"AR(1) simulation: mean {cnt.mean():.1f}, sd {cnt.std():.1f}, 95% interval [{lo:.0f}, {hi:.0f}], P(count >= {obs}) = {(cnt >= obs).mean():.3f}")
