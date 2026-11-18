"""Rows for Table tab:models from results/e11_model_classes.csv."""
import os, pandas as pd, numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
d = pd.read_csv(os.path.join(R, "e11_model_classes.csv"))
d["failed"] = (d.status == "diverged") | (d.gamma_ss > 1e3)
d["settled"] = ~d.failed & ((d.gamma_ss / d.gamma_501_550 - 1).abs() < 0.01)
lab = {("corrq", 0.5): r"correlated, $\ell=0.5$", ("corrq", 1.0): r"correlated, $\ell=1$", ("het", 0.0): r"heterogeneous"}
def f(x): return f"{x:.2f}"
def rng(v, fmt=f):
    return "--" if len(v) == 0 else (fmt(np.median(v)) + (f" [{fmt(v.min())}, {fmt(v.max())}]" if len(v) > 1 else ""))
tot_fail = 0
for (var, ell), g0 in d.groupby(["variant", "ell"], sort=False):
    for b, g in g0.groupby("b"):
        ok = g[g.settled]; un = g[~g.failed & ~g.settled]
        tot_fail += g.failed.sum()
        price = rng((ok.sdc_ss - 1) * 100, lambda x: f"{x:.0f}")
        print(f"{lab[(var, ell)]} & {b} & {len(g)} & {int(g.failed.sum())} & {len(un)} & {rng(ok.gamma_ss.values)} & "
              f"{rng(ok.rho_tr_ss.values)} & {price} & {f(g.covw_tr_min.min())} & "
              f"{f(g[~g.failed].covw_c_min.min()) if (~g.failed).any() else '--'} \\\\")
okall = d[~d.failed]
print("% failed total", int(tot_fail), "of", len(d), "; min cert cov (non-failed)", okall.covw_c_min.min(),
      "; max rho_c in failed", d[d.status == "diverged"].rho_c_max.max(),
      "; max slack settled", (d[d.settled].gamma_ss / d[d.settled].rho_tr_ss).max(),
      "; min uncorr worst", d.covw_tr_min.min())
print("% unsettled:", d[~d.failed & ~d.settled][["variant","ell","b","m","seed","gamma_301_350","gamma_501_550","gamma_ss"]].to_string())
print("% shift active in failed-ok:", d[d.failed & (d.status=='ok')][["variant","ell","b","m","shift_steps","gamma_ss"]].to_string())
