"""Price-targeted design rule (Algorithm 2 with a price target tau) evaluated against the closed-loop runs.
For each configuration (n, sw, Pe, m, seed), candidates are tried in order of cost: b = 1, 2, ... with p = 0
(sharp), and, where low-rank predictions are available (n = 240), p = 5 and 20 at each b before moving to b + 1.
The first candidate whose predicted steady-state price is <= tau is chosen; reported: whether its observed price
is <= tau + 0.5 percentage points, and whether a cheaper candidate would have met tau in closed loop."""
import sys, glob, numpy as np, pandas as pd
k = ["n", "sw", "Pe", "m", "b", "seed"]
d = pd.concat([pd.read_csv(f"results/e22_design_{w}.csv") for w in ("s240", "s600")])   # complete b = 1..4 per configuration
e = pd.concat([pd.read_csv(f"results/{f}") for f in ("e17_lowrank_s240.csv", "e17_lowrank_s600.csv", "e17_lowrank_hard.csv")])
x = d.merge(e, on=k)
def rule(tau, ranks):
    rows = []
    for key, g in x.groupby(["n", "sw", "Pe", "m", "seed"]):
        g = g.sort_values("b"); cands = []
        for _, r in g.iterrows():
            for p in ranks:
                pred = r.price_sharp_pred if p == 0 else r.get(f"price_r{p}_pred", np.nan)
                obs = r.price_0 if p == 0 else r[f"price_{p}"]
                wr = r.worst_0 if p == 0 else r[f"worst_{p}"]
                if np.isfinite(pred): cands.append((r.b, p, pred, obs, wr))
        chosen = next((c for c in cands if c[2] <= tau), None)
        if chosen is None: rows.append(dict(n=key[0], chosen=None)); continue
        cheaper_ok = any(c[3] <= tau for c in cands[:cands.index(chosen)])
        rows.append(dict(n=key[0], b=chosen[0], p=chosen[1], pred=chosen[2], obs=chosen[3], worst=chosen[4], ok=chosen[3] <= tau + 0.005, cheaper_ok=cheaper_ok))
    return pd.DataFrame(rows)
tex = []
for tau in (0.05, 0.10):
    for ranks in ((0,), (0, 5, 20)):
        r = rule(tau, ranks); r = r[r.n == 240] if ranks != (0,) else r
        got = r.dropna(subset=["b"]) if "b" in r else r
        print(f"tau={tau} ranks={ranks}: configs={len(r)}, design found={len(got)}, observed<=tau(+0.5pp)={int(got.ok.astype(bool).sum())}, "
              f"cheaper candidate would also meet tau={int(got.cheaper_ok.astype(bool).sum())}, max obs={100*got.obs.max():.2f}%  b-distribution={got.b.value_counts().sort_index().to_dict()} p={got.p.value_counts().to_dict()}")
        bp = ", ".join(f"{int(b)}/{int(p)}: {c}" for (b, p), c in got.groupby(["b", "p"]).size().items())
        tex.append(f"${100*tau:.0f}\\%$ & {'sharp' if ranks == (0,) else 'sharp, rank 5, 20'} & {len(r)} & {len(got)} & "
                   f"{int((got.obs <= tau).sum())} & {int(got.cheaper_ok.astype(bool).sum())} & ${100*got.obs.max():.1f}$ & ${np.floor(got.worst.max()*1e5)/1e5:.5f}$ & {bp} \\\\")
open("results/e22_rule_rows.tex", "w").write("\n".join(tex) + "\n"); print("\n".join(tex))
