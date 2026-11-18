"""Body of the main results table, Table 2 (T=600 runs, 236 in total; n=1200, b=2 split by m).
The printed rows also carry the medians of the basic, sharp and rank-20 prices over the 183 runs of the low-rank study
(e17_lowrank_*.csv); Table 2 shows the sharp and rank-20 medians; basic >= sharp >= rank-20 holds run by run
(Theorem 1(c), Suppl. Corollary S1)."""
import pandas as pd, numpy as np, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def load_t600():
    parts = []
    for f, keep in [("e1x_T600_n240.csv", None), ("e1x_T600_n240b34.csv", None),
                    ("e1x_T600_n600.csv", None), ("e1x_T600_nb34.csv", 600),
                    ("e1x_T600_n600b34.csv", None), ("e1x_T600_n1200.csv", None),
                    ("e1x_T600_n1200L1.csv", None), ("e1x_T600_n1200L2.csv", None)]:
        if not os.path.exists(os.path.join(R, f)): continue
        d = pd.read_csv(os.path.join(R, f))
        if keep is not None: d = d[d.n == keep]
        parts.append(d)
    d = pd.concat(parts)
    d = d[d.status == "ok"].drop_duplicates(subset=["n", "sw", "Pe", "m", "b", "seed"])
    return d

def load_e17():
    import glob
    fs = sorted(glob.glob(os.path.join(R, "e17_lowrank_*.csv")))
    e = pd.concat([pd.read_csv(f) for f in fs if "lownoise" not in f])
    return e.drop_duplicates(subset=["n", "sw", "Pe", "m", "b", "seed"])

if __name__ == "__main__":
    d = load_t600(); e = load_e17()
    print("runs:", len(d), file=sys.stderr)
    def q(s): return f"{np.median(s):.2f} [{s.min():.2f}, {s.max():.2f}]"
    out = []
    for n in (240, 600, 1200):
        cells = [(b, d[(d.n == n) & (d.b == b)], str(b)) for b in (1, 2, 3, 4)]
        if n == 1200:   # bimodal cell: split b = 2 by sensor count
            r2 = cells[1][1]
            cells[1:2] = [(2, r2[r2.m == 32], "2, $m{=}32$"), (2, r2[r2.m == 8], "2, $m{=}8$")]
        for i, (b, rr, lab) in enumerate(cells):
            lead = f"{n:>4}" if i == 0 else "    "
            ee = e[(e.n == n) & (e.b == b)]
            if n == 1200 and b == 2: ee = ee[ee.m == rr.m.iloc[0]]
            if len(ee):
                assert (ee.price_min >= ee.price_0 - 1e-9).all() and (ee.price_0 >= ee.price_20 - 1e-9).all()
                pm = f"{100 * np.median(ee.price_min):.1f}"
                sh = f"{100 * np.median(ee.price_0):.1f}"
                r20 = f"{100 * np.median(ee.price_20):.1f}"
            else:
                pm = sh = r20 = "--"
            out.append(f" {lead} & {lab} & {q(rr.gamma_ss)} & {np.median(rr.rho_tr_ss):.2f} & {np.median(rr.sdc_ss):.2f} & "
                       f"{rr.covw_tr_min.min():.3f} & {np.median(rr.nees_c_ss):.2f} & {len(ee)} & {pm} & {sh} & {r20} \\\\  % runs={len(rr)}")
        out.append(r"\addlinespace")
    print("\n".join(out[:-1]))
