"""make_figures.py -- figures and tables for Section 5 from results/ (run after sppf_certified.py)."""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(FIG, exist_ok=True)

# validated categorical slots (light): blue, orange, aqua; ink colours for text
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SEQ = ["#9ec5f0", "#5f9fe6", "#2a78d6", "#14508f"]          # one hue, light -> dark (b = 1..4)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.bbox": "tight", "font.family": "serif"})

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig); print("wrote", name)

def read_csv(name):
    with open(os.path.join(R, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try: r[k] = float(v)
            except (ValueError, TypeError): pass
    return rows

def nominal(ax, x=0.98, va="top"):
    ax.axhline(0.95, color=INK2, lw=0.9, ls=(0, (1, 1.5)))
    ax.annotate("nominal 0.95", xy=(x, 0.95), xycoords=("axes fraction", "data"),
                xytext=(0, -3 if va == "top" else 3), textcoords="offset points",
                ha="right", va=va, fontsize=7.5, color=INK2)

# ---------------------------------------------------------------- E3: coverage over time
def fig_e3():
    d = np.load(os.path.join(R, "e3_series.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for a, key, func, title in [(ax[0], "steady_b1", "worst", "(a) steady start, $b=1$: worst linear functional"),
                                (ax[1], "diffuse_b2", "mean", "(b) diffuse start, $b=2$: domain-mean temperature")]:
        t = np.arange(1, len(d[f"{key}__gamma"]) + 1)
        a.plot(t, d[f"{key}__mc__{func}_pred_tr"], color=ORANGE, ls="--", label="sparse filter, uncorrected")
        a.plot(t, d[f"{key}__mc__{func}_pred_c"], color=BLUE, label="certified (Algorithm 1)")
        a.plot(t[::6], d[f"{key}__mc__{func}_tr"][::6], "o", ms=3.2, color=ORANGE, mfc="white", mew=1.0)
        a.plot(t[::6], d[f"{key}__mc__{func}_c"][::6], "s", ms=3.0, color=BLUE, mfc="white", mew=1.0)
        nominal(a, x=0.98 if key.startswith("steady") else 0.55, va="top" if key.startswith("steady") else "top")
        a.set_title(title, fontsize=8.5, loc="left", color=INK)
        a.set_xlabel("time step $t$")
    ax[0].set_ylabel("coverage of 95% interval"); ax[0].set_ylim(0.80, 1.005); ax[1].set_ylim(0.80, 1.005)
    # inset with the full range of panel (b): the uncorrected coverage dips to about 0.2 after the diffuse start
    ins = ax[1].inset_axes([0.50, 0.10, 0.45, 0.42])
    t = np.arange(1, len(d["diffuse_b2__gamma"]) + 1)
    ins.plot(t, d["diffuse_b2__mc__mean_pred_tr"], color=ORANGE, ls="--", lw=0.9)
    ins.plot(t, d["diffuse_b2__mc__mean_pred_c"], color=BLUE, lw=0.9)
    ins.axhline(0.95, color=INK2, lw=0.6, ls=(0, (1, 1.5))); ins.set_ylim(0, 1.05)
    ins.set_yticks([0, 0.5, 1]); ins.tick_params(labelsize=6.5); ins.set_title("full range", fontsize=6.5, pad=1.5)
    h, l = ax[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", ls="", color=INK2, mfc="white", ms=3.5)]
    l += ["Monte Carlo (2000 runs)"]
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    save(fig, "fig_e3_coverage")

# ---------------------------------------------------------------- E17: scalar vs scalar-plus-low-rank certificates
def fig_e17():
    import pandas as pd
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.75))
    meth = [("price_0", "sharp (scalar)", ORANGE, "-", "o"),
            ("price_5", "rank 5", AQUA, "-", "s"), ("price_20", "rank 20", BLUE, "-", "^")]
    for a, n, title in [(ax[0], 240, "(a) $n=240$, 144 runs"), (ax[1], 600, "(b) $n=600$, 24 runs")]:
        d = pd.read_csv(os.path.join(R, f"e17_lowrank_s{n}.csv"))
        g = d.groupby("b")
        for k, (col, lab, c, ls, mk) in enumerate(meth):
            med = 100 * g[col].median(); hi = 100 * g[col].max(); x = med.index.values + (k - 1.0) * 0.06
            a.plot(x, med.values, ls=ls, marker=mk, ms=3.5, color=c, label=lab, mfc="white", mew=1.0)
            a.vlines(x, med.values, hi.values, color=c, lw=0.9, alpha=0.8)
        a.set_yscale("log"); a.set_xticks([1, 2, 3, 4]); a.set_xlabel("bandwidth $b$")
        a.set_title(title, fontsize=8.5, loc="left", color=INK); a.set_ylim(0.3, 1e4)
    ax[1].set_yticklabels([])
    ax[0].set_ylabel("price (%)")
    hh, ll = ax[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=8)
    save(fig, "fig_e17_lowrank")

# ---------------------------------------------------------------- E14: growth rate nu separates settled and unsettled runs
def fig_e14():
    import pandas as pd, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_tab3 import load_t600
    d = load_t600(); e = pd.read_csv(os.path.join(R, "e14_growth_rates.csv"))
    x = d.merge(e, on=["n", "sw", "Pe", "m", "b", "seed"])
    x["growth"] = x.gamma_ss / x.gamma_501_550; x["settled"] = (x.growth - 1).abs() < 0.01
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for st, c, mk, lab in [(True, INK2, "o", "settled"), (False, ORANGE, "s", "still rising at $t=600$")]:
        y = x[x.settled == st]
        ax.scatter(y.nu_gm_551_600, y.growth, s=14, marker=mk, facecolor="white", edgecolor=c, linewidth=1.0, label=f"{lab} ({len(y)})")
    ax.axvline(1.0, color=INK, lw=0.6, ls=":"); ax.axhline(1.0, color=INK, lw=0.6, ls=":")
    xx = np.linspace(1.0, x.nu_gm_551_600.max() * 1.001, 50)
    ax.plot(xx, xx ** 50, color=BLUE, lw=1.0, ls="--", label=r"lower bound $\nu^{50}$ (Eq. 9)")
    ax.set_xlabel(r"geometric mean of $\nu_t$, steps 551--600")
    ax.set_ylabel(r"growth of $\gamma_t$ over 50 steps")
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    save(fig, "fig_e14_nu")

# ---------------------------------------------------------------- E22: offline design (Algorithm 2) predicts the price
def fig_e22():
    import pandas as pd, glob
    k = ["n", "sw", "Pe", "m", "b", "seed"]
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "e22_design_*.csv")))])
    e = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("e17_lowrank_s240.csv", "e17_lowrank_s600.csv", "e17_lowrank_h600.csv", "e17_lowrank_h1200_0_2.csv",
                                                       "e17_lowrank_h1200_2_5.csv", "e17_lowrank_hard.csv")]).drop_duplicates(k)
    x = d.merge(e, on=k)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    fin = np.isfinite(x.price_sharp_pred)
    for n, mk in [(240, "o"), (600, "s"), (1200, "D")]:
        y = x[fin & (x.n == n)]
        if len(y): ax.scatter(100 * y.price_sharp_pred, 100 * y.price_0, s=13, marker=mk, facecolor="white", edgecolor=ORANGE,
                              linewidth=0.9, label=f"sharp, $n={n}$ ({len(y)})")
    y = x[(x.n == 240) & x.price_r20_pred.notna()] if "price_r20_pred" in x else x.iloc[:0]
    if len(y): ax.scatter(100 * y.price_r20_pred, 100 * y.price_20, s=10, marker="^", facecolor="white", edgecolor=BLUE, linewidth=0.8,
                          label=f"rank 20, $n=240$ ({len(y)})")
    lim = [0.3, 400]; ax.plot(lim, lim, color=INK2, lw=0.7, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted offline, steady state (%)"); ax.set_ylabel("observed in closed loop (%)")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    save(fig, "fig_e22_design")

# ---------------------------------------------------------------- E4: decay (along the sparse filter's own trajectory)
def fig_e4():
    rows = read_csv("e4w_decay_working.csv")
    from matplotlib.ticker import FuncFormatter, LogLocator
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for (n, col, mk) in [(240, BLUE, "o"), (600, ORANGE, "s"), (1200, AQUA, "^")]:
        rr = [r for r in rows if r["n"] == n and r["sw"] == 0.05]
        b = np.array([r["b"] for r in rr])
        ax[0].semilogy(b, [r["Dinf_rel"] for r in rr], marker=mk, color=col, ms=4, label=f"$n={n}$")
        ax[1].semilogy(b, [r["rel2"] for r in rr], marker=mk, color=col, ms=4, label=f"$n={n}$")
    ax[0].set_title(r"(a) $\|D_t\|_\infty/\|P_{t|t-1}^{-1}\|_\infty$ (Assumption S1)", fontsize=8.5, loc="left")
    ax[1].set_title(r"(b) $\|P_{t|t-1}^{1/2}D_tP_{t|t-1}^{1/2}\|_2$ (bounds $u_t-1$)", fontsize=8.5, loc="left")
    for a, lo in zip(ax, (1e-4, 1e-3)):
        a.set_xlabel("bandwidth $b$"); a.legend(fontsize=7.5); a.set_ylim(lo, 0.2)
        a.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        a.yaxis.set_minor_formatter(FuncFormatter(lambda v, p: ""))
        a.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    save(fig, "fig_e4_decay")
    out = []
    for n in (240, 600, 1200):
        for sw in (0.2, 0.05, 0.01):
            rr = [r for r in rows if r["n"] == n and r["sw"] == sw and r["b"] <= 5]
            if not rr: continue
            b = np.array([r["b"] for r in rr])
            th = lambda k: float(np.exp(np.polyfit(b, np.log([r[k] for r in rr]), 1)[0]))
            out.append((n, sw, round(th("Dinf"), 3), round(th("rel2"), 3)))
    print("fitted ratios (b=1..5) [n, sw, theta_Dinf, theta_rel2]:", out)
    return out

# ---------------------------------------------------------------- E5: cost
def fig_e5():
    rows = read_csv("e5_cost.csv")
    n = np.array([r["n"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.loglog(n, [r["t_factor_Y"] + r["t_mean_solve"] for r in rows], "o-", color=BLUE, ms=4,
              label="sparse update (factor + solve)")
    ax.loglog(n, [r["t_lanczos_eps0.002"] for r in rows], "s-", color=ORANGE, ms=4,
              label=r"certificate, $\varepsilon=0.002$, Prop. S1(ii)")
    ax.loglog(n, [r["t_lanczos_all_eps0.002"] for r in rows], "s:", color=ORANGE, ms=3.5, mfc="white",
              label=r"certificate, $\varepsilon=0.002$, Prop. S1(iii), $t=10^3$")
    ax.loglog(n, [r["t_lanczos_eps0.01"] for r in rows], "^--", color=AQUA, ms=4,
              label=r"certificate, $\varepsilon=0.01$, Prop. S1(ii)")
    if os.path.exists(os.path.join(R, "e12_verified_cost.csv")):     # estimate-then-verify (Lemma 3(b))
        v = read_csv("e12_verified_cost.csv")
        ax.loglog([r["n"] for r in v], [r["t_lanczos_est"] + r["t_verify"] for r in v], "v-", color=INK2, ms=4,
                  mfc="white", label=r"verified certificate ($k=40$ + $LDL^\top$ test)")
    nd = [(r["n"], r["t_dense_kf"]) for r in rows if r["t_dense_kf"] == r["t_dense_kf"]]
    if nd:
        x, y = zip(*nd); ax.loglog(x, y, "D", color=INK2, ms=4, label="dense Kalman step")
        xx = np.array([nd[0][0], 1e5]); ax.loglog(xx, nd[0][1] * (xx / nd[0][0]) ** 3, ":", color=INK2, lw=1)
    ax.set_xlabel("state dimension $n$"); ax.set_ylabel("time per step [s]")
    f20 = os.path.join(R, "e20_large_n10000_b3.json")
    if os.path.exists(f20):                                           # measured closed loop, warm-started verified certificate
        import json
        rr = json.load(open(f20))["rec"]
        ax.plot([1e4], [np.mean([x["t_cert"] for x in rr])], "*", ms=10, color=INK, mfc=ORANGE, mew=0.8,
                label=r"verified, warm-started, closed loop ($b=3$, Sec. 5.9)")
    ax.set_ylim(1e-5, 1e3); ax.legend(fontsize=6.3, loc="lower right")
    save(fig, "fig_e5_cost")

# ---------------------------------------------------------------- E6: low process noise
def fig_e6():
    d = np.load(os.path.join(R, "e6_series.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for b in (1, 2, 3, 4):
        g = d[f"b{b}__gamma"]; t = np.arange(1, len(g) + 1)
        ax[0].semilogy(t, g, color=SEQ[b - 1], label=f"$b={b}$")
        ax[1].plot(t, d[f"b{b}__cov_worst_tr"], color=SEQ[b - 1], ls="--", label=f"$b={b}$")
    ax[0].set_title(r"(a) certificate $\gamma_t$", fontsize=8.5, loc="left")
    ax[1].set_title("(b) uncorrected filter: worst-functional coverage", fontsize=8.5, loc="left")
    nominal(ax[1]); ax[1].set_ylim(0, 1.03)
    ax[0].legend(fontsize=7.5, ncol=2, loc="lower right"); ax[1].legend(fontsize=7.5, ncol=2, loc="lower left",
                                                                        bbox_to_anchor=(0.0, 0.55))
    for a in ax: a.set_xlabel("time step $t$")
    save(fig, "fig_e6_diagnostic")

# ---------------------------------------------------------------- E1/E2: guarantee + tightness
def fig_e2_and_tables():
    rows = []
    for nm in ("e1_n240.csv", "e1_n600.csv", "e1_n1200.csv"):
        if os.path.exists(os.path.join(R, nm)): rows += [r for r in read_csv(nm) if r.get("status") == "ok"]
    if not rows: print("no e1 results yet"); return
    # Table 3 / Fig. 1 use the T=600 reruns of all configurations (plus the extra n=1200 layouts) when present
    t600 = [("e1x_T600_n240.csv", None), ("e1x_T600_n240b34.csv", None), ("e1x_T600_n600.csv", None),
            ("e1x_T600_nb34.csv", 600), ("e1x_T600_n600b34.csv", None), ("e1x_T600_n1200.csv", None)]
    t600 += [(f, None) for f in ("e1x_T600_n1200L1.csv", "e1x_T600_n1200L2.csv")
             if os.path.exists(os.path.join(R, f))]   # extra n=1200 layouts (E10), when present
    if all(os.path.exists(os.path.join(R, f)) for f, _ in t600):
        seen, rows = set(), []
        for f, keep in t600:
            for r in read_csv(f):
                key = (r["n"], r["sw"], r["Pe"], r["m"], r["b"], r["seed"])
                if r.get("status") == "ok" and (keep is None or r["n"] == keep) and key not in seen:
                    seen.add(key); rows.append(r)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    cols = {240: (BLUE, "o"), 600: (ORANGE, "s"), 1200: (AQUA, "^")}
    sel = lambda r: r["sw"] == 0.05
    for n, (c, mk) in cols.items():
        bs = sorted({r["b"] for r in rows if r["n"] == n})
        if not bs: continue
        for j, key in enumerate(["gamma_ss", "sdc_ss"]):
            med = [np.median([r[key] for r in rows if r["n"] == n and r["b"] == b and sel(r)]) for b in bs]
            lo = [np.min([r[key] for r in rows if r["n"] == n and r["b"] == b and sel(r)]) for b in bs]
            hi = [np.max([r[key] for r in rows if r["n"] == n and r["b"] == b and sel(r)]) for b in bs]
            off = {240: -0.08, 600: 0.0, 1200: 0.08}[n]
            ax[j].errorbar(np.array(bs) + off, np.array(med) - 1, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
                           marker=mk, color=c, ms=4, capsize=2, lw=1.4, label=f"$n={int(n)}$")
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].set_title(r"(a) price of sparsification $\gamma_t-1$ (end of run)", fontsize=8.5, loc="left")
    ax[1].set_title(r"(b) certified / optimal std. dev. $-\,1$", fontsize=8.5, loc="left")
    for a in ax: a.set_xlabel("bandwidth $b$"); a.set_xticks([1, 2, 3, 4]); a.legend(fontsize=7.5)
    save(fig, "fig_e2_tightness")
    # Table: guarantee and tightness summary
    with open(os.path.join(FIG, "table_e1.txt"), "w") as f:
        def line(s): print(s); f.write(s + "\n")
        line("Guarantee check (exact second moments): max over runs and time of lambda_max((Pc)^-1 M)")
        for n in (240, 600, 1200):
            rr = [r for r in rows if r["n"] == n]
            if rr: line(f"  n={n}: runs={len(rr)}, max rho_c = {max(r['rho_c_max'] for r in rr):.9f}, "
                        f"max rho_uncorrected = {max(r['rho_tr_max'] for r in rr):.3f}")
        line("\nBy bandwidth (all runs, median [min, max]):")
        line(f"{'n':>5} {'b':>2} | {'gamma_ss':>22} | {'cert sd/opt':>22} | {'worst-cov uncorr (min)':>24} | {'worst-cov cert (min)':>21} | {'rmse/opt':>9}")
        for n in (240, 600, 1200):
            for b in (1, 2, 3, 4):
                rr = [r for r in rows if r["n"] == n and r["b"] == b]
                if not rr: continue
                q = lambda k: f"{np.median([r[k] for r in rr]):.3f} [{min(r[k] for r in rr):.3f},{max(r[k] for r in rr):.3f}]"
                line(f"{n:>5} {b:>2} | {q('gamma_ss'):>22} | {q('sdc_ss'):>22} | {min(r['covw_tr_min'] for r in rr):>24.3f} | "
                     f"{min(r['covw_c_min'] for r in rr):>21.3f} | {np.median([r['rmse_ss'] for r in rr]):>9.4f}")

if __name__ == "__main__":
    which = sys.argv[1:] or ["e3", "e4", "e5", "e6", "e2"]
    for w in which: {"e3": fig_e3, "e14": fig_e14, "e22": fig_e22, "e17": fig_e17, "e4": fig_e4, "e5": fig_e5, "e6": fig_e6, "e2": fig_e2_and_tables}[w]()
