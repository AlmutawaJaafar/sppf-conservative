"""Table 3 body: the n=10^4 runs, Monte Carlo (E20) against exact moments and the exact Kalman filter (E35).
Columns: domain-mean coverage at t=300 (Monte Carlo, exact), exact worst-functional coverage at t=300, and the price
at t=300 relative to the Kalman filter (exact) and to the uncorrected filter (E20). Prints a summary of the
exact worst-direction ratios over all checkpoints. usage: python gen_tab_large.py"""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
names = [("unc", "uncorrected"), ("cert", "scalar certificate"), ("r5", "rank-5 certificate"), ("r20", "rank-20 certificate")]
mc_key = dict(unc="cov_mean_unc", cert="cov_mean_cert", r5="cov_mean_r5", r20="cov_mean_r20")
pu_key = dict(cert="price_scalar_vs_unc", r5="price_r5_vs_unc", r20="price_r20_vs_unc")
pct = lambda v: (f"{100 * v:.1f}" if abs(v) < 0.1 else f"{100 * v:.0f}").replace("-", "$-$")
for b in (1, 2, 3):
    f20 = os.path.join(R, f"e20_large_n10000_b{b}_r5-20.json"); f35 = os.path.join(R, f"e35_exact_n10000_b{b}_r5-20.json")
    if not os.path.exists(f35):
        print(f"% b={b}: {f35} missing"); continue
    m = json.load(open(f20))["rec"][-1]; ex = json.load(open(f35))["rec"]; e = ex[-1]
    chk = [r for r in ex if "ex_ratio_unc" in r]
    for i, (k, lab) in enumerate(names):
        rk = "ex_ratio_cert" if k == "cert" else f"ex_ratio_{k}"
        worst = max(r[rk] for r in chk)
        pk = f"ex_price_{k}"
        row = [str(b) if i == 0 else "", lab, f"{m[mc_key[k]]:.3f}", f"{e[f'ex_cov_mean_{k}']:.3f}",
               f"{e[f'ex_wcov_{k}']:.3f}", pct(e[pk]), "--" if k == "unc" else pct(m[pu_key[k]])]
        print("  " + " & ".join(row) + r" \\" + f"   % max ratio over checkpoints {worst:.4f}")
    print(f"% b={b}: estimate within {100 * max(r['ex_subopt'] for r in chk):.2f}% of Kalman (RMSE); checkpoints {[r['t'] for r in chk]}")
    print(r"\addlinespace")
