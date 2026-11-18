"""Suppl. Table S7 (tab:mc): independent Monte Carlo check, transposed (cases as columns)."""
import pandas as pd, numpy as np, os
_HERE = os.path.dirname(os.path.abspath(__file__))
R = next((p for p in (os.path.join(_HERE, "results"), os.path.join(_HERE, "..", "exp", "results")) if os.path.isdir(p)),
         os.path.join(_HERE, "results"))   # repository layout first, development layout second
d = pd.read_csv(os.path.join(R, "e9_montecarlo.csv")).set_index("case")
z = np.load(os.path.join(R, "e9_series.npz"))
cases = ["lownoise", "diffuse", "b1large", "growing", "growing1200"]
head = [r"low noise", r"diffuse start", r"$b=1$", r"$\gamma_t$ growing", r"$\gamma_t$ growing"]
pe = {"lownoise": 10, "diffuse": 10, "b1large": 10, "growing": 10, "growing1200": 50}
def g(x):
    if x < 1e4: return f"{x:.3g}"
    m, e = f"{x:.1e}".split("e+"); return f"${m}{{\\times}}10^{{{int(e)}}}$"
rows = [("P\\'eclet number", lambda k, r: str(pe[k])), ("$n$", lambda k, r: str(int(r.n))), ("$b$", lambda k, r: str(int(r.b))),
        ("steps $T$", lambda k, r: str(int(r["T"]))), ("trajectories", lambda k, r: str(int(r.mc))),
        ("$\\gamma_T$", lambda k, r: g(r.gamma_end)), None,
        ("certified, worst functional", lambda k, r: f"{r.worst_c_emp_min:.3f}"),
        ("certified, domain mean", lambda k, r: f"{r.mean_c_emp_min:.3f}"),
        ("certified, far from sensors", lambda k, r: f"{r.unobs_c_emp_min:.3f}"),
        ("uncorrected, worst functional", lambda k, r: f"{r.worst_tr_emp_min:.3f}"),
        ("uncorrected, domain mean", lambda k, r: f"{r.mean_tr_emp_min:.3f}"),
        ("uncorrected, far from sensors", lambda k, r: f"{r.unobs_tr_emp_min:.3f}"), None,
        ("end of run, certified", lambda k, r: f"{z[k + '__worst_c'][-50:].mean():.3f}"),
        ("end of run, uncorrected", lambda k, r: f"{z[k + '__worst_tr'][-50:].mean():.3f}"),
        ("largest $|$empirical $-$ exact$|$", lambda k, r: f"{max(r.worst_maxdiff_c, r.mean_maxdiff_c, r.unobs_maxdiff_c):.3f}")]
out = ["\\begin{tabular}{lccccc}", "\\toprule", " & " + " & ".join(head) + " \\\\", "\\midrule"]
for row in rows:
    if row is None:
        out.append("\\midrule"); continue
    lab, f = row; out.append(lab + " & " + " & ".join(f(k, d.loc[k]) for k in cases) + " \\\\")
out += ["\\bottomrule", "\\end{tabular}"]
print("\n".join(out))
