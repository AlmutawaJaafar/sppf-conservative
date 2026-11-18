"""Supplement Table S2: the decay bound of Supplement Theorem S1 against the exact truncation error, and the explicit bandwidth."""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
d = json.load(open(os.path.join(R, "e34_decay_dense.json")))
L = {r["b"]: r for r in json.load(open(os.path.join(R, "e34_decay_large.json")))}
b0 = json.load(open(os.path.join(R, "e34_decay_b0.json")))
rows = []
for Pe in (10.0, 50.0, 90.0):
    for b in (2, 3, 4):
        a = [r for r in d if r["Pe"] == Pe and r["b"] == b]
        r240 = [r for r in a if r["n"] == 240][0]; r600 = [r for r in a if r["n"] == 600][0]
        big = f"{L[b]['bound']:.1f}" if (Pe == 50.0 and b in L) else "--"
        rows.append(f"{int(Pe)} & {b} & {r600['Dinf']:.3f} & {r240['bound']:.1f} & {r600['bound']:.1f} & {big} & "
                    f"{r600['ratio']:.0f} & {r600['alpha']:.2f} & {r600['omega']:.2f} \\\\")
    rows.append(r"\addlinespace")
print("\n".join(rows[:-1]))
print("% b0 (self-consistent, n=600):")
for Pe in (10.0, 50.0, 90.0):
    r = [x for x in b0 if x["Pe"] == Pe]
    bb = [x["b"] for x in r if x["eps_bound"] < x["ceta"]]; be = [x["b"] for x in r if x["eps_true"] < x["ceta"]]
    print(f"% Pe={Pe}: b0={bb[:1]}, exact-D condition from b={be[:1]}")
