"""E30: fixed-shape certificates (Theorem 3, including part (v)) for the Crank--Nicolson runs of E26.

Same computation as e27_shape.py, with the propagator of e26_variants.py ('cn'): for each stable configuration the
rank-p map is iterated with S, Y frozen to a candidate shape, and its rate r, the super-solution test and the
guaranteed steady-state price are reported.
usage: python e30_shape_cn.py  ->  results/e30_shape_cn.csv
"""
import sys, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, ".")
import sppf_certified as sc, e27_shape as e27, e26_variants as e26

class CNModel(sc.Model):
    def __init__(self, Nx, Ny, Pe=50.0, **kw):
        super().__init__(Nx, Ny, Pe=Pe, **kw); self.F = e26.propagators("cn", self, Pe)[0](0)

e27.Model = CNModel
if __name__ == "__main__":
    st = pd.read_csv("results/e26_cn.csv"); st = st[st.failed == False]
    jobs = [(240, 0.05, r.Pe, int(r.m), int(r.b), 0, 600, (0, 1, 5, 20)) for r in st.itertuples()]
    with Pool(1) as p: out = [r for rows in p.map(e27.job, jobs) for r in rows]
    d = pd.DataFrame(out); d.to_csv("results/e30_shape_cn.csv", index=False)
    print(d.to_string(float_format=lambda x: f"{x:.5g}"))
