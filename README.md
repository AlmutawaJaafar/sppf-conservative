# Certified conservative covariance for sparse-precision Kalman filtering

Code and results for the manuscript *Provably Conservative Covariance for
Sparse-Precision Kalman Filters, with Application to Advection–Diffusion* (J. Almutawa). Columns and variables named `min` (e.g. `price_min`, `worst_min`) refer to
the certificate the paper now calls the *basic* certificate (formerly "minimal").

[![reproduce](https://github.com/AlmutawaJaafar/sppf-conservative/actions/workflows/reproduce.yml/badge.svg)](https://github.com/AlmutawaJaafar/sppf-conservative/actions/workflows/reproduce.yml)

`sppf_certified.py` implements Algorithm 1 of the paper line by line and
evaluates it with the **exact** error second moments (no Monte Carlo error),
plus a Monte Carlo cross-check. `make_figures.py` produces every figure and the
table data of Section 5 from `results/`.

> Version note: this release (v2.0) replaces the scripts of the earlier
> submission (`sppf_experiments.py`, `sppf_diagnostics.py`, `sppf_scaling.py`),
> which implemented a different, superseded method.

## Requirements

Python 3.10+ with NumPy, SciPy, Matplotlib and pandas (tested with Python 3.13,
NumPy 2.5, SciPy 1.18, Matplotlib 3.11, pandas 3.0).

```
pip install -r requirements.txt   # numpy, scipy, matplotlib, pandas (pinned versions used for the paper)
./run_all.sh quick   # regenerate every table and figure from results/ and run the theory checks
./run_all.sh full    # rerun every experiment first (about two days on two cores)
```

## Reproducing the paper

| Command | Paper item | Output | Approx. time (2 cores) |
|---|---|---|---|
| `python sppf_certified.py e1 240` | Sec. 5.2 (initial T=150 runs) | `results/e1_n240.csv` | 15 min |
| `python sppf_certified.py e1 600` | Sec. 5.2 (initial T=150 runs) | `results/e1_n600.csv` | 40 min |
| `python sppf_certified.py e1 1200` | Sec. 5.2 (initial T=150 runs) | `results/e1_n1200.csv` | 35 min |
| `python sppf_certified.py e1x 240` / `e1x 240b34` | Sec. 5.2, Table 2 (all runs repeated with T=600) | `results/e1x_T600_n240.csv`, `results/e1x_T600_n240b34.csv` | 25 min |
| `python sppf_certified.py e1x 600` / `e1x b34` / `e1x 600b34` | Sec. 5.2, Table 2 (T=600) | `results/e1x_T600_n600.csv`, `results/e1x_T600_nb34.csv`, `results/e1x_T600_n600b34.csv` | 3 h |
| `python sppf_certified.py e1x 1200` | Sec. 5.2, Table 2 (T=600) | `results/e1x_T600_n1200.csv` | 70 min |
| `python sppf_certified.py e1x 1200L1` / `e1x 1200L2` | Sec. 5.2, Table 2 (two further sensor layouts at n=1200, T=600) | `results/e1x_T600_n1200L1.csv`, `results/e1x_T600_n1200L2.csv` | 70 min each |
| `python sppf_certified.py e3` | Sec. 5.3, Fig. 2 (coverage, Monte Carlo) | `results/e3_series.npz` | 2 min |
| `python sppf_certified.py e9` | Sec. 5.3, Suppl. Table S7 (independent Monte Carlo in hard regimes: state simulation) | `results/e9_montecarlo.csv`, `results/e9_series.npz` | 40 min |
| `python sppf_certified.py e11` | Sec. 5.3, Suppl. Table S5 (correlated process noise; heterogeneous diffusion with shear) | `results/e11_model_classes.csv` | 90 min |
| `python sppf_certified.py e4w` | Suppl. Sec. S3, Numerical evidence (decay along the sparse filter) | `results/e4w_decay_working.csv` | 10 min |
| `python sppf_certified.py e5` | Suppl. Sec. S8.2 (cost study; the randomised rows are no longer in the paper) | `results/e5_cost.csv`, `results/e5_lanczos_accuracy.csv` | 10 min |
| `python sppf_certified.py e6` | Sec. 5.4, Suppl. Sec. S10 (low process noise; worst-functional coverage per bandwidth) | `results/e6_series.npz` | 1 min |
| `python sppf_certified.py thm2` | Sec. 5.2, Theorem 2 bound vs observed | `results/thm2_bound_check.csv` | 1 min |
| `python sppf_certified.py e8` | Sec. 5.2, downward-shift (conservative) construction | `results/e8_conservative_baseline.csv` | 5 min |
| `python sppf_certified.py e7` | randomised certificate in closed loop (no longer in the paper; removed with former Suppl. Sec. S4) | `results/e7_randomised_closed_loop.csv` | 60 min |
| `python e12_verified_cost.py` | Sec. 4.3, Suppl. Sec. S8.2 (verified certificate: k=40 Lanczos + one sparse LDL' test of the block matrix of Lemma 3(b)) | `results/e12_verified_cost.csv` | 5 min |
| `python sppf_certified.py e13 small` / `e13 big2` | Sec. 5.2 (sharp certificate of Theorem 1(b) vs basic certificate) | `results/e13_sharp_small.csv`, `results/e13_sharp_big.csv` | 2 h / 1.5 h |
| `python sppf_certified.py e15` | Sec. 5.5 (warm-started estimate-then-verify certificate in closed loop: tests per step, gamma vs basic) | `results/e15_predict_verify.csv` | 1 h |
| `python e14_growth_rates.py` | Sec. 5.2, Suppl. Sec. S7.1, Suppl. Eq. (S1) and Theorem 3 (lower and upper growth rates nu, nu#, u(1-eta) along every T=600 run) | `results/e14_growth_rates.csv` | 1.5 h |
| `python e16_misspec.py` | Sec. 5.3, misspecified noise covariances (filter run with c_Q Q or c_R R) | `results/e16_misspec.csv` | 10 min |
| `python e17_lowrank.py s240` / `s600` / `h600` / `h1200` | Sec. 5.4, Table 2 (scalar-plus-low-rank certificates, Lemma 1 and Corollary 3; basic, sharp, rank 5, rank 20) | `results/e17_lowrank_s240.csv`, `results/e17_lowrank_s600.csv`, `results/e17_lowrank_hard.csv` (concatenation of the `h600` and `h1200` outputs; the separate `e17_lowrank_h600.csv`, `e17_lowrank_h1200_*.csv` are also stored) | 40 min / 50 min / 20 min / 2.5 h |
| `python e18_offline.py` | offline steady-state basic certificate vs online certificate (no longer in the paper) | `results/e18_offline.csv` | 5 min |
| `python e17_lowrank.py lownoise` | Sec. 5.4 (low-rank certificates with weak process noise, n=240, sigma_w=0.01, T=150) | `results/e17_lowrank_lownoise.csv` | 2 min |
| `python e19_rank_rates.py s240` / `s600` / `h600` / `h1200` / `lownoise` | Sec. 5.4, Suppl. Proposition S2 (rank diagnostic: rates nu#_(i) for every E17 run) | `results/e19_rank_rates_*.csv`, combined in `results/e19_rank_rates.csv` | 15 / 10 / 3 / 25 / 1 min |
| `python e20_large.py 30 20 3 30 200 check` | Sec. 4.3, Suppl. Sec. S8.1 (probed S_t against the exact B_b(P^{-1}) on a 30x20 grid) | `results/e20_large_n600_b3.json` | 1 min |
| `python e20_large.py 100 100 3 300 400` / `... 100 100 5 300 400` / `... 100 100 3 300 400 5,20` | Sec. 5.5 (end to end at n=10^4: Woodbury + probing S_t, verified scalar certificate, low-rank certificates p=5,20, 400 Monte Carlo trajectories) | `results/e20_large_n10000_b3.json`, `..._b5.json`, `..._b3_r5-20.json` | 15 / 35 / 40 min |
| `python e20_large.py 100 100 1 300 400 5,20` / `... 100 100 2 300 400 5,20` | Sec. 5.5, Table 3 (the failing large case: at b=1 the uncorrected sparse filter at n=10^4 under-covers the domain mean; scalar, rank-5 and rank-20 certificates; b=2 for comparison) | `results/e20_large_n10000_b1_r5-20.json`, `results/e20_large_n10000_b2_r5-20.json`, logs `results/e20_n10000_b{1,2}_lr.log` | 40-60 min each |
| `python e21_enkf.py` | Sec. 5.6 (tuning grid of the localized EnKF with RTPS inflation, layout 0) | `results/e21_enkf.csv` | 60 min |
| `python e21_enkf.py layouts` | Sec. 5.6, Suppl. Table S6 (tuned EnKFs on three sensor layouts; localized and raw-ensemble variances) | `results/e21_enkf_layouts.csv` | 40 min |
| `python e21_enkf.py layouts 1` | Sec. 5.6, Suppl. Table S6 (sparse filter at b=1, scalar and rank-20 certified, against the same EnKFs) | `results/e21_enkf_layouts_b1.csv` | 45 min |
| `python e22_design.py s240` / `s600` / `h600` / `h1200` | Sec. 5.4, Suppl. Sec. S7.2 (offline design; its scatter figure is no longer in the paper: data-free steady state of the sharp and rank-p certificates, predicted price, rates nu#_(i), for every E17 configuration) | `results/e22_design_*.csv` | 25 / 15 / 5 / 30 min |
| `python e22_rule.py` | Sec. 5.4, Suppl. Sec. S7.2 (price-targeted design rule; the former table is now one paragraph) | stdout, `results/e22_rule_rows.tex` | seconds |
| `python e23_verify_lowrank_large.py 100 100 3 40` | Sec. 5.5 (sparse bordered-matrix inertia test of the rank-5 and rank-20 certificates at n=10^4, with a lowered-scalar control) | `results/e23_verify_lowrank_large.json` | 25 min |
| `python e24_mc_bootstrap.py` | Suppl. Sec. S10 (count of certified Monte Carlo estimates below 0.95 against sampling error with temporally correlated estimates, AR(1) simulation) | stdout | seconds |
| `python e25_offline_large.py 3 300 5,20` / `... 5 300` / `... 3 300 20 1.001` / `... 1 300` / `... 2 300 20 1.001` / `... 1 300 20 1.001` | Sec. 5.5 (Algorithm 2 matrix-free at n=10^4: rates nu, nu#_(i), sharp steady state, rank-p steady states; the last with the closed-loop safety factor) | `results/e25_offline_large_b3.json`, `..._b5.json`, `..._b1.json`, `..._b3_safe.json`, `..._b2_safe.json`, `..._b1_safe.json` | 30 / 15 / 20 / 13 / 16 min |
| `python e26_variants.py cn` / `tv` | Sec. 5.3 (Crank--Nicolson with central differences; periodically varying velocity; basic, sharp, rank-5, rank-20 certificates with exact moments, and the offline (periodic) prediction) | `results/e26_cn.csv`, `results/e26_tv.csv` | 15 / 15 min |
| `python e27_shape.py s240a` / `s600a` | Sec. 4.4 and Suppl. Proposition S1 (example), Sec. 5.4 (rates of fixed scalar-plus-rank-p shapes, super-solution test, guaranteed steady-state price; Theorem 3, Theorem 3(v)) | `results/e27_shape_*.csv` | 60 / 40 min |
| `python e28_alarm.py` | not in the paper (exact miss and false-alarm rates of per-node threshold alarms; removed from the supplement) | `results/e28_alarm.csv` | 10 min |
| `python e29_informative.py` (then `... unc`) | Sec. 5.2 (informative data: sigma_v=0.1, m in {32,64}; certificates, prices, offline prediction, nodal variance reduction, uncorrected worst ratio) | `results/e29_informative.csv` | 25 min |
| `python e30_shape_cn.py` | Sec. 5.3 (fixed-shape certificates for the Crank--Nicolson runs: rates, super-solution test, guaranteed price) | `results/e30_shape_cn.csv` | 25 min |
| `python e31_taper.py taper` / `taper2` / `taper3` / `taper4` | Sec. 5.6 (Gaspari--Cohn tapered precision, support radius K(b+1/2), K=1..4, as working precision: uncorrected worst ratio, certificates, prices) | `results/e31_taper*.csv` | 10--40 min each |
| `python e32_large_cn.py 100 100 3 200 400 20 50` | Sec. 5.5 (Crank--Nicolson at n=10^4, fully sparse: probing with block PCG, scalar certificate verified by a 3n-order inertia test, rank-20 certificate; Monte Carlo coverage) | `results/e32_cn_n10000_b3_Pe50.json`, log `results/e32_cn_b3_Pe50.log` | 1-2 h |
| `python e33_rank_monotone.py example` / `random` | Sec. 4.4, Suppl. Sec. S2, monotonicity paragraph (rank-2 certificate above rank-1 for one functional after two steps; total variance after a change of coordinates) and the random-model check | stdout | seconds / 10 min |
| `python e34_decay.py dense` / `large` / `b0` / `sharp` / `cn` | Sec. 4.2, Suppl. Theorem S1, Sec. S3 and Table S2 (computable form (iii) against the exact u_t; Crank--Nicolson; decay bound vs exact truncation error at n=240, 600; omega and bound at n=10^4; closed-form bandwidth) | `results/e34_decay_dense.json`, `..._large.json`, `..._b0.json`, `..._sharp.json`, `..._cn.json` | 10 min / 5 min / 1 h |
| `python e35_exact_large.py 100 100 1 300 400 5,20` / `... 2 ...` / `... 3 ...` | Sec. 5.5, Table 3 (exact check of the n=10^4 runs: dense Kalman filter and exact error covariance of the sparse filter propagated alongside the sparse pipeline, same seed and sensors; exact worst-direction ratio, exact coverage and price against the Kalman filter every 50 steps; dense moments in single precision) | `results/e35_exact_n10000_b{1,2,3}_r5-20.json`, logs `results/e35_n10000_b*.log` | about 2 h each |
| `python e35_exact_large.py 100 100 3 300 400 sharp` | Sec. 5.5, Table 3 (same run with the sharp scalar certificate, Theorem 1(a), verified at every step by the sparse test of Lemma 3(a) with gamma Y_t - J) | `results/e35_exact_n10000_b3_sharp.json` | about 2 h |
| `python e36_shape_online.py upwind` / `cn` | Sec. 5.4 (fixed-shape certificate h_t Psi of Theorem 3 run online from t=0 with Psi from Algorithm 2: exact worst-direction ratio, price against the adaptive rank-p certificate) | `results/e36_shape_online_upwind.csv`, `results/e36_shape_online_cn.csv` | 1 h |
| `python e37_supersol_large.py 3 300 20` | Sec. 5.5, Suppl. Sec. S8.1 (rank-20 super-solution at n=10^4, b=3: fixed point of the rank-p map, bordered test at scalings 1.001, 1.0001, 0.999; steady-state price bound of the fixed shape) | `results/e37_supersol_large_b3.json`, `results/e37_supersol_large_b3.log` | 9 min |
| `python e38_adaptive_inflation.py` | Sec. 5.6, Suppl. Sec. S4 (innovation-based adaptive inflation of the reported covariance; Monte Carlo, n=240) | `results/e38_adaptive_inflation.csv` | 5 min |
| `python e17_lowrank.py t1200` / `t1200b` | Table 2 (sharp and rank-20 prices for the settled n=1200 cells, one layout) | `results/e17_lowrank_t1200.csv`, `results/e17_lowrank_t1200b.csv` | 1 h |
| `python -c "import e23_verify_lowrank_large as e; e.main(100,100,3,300,check_every=100)"` | Sec. 5.5 (bordered test at t=100, 200, 300 of the 300-step run) | `results/e23_verify_lowrank_large_T300.json` | 60 min |
| `python e23_verify_lowrank_large.py every` | Sec. 5.5, Suppl. Sec. S8.1 (rank-20 certificate of the reported design, n=10^4, b=3, verified by the bordered test at every one of the 300 steps; control with g lowered by 0.1% every 50 steps) | `results/e23_verify_lowrank_large_every_b3.json`, `results/e23_every_b3.log` | 90 min |
| `python ex_two_state.py` | Suppl. Example S1 (two-state system: basic certificate diverges, sharp certificate bounded) | stdout | seconds |
| `python verify_theory.py` | Numerical checks of Lemma 3, Theorem 1(b), the sharp bracket of Theorem 3(ii) and the Suppl. Sec. S2 bordered-matrix test of low-rank certificates (400 random cases and along a rank-5 certificate) | stdout | 3 min |
| `python timing_repeat.py` | Sec. 5.5 (timing variability at n=99,904, three repetitions) | `results/e5_timing_repeat.csv` | 3 min |
| `python gen_tab3.py` | Table 2 body (LaTeX rows, including the sharp and rank-20 prices over the 183 low-rank runs) | stdout | seconds |
| `python gen_tab_mc.py` | Suppl. Table S7 (complete tabular) | stdout | seconds |
| `python gen_tab_decay.py` | Suppl. Table S2 body (decay bound vs exact truncation error) | stdout | seconds |
| `python gen_tab_e11.py` | Suppl. Table S5 body | stdout | seconds |
| `python gen_tab_large.py` | Table 3 body (n=10^4: Monte Carlo from E20, exact coverage and prices from E35) | stdout | seconds |
| `python make_figures.py e2 e3 e4 e5 e6 e14 e17 e22` | main Fig. 2 (e2, e4, e5, e6, e14, e17 and e22 make figures no longer in the paper) (Fig. 1 is a TikZ schematic in the manuscript), table summary (Table 2 and `table_e1.txt` use the T=600 runs when present; Fig. 2 uses the T=150 coverage run) | `figures/*.pdf`, `figures/table_e1.txt` | seconds |

`python sppf_certified.py e4` reproduces an earlier proxy computation (decay
along the exact Kalman filter) that is kept for reference but not used in the
paper.

Set `OMP_NUM_THREADS=1` when running the sweeps; they parallelise over two
processes.

## Integrity of the stored results

`results/SHA256SUMS` lists the SHA-256 checksum of every stored result file. Verify with

```
sha256sum -c results/SHA256SUMS
```

The GitHub Actions workflow `.github/workflows/reproduce.yml` checks these checksums and runs
`./run_all.sh quick` on every push (status badge above). `make_checksums.sh` regenerates the list.

## Results and the scripts that check them

| Result in the paper | Checked by | Kind of check |
|---|---|---|
| Lemma 1 (certified covariance sequences), Theorem 1 (scalar certificates), Corollary 1 | every experiment script (worst-direction ratio `worst_*` <= 1 from exact error moments); `sppf_certified.py e3`/`e9`, `e20_large.py` (Monte Carlo), `e35_exact_large.py` (exact moments and exact Kalman filter at n=10^4) | exact moments; independent simulation |
| Lemma 3 (sparse matrix test), Theorem 1(b) (sharp certificate and its test) | `verify_theory.py` (checks 1A, 1B, optimality) | identity on random and model matrices |
| Oracle factor (Sec. 5.1), slack | `verify_theory.py` (last check), `sppf_certified.py e1x` (oracle column of Table 2) | exact moments |
| Theorem 2 (size, including the former Corollary 2) | `verify_theory.py` (check 1C), `sppf_certified.py e1x` | inequality on model runs |
| Suppl. Theorem S1 (decay of the working precision; a posteriori check, summarised in Sec. 4.2) | `e34_decay.py` | bound >= exact truncation error in every case; mesh independence |
| Suppl. Eq. (S1) and Theorem 3 (growth rates), Suppl. Example S1 | `ex_two_state.py`, `e14_growth_rates.py`, `e19_rank_rates.py`, `e22_design.py`, `e25_offline_large.py` | rates against observed growth |
| Lemma 1, Corollary 3 (low rank), Suppl. Proposition S2 (rank rates), Suppl. Corollary S1 (rank p never above sharp), bordered test | `verify_theory.py` (bordered test vs direct check; last check for Suppl. Corollary S1), `e17_lowrank.py`, `e23_verify_lowrank_large.py` | exact moments; sparse inertia test at n=10^4 |
| Theorem 3, Corollary 2 (frozen limit, super-solutions), Suppl. Proposition S1 (fixed shapes) | `e27_shape.py`, `e30_shape_cn.py`, `e36_shape_online.py` (fixed shape run online) | rates, super-solution test, exact worst-direction ratio of the online fixed shape |
| Algorithm 2 and the design rule | `e22_design.py`, `e22_rule.py`, `e25_offline_large.py` | prediction against closed loop |

Tests of mathematical identities (`verify_theory.py`, `ex_two_state.py`) are separate from tests of the practical sparse
implementation (`e20_large.py`, `e23_verify_lowrank_large.py`, `e25_offline_large.py`). All other experiments use the dense
reference implementation and exact error moments; Monte Carlo is used in `sppf_certified.py e3`/`e9`, `e20_large.py` and `e21_enkf.py`,
and `e35_exact_large.py` adds exact moments and the exact Kalman filter to the n=10^4 runs of `e20_large.py`.

## What is computed exactly, and what is practical

- The certificate in the experiments is computed by a dense symmetric
  eigensolver with a relative safety factor of 1e-10 (except `e7`, which uses
  a randomised Lanczos certificate inside the filter loop; no longer in the paper). This is a practical
  numerical value, not a verified finite-precision upper bound (Sec. 4.3).
- The sparse prior precision S_t is formed from a dense inverse, which isolates
  the effect of sparsification on uncertainty but is not scalable. The cost
  study (`e5`, `e12_verified_cost.py`) times the sparse update and the verified certificate
  on the sparsity pattern only. Factorisations use SciPy's SuperLU (an LU
  solver) in symmetric mode as a stand-in for sparse Cholesky.

## License

Released under the MIT License (see `LICENSE`).

## Changes in this release

- `SpChol` (sparse LDL' factor used only in the cost study) applied its fill-reducing permutation in the
  wrong direction, so the Lanczos operator in `e5` was a permuted rather than the intended matrix. The
  operations performed, and therefore all reported timings, are unaffected, and no reported eigenvalue
  comes from that harness; the convention is now fixed and tested (`L_S L_S' = S` to rounding error), and
  SuperLU equilibration is disabled so that the pivot signs give the inertia.
