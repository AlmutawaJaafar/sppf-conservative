# Conservative, Calibrated Uncertainty in Sparse-Precision Kalman Filtering

Code and synthetic experiments for the paper

> **Conservative, Calibrated Uncertainty in Sparse-Precision Kalman Filtering of
> Advection–Diffusion Fields**, J. Almutawa.

The method keeps a *banded* (sparse) precision in a Kalman filter for
advection–diffusion fields and adds a computable, matrix-free covariance
**inflation**, driven by a local measure of the discarded precision mass, that
provably Loewner-dominates the true posterior — turning an overconfident
reduced-order filter into one with guaranteed over-coverage.

This repository reproduces every figure and table in the paper from synthetic
data. No external datasets are required.

## Installation

```bash
git clone https://github.com/<author>/sppf-conservative.git
cd sppf-conservative
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10. Dependencies: `numpy`, `scipy`, `matplotlib` (versions
pinned in `requirements.txt`). All outputs are written to `results/`.

## Reproducing the paper

Each experiment is a `mode` of `sppf_experiments.py` (set at the top of the file
in `CONFIG["mode"]`, or overridden as shown). Runtimes are approximate on a
laptop; the full calibration sweep is the only long one.

| Paper item | Command | Output(s) |
|---|---|---|
| Tables 1–2, Figs. 3–5 (calibration, solvers, regime map) | set `CONFIG["mode"]="full"` then `python3 sppf_experiments.py` | `table_main.csv`, `table_solvers.csv`, `fig_nees_reliability.png`, `fig_decay_overlay.png`, `fig_regime_map.png`, `fig_sensor_density.png` |
| Driver comparison (c\* scaling, Fig. `fig:sweeps`) | `CONFIG["mode"]="gtune"` | `fig_gtune_drivers.png`, `cstar_log.csv` |
| Bandwidth sweep (Fig. 1, C1/C3) | `CONFIG["mode"]="bandwidth"` | `fig_bandwidth_decay.png`, `fig_bandwidth_constants.png`, `bandwidth_log.csv` |
| Scaling to n=10⁵ (Table 3, Fig. 6) | `python3 sppf_scaling.py` | `fig_scaling_time.png`, `fig_scaling_mem.png`, `scaling_log.csv` |
| Model-mismatch test | `CONFIG["mode"]="mismatch"` | `mismatch_log.csv` |
| Localization head-to-head (Table 4, C4) | `CONFIG["mode"]="localization"` | `localization_log.csv` |

To override the mode without editing the file:

```bash
python3 -c "import sppf_experiments as S; S.CONFIG['mode']='localization'; S.main()"
```

Quick smoke test (a few minutes): `CONFIG["mode"]="small"`.

## Files

- `sppf_experiments.py` — main experiment suite (calibration, solvers, regime
  map, driver/bandwidth/mismatch/localization studies).
- `sppf_scaling.py` — sparse vs. dense per-step cost/timing to n=10⁵.
- `sppf_diagnostics.py` — lighter diagnostic harness (D1–D4).
- `results/` — figures and CSV logs produced by the scripts (committed copies
  match the paper).

## Notes on the numbers

- All NEES/n values reported in the paper are full-horizon averages (burn-in of
  6 steps) and therefore include the startup transient, where the truncated
  filter is most overconfident; steady-state overconfidence is milder. See §7 of
  the paper. The committed `CONFIG` values reproduce the published figures.
- The scaling study (`sppf_scaling.py`) times the per-step linear algebra on the
  true band-`b` grid sparsity pattern; it is a *cost* study, not a full
  calibrated run at n=10⁵.
- Random seed is fixed (`CONFIG["seed"]=0`); minor numerical differences across
  BLAS/OS are expected.

## Citation

```bibtex
@article{almutawa2026sppf,
  author  = {Almutawa, Jaafar},
  title   = {Conservative, Calibrated Uncertainty in Sparse-Precision Kalman
             Filtering of Advection--Diffusion Fields},
  year    = {2026},
  note    = {Preprint}
}
```

## License

Released under the MIT License (see `LICENSE`).
