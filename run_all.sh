#!/usr/bin/env bash
# Reproduce the paper.
#   ./run_all.sh quick   regenerate every table and figure from the stored results in results/ and run the
#                        theory checks (a few minutes)
#   ./run_all.sh full    rerun every experiment first (about two days on two cores), then do the quick step
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-quick}"
if [ "$mode" = "full" ]; then
  for a in "e1 240" "e1 600" "e1 1200" "e1x 240" "e1x 240b34" "e1x 600" "e1x b34" "e1x 600b34" "e1x 1200" "e1x 1200L1" "e1x 1200L2" \
           e3 e9 e11 e4w e5 e6 thm2 e8 e7 "e13 small" "e13 big2" e15; do
    python sppf_certified.py $a
  done
  python e12_verified_cost.py
  python e14_growth_rates.py
  python e16_misspec.py
  for w in s240 s600 h600 h1200 lownoise; do python e17_lowrank.py $w; done
  python e18_offline.py
  for w in s240 s600 h600 h1200 lownoise; do python e19_rank_rates.py $w; done
  python e20_large.py 30 20 3 30 200 check
  python e20_large.py 100 100 3 300 400
  python e20_large.py 100 100 5 300 400
  python e20_large.py 100 100 3 300 400 5,20
  python e20_large.py 100 100 1 300 400 5,20
  python e20_large.py 100 100 2 300 400 5,20
  python e35_exact_large.py 100 100 1 300 400 5,20
  python e35_exact_large.py 100 100 2 300 400 5,20
  python e35_exact_large.py 100 100 3 300 400 5,20
  python e35_exact_large.py 100 100 3 300 400 sharp
  python e37_supersol_large.py 3 300 20
  python e36_shape_online.py upwind 2
  python e36_shape_online.py cn 2
  python e21_enkf.py
  python e21_enkf.py layouts
  python e21_enkf.py layouts 1
  for w in s240 s600 h600 h1200; do python e22_design.py $w; done
  python e23_verify_lowrank_large.py 100 100 3 40
  python e23_verify_lowrank_large.py every
  python e25_offline_large.py 3 300 5,20
  python e25_offline_large.py 5 300
  python e25_offline_large.py 3 300 20 1.001
  python e25_offline_large.py 1 300
  python e25_offline_large.py 2 300 5 1.0 _r5
  python e25_offline_large.py 2 300 5 1.001 _r5
  python e26_variants.py cn
  python e26_variants.py tv
  python e38_adaptive_inflation.py
  python e27_shape.py s240a 1
  python e27_shape.py s600a 1
  python e28_alarm.py
  python e29_informative.py && python e29_informative.py unc
  python e30_shape_cn.py
  for K in '' 2 3 4; do python e31_taper.py taper$K; done
  python e17_lowrank.py t1200
  python e17_lowrank.py t1200b
  python e32_large_cn.py 100 100 3 200 400 20 50
  python e34_decay.py dense && python e34_decay.py large && python e34_decay.py b0 && python e34_decay.py sharp && python e34_decay.py cn
  python timing_repeat.py
fi
python verify_theory.py
python ex_two_state.py
python e33_rank_monotone.py example
python gen_tab3.py > table_main_body.tex
python gen_tab_mc.py > table_mc.tex
python gen_tab_decay.py > table_decay.tex
python gen_tab_e11.py > table_models_body.tex
python e22_rule.py
python e24_mc_bootstrap.py
python make_figures.py e2 e3 e4 e5 e6 e14 e17 e22
echo "done: tables in table_*.tex, figures in figures/"
