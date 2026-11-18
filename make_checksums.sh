#!/usr/bin/env bash
# Regenerate results/SHA256SUMS after rerunning experiments.
set -euo pipefail
cd "$(dirname "$0")"
( cd results && sha256sum $(ls | grep -v SHA256SUMS | sort) ) > results/SHA256SUMS.tmp
sed 's#  #  results/#' results/SHA256SUMS.tmp > results/SHA256SUMS && rm results/SHA256SUMS.tmp
echo "wrote results/SHA256SUMS"
