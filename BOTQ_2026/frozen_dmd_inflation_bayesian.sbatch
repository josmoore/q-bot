#!/usr/bin/env bash
# Bayesian blend of the frozen-prior DMD/inflation ridge candidates and equal weight.
#SBATCH --job-name=botq-frozen-dmd-bma
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00

set -euo pipefail
cd "$HOME/q-bot"
cache="$HOME/nobackup/autodelete/botq2026/corrected-signals"
export PYTHONPATH="$HOME/libraries/sf-quant/src"
export POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python="$HOME/sf-quant-comb-alphas/.venv/bin/python"

"$python" -u -m BOTQ_2026.walk_forward \
  --assets "$cache/assets.parquet" \
  --forecasts "$cache/forecasts-through-2024.parquet" \
  --exits "$cache/exit_return_audit.parquet" \
  --out "$cache/results-frozen-dmd-inflation-bayesian" \
  --end-date 2024-12-31 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --sparse-solver \
  --centered-grid \
  --dmd-inflation \
  --include-equal \
  --candidate-returns "$cache/results-frozen-dmd-inflation-candidates/returns.parquet"
