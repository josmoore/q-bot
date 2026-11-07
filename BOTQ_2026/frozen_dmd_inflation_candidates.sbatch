#!/usr/bin/env bash
# Frozen non-zero-prior DMD/inflation candidate backtest.
# The ridge center freezes after the 252-session warm-up at its final
# maximum signed coefficient; the run also emits equal-weight, 95/5,
# residualized DMD, and matched-universe raw-DMD portfolios.
#SBATCH --job-name=botq-frozen-dmd-infl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

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
  --out "$cache/results-frozen-dmd-inflation-candidates" \
  --end-date 2024-12-31 \
  --workers "$SLURM_CPUS_PER_TASK" \
  --sparse-solver \
  --centered-grid \
  --dmd-inflation
