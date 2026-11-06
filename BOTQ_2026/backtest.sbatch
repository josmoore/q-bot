#!/usr/bin/env bash
#SBATCH --job-name=botq-backtest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
cd "$HOME/q-bot"
cache="${BOTQ_CACHE_DIR:-$HOME/nobackup/autodelete/botq2026}"
export PYTHONPATH="$HOME/libraries/sf-quant/src"
export POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python="$HOME/sf-quant-comb-alphas/.venv/bin/python"
if [[ "${1:-}" == "--validation-only" ]]; then
  "$python" -u -m BOTQ_2026.assemble_forecasts "$cache" --last-year 2009
  forecasts="$cache/forecasts-through-2009.parquet"
  output="$cache/validation"
else
  last_year="${BOTQ_FORECAST_LAST_YEAR:-2025}"
  "$python" -u -m BOTQ_2026.assemble_forecasts "$cache" --last-year "$last_year"
  if [[ "$last_year" == 2025 ]]; then
    forecasts="$cache/forecasts.parquet"
  else
    forecasts="$cache/forecasts-through-$last_year.parquet"
  fi
  output="${BOTQ_OUTPUT_DIR:-$cache/results}"
fi
"$python" -u -m BOTQ_2026.walk_forward --assets "$cache/assets.parquet" \
  --forecasts "$forecasts" --out "$output" --exits "$cache/exit_return_audit.parquet" "$@"
