#!/usr/bin/env bash
#SBATCH --job-name=botq-inflation
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/q-bot"
export POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK"
"$HOME/sf-quant-comb-alphas/.venv/bin/python" -u -m BOTQ_2026.inflation_stage \
  "${BOTQ_CACHE_DIR:-$HOME/nobackup/autodelete/botq2026}"
