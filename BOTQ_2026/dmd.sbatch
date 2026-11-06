#!/usr/bin/env bash
#SBATCH --job-name=botq-dmd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-127%12
set -euo pipefail
cd "$HOME/q-bot"
export POLARS_MAX_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$HOME/sf-quant-comb-alphas/.venv/bin/python" -u -m BOTQ_2026.dmd_stage \
  "${BOTQ_CACHE_DIR:-$HOME/nobackup/autodelete/botq2026}" "$SLURM_ARRAY_TASK_ID" 128
