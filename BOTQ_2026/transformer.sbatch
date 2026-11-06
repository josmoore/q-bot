#!/usr/bin/env bash
#SBATCH --job-name=botq-transformer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=2005-2025%4
set -euo pipefail
cd "$HOME/q-bot"
cache="$HOME/nobackup/autodelete/botq2026"
export PYTHONPATH="$cache/torch-deps"
export POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS=1
"$HOME/sf-quant-comb-alphas/.venv/bin/python" -u -m BOTQ_2026.transformer_stage \
  "$cache" "$SLURM_ARRAY_TASK_ID" "$@"
