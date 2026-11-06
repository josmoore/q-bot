#!/usr/bin/env bash
#SBATCH --job-name=botq-hmm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/q-bot"
cache="$HOME/nobackup/autodelete/botq2026"
export PYTHONPATH="$cache/hmm-deps"
export POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$HOME/sf-quant-comb-alphas/.venv/bin/python" -u -m BOTQ_2026.hmm_stage "$cache/hmm_scores.parquet"
