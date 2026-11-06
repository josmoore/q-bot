#!/usr/bin/env bash
#SBATCH --job-name=botq-assets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/q-bot"
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export BOTQ_CACHE="$HOME/nobackup/autodelete/botq2026"
mkdir -p "$BOTQ_CACHE"
"$HOME/sf-quant-comb-alphas/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import polars as pl

root = Path.home() / "groups/grp_quant/database/production/assets"
out = Path(os.environ["BOTQ_CACHE"]) / "assets.parquet"
sources = [root / f"assets_{year}.parquet" for year in range(1995, 2026)]
# Full daily history preserves warm-up rows and returns for names leaving the index.
pl.scan_parquet(sources).select(
    "date", "barrid", "in_universe", "market_cap", "return", "predicted_beta",
    "price", "daily_volume", "cusip",
).sink_parquet(out.with_suffix(".tmp"), compression="zstd")
out.with_suffix(".tmp").replace(out)
print(out, flush=True)
PY
