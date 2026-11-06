#!/usr/bin/env bash
#SBATCH --job-name=botq-audit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
export POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK"
"$HOME/sf-quant-comb-alphas/.venv/bin/python" -u - <<'PY'
from datetime import date
from pathlib import Path
import polars as pl

root = Path.home() / "groups/grp_quant/database/production/assets"
for year in range(2005, 2026):
    paths = [root / f"assets_{y}.parquet" for y in range(year, min(year + 1, 2025) + 1)]
    assets = pl.scan_parquet(paths).filter(pl.col("date") <= date(min(year + 1, 2025), 1, 15)
                                          if year < 2025 else pl.lit(True))
    calendar = assets.select("date").unique().sort("date").with_columns(pl.col("date").shift(-1).alias("next"))
    members = assets.filter(pl.col("in_universe") & (pl.col("date").dt.year() == year))
    realized = assets.select(pl.col("date").alias("next"), "barrid", "return")
    data = members.select("date", "barrid", "predicted_beta").join(calendar, on="date").drop_nulls("next").join(
        realized, on=["next", "barrid"], how="left", validate="m:1")
    print(year, data.select(
        pl.col("next").max().alias("last_return_date"),
        pl.col("date").n_unique().alias("sessions"),
        pl.col("return").null_count().alias("missing_returns"),
        pl.col("predicted_beta").null_count().alias("missing_beta"),
    ).collect().to_dicts())
PY
