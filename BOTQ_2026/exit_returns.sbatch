#!/usr/bin/env bash
#SBATCH --job-name=botq-exits
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

root = Path.home() / "groups/grp_quant/database"
cache = Path.home() / "nobackup/autodelete/botq2026"
delist = pl.scan_parquet(root / "production/crsp_delisting/crsp_delisting.parquet")
rows = []
for year in range(2005, 2026):
    paths = [root / "production/assets" / f"assets_{y}.parquet"
             for y in range(year, min(year + 1, 2025) + 1)]
    assets = pl.scan_parquet(paths)
    if year < 2025:
        assets = assets.filter(pl.col("date") <= date(year + 1, 1, 15))
    calendar = assets.select("date").unique().sort("date").with_columns(
        pl.col("date").shift(-1).alias("label_available_date"))
    missing = (
        assets.filter(pl.col("in_universe") & (pl.col("date").dt.year() == year))
        .select("date", "barrid", pl.col("cusip").str.slice(0, 8).alias("cusip"))
        .join(calendar, on="date").drop_nulls("label_available_date")
        .join(assets.select(pl.col("date").alias("label_available_date"), "barrid", "return"),
              on=["label_available_date", "barrid"], how="left")
        .filter(pl.col("return").is_null())
    )
    crsp = pl.scan_parquet(root / "research/crsp_daily" / f"crsp_daily_{year}.parquet")
    mapping = crsp.select("date", "cusip", "permno").drop_nulls().unique().group_by(
        "date", "cusip").agg(pl.col("permno").first(), pl.len().alias("count")).filter(pl.col("count") == 1)
    return_paths = [root / "research/crsp_daily" / f"crsp_daily_{y}.parquet"
                    for y in range(year, min(year + 1, 2025) + 1)]
    crsp_returns = pl.scan_parquet(return_paths).select(
        pl.col("date").alias("label_available_date"), "permno",
        pl.when(pl.col("ret").is_finite() & (pl.col("ret") >= -1))
        .then(pl.col("ret")).alias("crsp_return"),
    )
    matched = (
        missing.join(mapping, on=["date", "cusip"], how="left")
        .join(delist.select("permno", pl.col("delistingdt").alias("label_available_date"), "delret"),
              on=["permno", "label_available_date"], how="left", validate="m:1")
        .join(crsp_returns, on=["permno", "label_available_date"], how="left", validate="m:1")
        .select("date", "label_available_date", "barrid", "permno", "delret", "crsp_return").collect()
    )
    rows.append(matched)
    print(year, matched.height, matched["permno"].null_count(), matched["delret"].is_not_null().sum(),
          matched["crsp_return"].is_not_null().sum())
pl.concat(rows).write_parquet(cache / "exit_return_audit.parquet")
PY
