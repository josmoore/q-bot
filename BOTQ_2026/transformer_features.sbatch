#!/usr/bin/env bash
#SBATCH --job-name=botq-features
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "$HOME/q-bot"
export POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK"
"$HOME/sf-quant-comb-alphas/.venv/bin/python" -u - <<'PY'
from datetime import date
from pathlib import Path
import polars as pl
from BOTQ_2026.market_data import load_daily_cross_section
from BOTQ_2026.transformer import (
    STOCK_FEATURES, MARKET_FEATURES, build_stock_features,
    build_market_features, cross_sectional_zscores,
)

root = Path.home()
cache = root / "nobackup/autodelete/botq2026"
market = build_market_features(pl.read_parquet(cache / "yahoo_market.parquet"))
for year in range(2000, 2026):
    output = cache / f"transformer-features-{year}.parquet"
    if output.exists():
        continue
    source = load_daily_cross_section(root, date(year - 2, 1, 1), date(year, 12, 31))
    panel = build_stock_features(source).filter(pl.col("date").dt.year() == year)
    panel = cross_sectional_zscores(panel.select("date", "barrid", *STOCK_FEATURES).collect())
    panel = panel.join(market, on="date", how="left").with_columns(
        pl.col(*STOCK_FEATURES, *MARKET_FEATURES).cast(pl.Float32)
    )
    temp = output.with_suffix(".tmp")
    panel.write_parquet(temp)
    temp.replace(output)
    print(year, panel.height, flush=True)
PY
