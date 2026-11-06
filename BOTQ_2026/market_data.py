"""Dated equity and macro inputs used by BOTQ_2026.

Russell membership, market capitalization, returns, price, volume, beta and
specific risk come from the production sf-quant ``assets`` table. Daily open,
ask/high, bid/low and close come from the research CRSP daily table. Its CUSIP
is eight characters; the asset table's CUSIP is nine, so the dated join uses
the first eight characters and checks for duplicates before returning data.
The CRSP guide defines askhi/bidlo as daily high/low trade prices, falling back
to closing ask/bid when a trade price is unavailable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl


def load_daily_cross_section(
    root: Path,
    start: date,
    end: date,
    *,
    assets_database: str = "production",
    crsp_database: str = "research",
) -> pl.LazyFrame:
    """Return lazy point-in-time Russell rows with CRSP daily price features."""
    asset_paths = [
        root / "groups/grp_quant/database" / assets_database / "assets" / f"assets_{year}.parquet"
        for year in range(start.year, end.year + 1)
    ]
    crsp_paths = [
        root / "groups/grp_quant/database" / crsp_database / "crsp_daily" / f"crsp_daily_{year}.parquet"
        for year in range(start.year, end.year + 1)
    ]
    absent = [p for p in asset_paths + crsp_paths if not p.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing dated partitions: {absent[:4]}")
    assets = (
        pl.scan_parquet([str(p) for p in asset_paths])
        .filter(pl.col("date").is_between(start, end) & pl.col("in_universe"))
        .select("date", "barrid", "cusip", "price", "return", "market_cap", "daily_volume",
                "predicted_beta", "specific_risk", "in_universe")
        .with_columns(pl.col("cusip").str.slice(0, 8).alias("cusip8"))
    )
    crsp = (
        pl.scan_parquet([str(p) for p in crsp_paths])
        .filter(pl.col("date").is_between(start, end))
        .select("date", pl.col("cusip").alias("cusip8"), "openprc", "askhi", "bidlo", "prc")
    )
    joined = assets.join(crsp, on=["date", "cusip8"], how="left", validate="m:1")
    return joined.drop("cusip8")


def validate_cross_section(panel: pl.DataFrame, day: date) -> None:
    """Fail on broken identifiers or missing point-in-time risk inputs."""
    if panel["barrid"].n_unique() != panel.height:
        raise ValueError(f"{day}: duplicate Russell barrids after CRSP join")
    if panel["benchmark_weight"].null_count() if "benchmark_weight" in panel.columns else False:
        raise ValueError(f"{day}: null benchmark weights")
    missing = panel.select(
        pl.col("openprc").is_null().sum().alias("open"),
        pl.col("askhi").is_null().sum().alias("high"),
        pl.col("bidlo").is_null().sum().alias("low"),
    ).row(0, named=True)
    if any(missing.values()):
        raise ValueError(f"{day}: missing CRSP OHLC quote values {missing}")
