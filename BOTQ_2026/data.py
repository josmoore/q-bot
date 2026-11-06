"""Point-in-time Russell panel from the existing sf-quant asset partitions.

See the repository's BYU_ORC_AGENTS.md before running this on ORC. Keep large
scans inside a Slurm allocation; this module only constructs lazy Polars plans.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl


ASSET_COLUMNS = (
    "date",
    "barrid",
    "price",
    "return",
    "market_cap",
    "daily_volume",
    "predicted_beta",
    "specific_risk",
    "in_universe",
    "russell_1000",
    "russell_2000",
)


def asset_partition(root: Path, database: str, year: int) -> Path:
    """Resolve one annual partition without scanning a shared directory."""
    return (
        root
        / "groups"
        / "grp_quant"
        / "database"
        / database
        / "assets"
        / f"assets_{year}.parquet"
    )


def scan_assets(root: Path, database: str, start: date, end: date) -> pl.LazyFrame:
    """Scan dated rows, retaining delisted names and historical memberships.

    ``in_universe`` is the same-day membership field used by sf-quant's
    benchmark. It is not inferred from the current index membership.
    """
    if start > end:
        raise ValueError("start must be on or before end")
    partitions = [asset_partition(root, database, y) for y in range(start.year, end.year + 1)]
    missing = [p for p in partitions if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing annual asset partitions: {missing[:3]}")
    return (
        pl.scan_parquet([str(p) for p in partitions])
        .filter(pl.col("date").is_between(start, end))
        .select(ASSET_COLUMNS)
    )


def benchmark_weights(assets: pl.LazyFrame) -> pl.LazyFrame:
    """Dated Russell 1000 + 2000 market-cap weights, matching sf-quant."""
    return (
        assets.filter(pl.col("in_universe") & (pl.col("market_cap") > 0))
        .with_columns(
            (pl.col("market_cap") / pl.col("market_cap").sum().over("date"))
            .alias("benchmark_weight")
        )
        .select("date", "barrid", "benchmark_weight")
    )


def audit_one_date(assets: pl.LazyFrame, day: date) -> dict:
    """Small diagnostic; use in a Slurm job for a representative data check."""
    row = (
        assets.filter(pl.col("date") == day)
        .select(
            pl.len().alias("asset_rows"),
            pl.col("in_universe").sum().alias("russell_members"),
            (pl.col("in_universe") & pl.col("return").is_null())
            .sum()
            .alias("missing_member_returns"),
            (pl.col("in_universe") & pl.col("predicted_beta").is_null())
            .sum()
            .alias("missing_member_betas"),
        )
        .collect()
    )
    return row.row(0, named=True)
