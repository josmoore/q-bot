"""Measure HMM forecast accuracy against the backtest's Russell 3000 returns.

On BYU, run from q-bot with the BOTQ Python environment:
    python -m BOTQ_2026.hmm_correlation ~/nobackup/autodelete/botq2026/corrected-signals

The forecast at close t targets the return from close t+1 to close t+2.
Use the asset trading calendar to align dates, then restrict the sample to
the backtest's realized benchmark returns.
"""

import argparse
from pathlib import Path

import polars as pl


def correlate(cache: Path) -> tuple[int, float]:
    calendar = (
        pl.scan_parquet(cache / "assets.parquet").select("date").unique().sort("date")
        .with_columns(pl.col("date").shift(2).alias("signal_date")).collect()
    )
    actual = (
        pl.read_parquet(cache / "results/returns.parquet")
        .select("date", "benchmark_return").unique()
        .join(calendar, on="date", validate="1:1")
    )
    forecasts = pl.read_parquet(cache / "hmm_scores.parquet").select(
        "signal_date", "expected_market_return"
    )
    paired = actual.join(forecasts, on="signal_date", how="left", validate="1:1")
    valid = paired.select(
        pl.all_horizontal(pl.col("benchmark_return", "expected_market_return").is_finite())
        .fill_null(False).all()
    ).item()
    if paired.height < 2 or not valid:
        raise ValueError("Expected a finite HMM forecast and benchmark return for every test session")
    correlation = paired.select(pl.corr("expected_market_return", "benchmark_return")).item()
    if correlation is None or not -1 <= correlation <= 1:
        raise ValueError("Pearson correlation is undefined for these forecasts or returns")
    return paired.height, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="Directory containing assets, HMM scores, and results")
    count, correlation = correlate(parser.parse_args().cache)
    print(f"Pearson correlation {correlation:.6f} across {count:,} trading sessions")
