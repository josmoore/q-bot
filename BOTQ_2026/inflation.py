"""Inflation-beta signal built from ALFRED vintage CPI and monthly CRSP returns.

Every CPI value must be selected from the vintage available on the signal
date. See README for the source and release-lag rules. Input arrays are dates
by stocks; all correlations are calculated from history strictly before the
signal date.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def inflation_beta_signal(
    monthly: pl.DataFrame,
    *,
    return_column: str = "monthly_return",
    inflation_column: str = "inflation_change",
    window_months: int = 60,
    min_observations: int = 36,
) -> pl.DataFrame:
    """Return a dated stock score equal to negative rolling inflation beta.

    The score has higher values for lower inflation sensitivity, following the
    negative inflation risk price motivating Duarte (2013). Each asset's
    beta uses its own valid history; cross-sectional ranks are formed later on
    the daily investable universe.
    """
    required = {"date", "barrid", return_column, inflation_column}
    if missing := required - set(monthly.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    monthly = monthly.sort(["barrid", "date"])
    return (
        monthly.with_columns(
            pl.col(return_column).rolling_cov(
                pl.col(inflation_column), window_size=window_months,
                min_periods=min_observations,
            ).over("barrid").alias("cov_ri"),
            pl.col(inflation_column).rolling_var(
                window_size=window_months, min_periods=min_observations,
            ).over("barrid").alias("var_inflation"),
        )
        .with_columns((-pl.col("cov_ri") / pl.col("var_inflation")).alias("inflation_beta_alpha"))
        .select("date", "barrid", "inflation_beta_alpha")
    )


def market_beta_residual(
    daily: pl.DataFrame,
    *,
    return_column: str = "return",
    benchmark_return_column: str = "benchmark_return",
    beta_column: str = "predicted_beta",
) -> pl.DataFrame:
    """Make next-close targets using point-in-time market beta and benchmark."""
    required = {"date", "barrid", return_column, benchmark_return_column, beta_column}
    if missing := required - set(daily.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return daily.with_columns(
        (pl.col(return_column) - pl.col(beta_column) * pl.col(benchmark_return_column))
        .shift(-1).over("barrid").alias("forward_residual_return"),
        pl.col("date").shift(-1).over("barrid").alias("label_available_date"),
    )


def _demeaned_rank(values: np.ndarray) -> np.ndarray:
    """Cross-sectional rank score with zero mean and unit standard deviation."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if valid.sum() < 2:
        return out
    ranks = np.argsort(np.argsort(values[valid])).astype(float)
    ranks = ranks / (len(ranks) - 1) - 0.5
    sd = ranks.std()
    out[valid] = (ranks - ranks.mean()) / (sd if sd > 0 else 1.0)
    return out
