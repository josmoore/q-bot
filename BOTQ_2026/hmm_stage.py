"""Adapt q-bot's causal two-state HMM into a daily Russell 3000 forecast."""

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import polars as pl

from BOTQ_2026.market_data import load_daily_cross_section
from HiddenMarkovModel.market_data import fit_asof_hmm_regimes


def market_history(root: Path, cache: Path) -> pl.DataFrame:
    """Aggregate q-bot's HMM features across dated Russell constituents."""
    panels = []
    for year in range(2000, 2025):
        panel = load_daily_cross_section(root, date(year, 1, 1), date(year, 12, 31))
        daily = pl.col("return") / 100
        stock_return = pl.when(daily.is_finite() & (daily > -1)).then(daily.log1p())
        valid_range = (pl.col("askhi") >= pl.col("bidlo")) & (pl.col("bidlo") > 0)
        valid_open = (pl.col("openprc") > 0) & (pl.col("prc").abs() > 0)
        # Scale today's open/close by the close-to-close gross return to
        # approximate the adjusted overnight gap, including distributions.
        gap = (pl.col("openprc") / pl.col("prc").abs()).log() + stock_return
        panels.append(panel.group_by("date").agg(
            stock_return.mean().alias("market_return"),
            stock_return.std().alias("dispersion"),
            pl.when(valid_range).then((pl.col("askhi") / pl.col("bidlo")).log()).mean().alias("range"),
            pl.when(valid_open).then(gap).mean().alias("opening_gap"),
        ).collect())
    features = pl.concat(panels).sort("date")
    calendar = features.select("date").with_columns(pl.col("date").shift(-1).alias("return_date")).lazy()
    assets = pl.scan_parquet(cache / "assets.parquet").filter(
        pl.col("date").is_between(date(2000, 1, 1), date(2024, 12, 31))
    )
    fallback = pl.scan_parquet(cache / "exit_return_audit.parquet").select(
        pl.col("label_available_date").alias("return_date"), "barrid", "delret", "crsp_return"
    )
    benchmark_years = []
    for year in range(2000, 2025):
        benchmark_years.append((
            assets.filter(pl.col("in_universe") & (pl.col("date").dt.year() == year))
            .select("date", "barrid", "market_cap")
            .join(calendar, on="date").drop_nulls("return_date")
            .join(assets.filter(pl.col("date").dt.year().is_between(year, year + 1)).select(
                pl.col("date").alias("return_date"), "barrid", (pl.col("return") / 100).alias("return")
            ), on=["return_date", "barrid"], how="left", validate="m:1")
            .join(fallback.filter(pl.col("return_date").dt.year().is_between(year, year + 1)), on=["return_date", "barrid"], how="left", validate="m:1")
            .group_by("return_date").agg(
                ((pl.col("market_cap") * pl.coalesce("return", "delret", "crsp_return").fill_null(0)).sum() /
                 pl.col("market_cap").sum()).alias("benchmark_return")
            ).collect()
        ))
    benchmark = pl.concat(benchmark_years)
    return features.join(benchmark, left_on="date", right_on="return_date", how="left").sort("date")


def forecast(history: pl.DataFrame) -> pl.DataFrame:
    """Calibrate filtered state probabilities to the return earned after next-close execution."""
    market = history.to_pandas().set_index("date")
    market.index = pd.DatetimeIndex(market.index)
    features = pd.DataFrame({
        "lag_market_return": market["market_return"].shift(),
        "lag_20d_volatility": market["market_return"].rolling(20, min_periods=10).std().shift(),
        "lag_intraday_range": market["range"].shift(),
        "lag_cross_sectional_dispersion": market["dispersion"].shift(),
        "opening_gap": market["opening_gap"],
    })
    labels = market["benchmark_return"].shift(-2)
    available = market.index.to_series().shift(-2)
    predictions = []
    for year in range(2005, 2025):
        cutoff = pd.Timestamp(year, 1, 1)
        # The repository filter uses only past and current observations.
        # Refit annually on history from 2000 and exclude unobservable labels.
        regimes = fit_asof_hmm_regimes(
            features.loc[features.index.year <= year],
            training_end=cutoff, n_states=2, random_seed=42,
            include_probabilities=True,
        )
        probabilities = regimes[["regime_probability_0", "regime_probability_1"]]
        known = available.reindex(probabilities.index).lt(cutoff) & labels.reindex(probabilities.index).notna()
        state_returns, *_ = np.linalg.lstsq(
            probabilities.loc[known].to_numpy(), labels.reindex(probabilities.index).loc[known].to_numpy(),
            rcond=None,
        )
        current = probabilities.loc[probabilities.index.year == year]
        predictions.append(pd.Series(current.to_numpy() @ state_returns, index=current.index))
    expected = pd.concat(predictions).sort_index()
    # Exclude today's forecast from its own standardization history.
    prior = expected.shift().expanding(min_periods=252)
    z_score = (expected - prior.mean()) / prior.std()
    return pl.DataFrame({
        "signal_date": expected.index.date.tolist(),
        "expected_market_return": expected.to_numpy(),
        "hmm_z_score": z_score.to_numpy(),
    }, schema_overrides={"signal_date": pl.Date})


if __name__ == "__main__":
    output = Path(sys.argv[1])
    scores = forecast(market_history(Path.home(), output.parent))
    test = scores.filter(pl.col("signal_date") >= date(2009, 12, 1))
    if not test["hmm_z_score"].is_finite().fill_null(False).all():
        raise ValueError("The HMM returned non-finite test scores")
    scores.write_parquet(output.with_suffix(".tmp"))
    output.with_suffix(".tmp").replace(output)
    print(f"Saved {scores.height} as-of HMM forecasts to {output}", flush=True)
