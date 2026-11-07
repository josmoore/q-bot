"""Fit a compact grid of dated ridge coefficients without portfolio optimization."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from BOTQ_2026.combination import SIGNALS, daily_ridge_inputs, fit_daily_ridge
from BOTQ_2026.walk_forward import build_panel


HALF_LIVES = (63, 252, 1260)
PENALTIES = (1.0, 5.0, 15.0, 50.0)


def run(cache: Path, output: Path, start: date, end: date) -> None:
    assets = pl.scan_parquet(cache / "assets.parquet").filter(
        pl.col("date").is_between(date(2005, 1, 1), end)
    ).select("date", "barrid", "in_universe", "market_cap", "return", "predicted_beta").collect()
    panel = build_panel(
        assets,
        pl.read_parquet(cache / "forecasts-through-2024.parquet"),
        pl.read_parquet(cache / "exit_return_audit.parquet"),
    )
    inputs = daily_ridge_inputs(panel.select(
        "signal_date", "label_available_date", "forward_residual_return", *SIGNALS
    ))
    dates = panel.filter(pl.col("date").is_between(start, end)).select("date").unique().sort("date")["date"].to_list()
    rows = []
    for as_of in dates:
        for half_life in HALF_LIVES:
            for penalty in PENALTIES:
                fit = fit_daily_ridge(inputs, as_of=as_of, half_life_days=half_life, penalty=penalty)
                rows.append({
                    "date": as_of,
                    "half_life_sessions": half_life,
                    "penalty": penalty,
                    "observations": fit.n_observations,
                    **dict(zip(SIGNALS, fit.coefficients)),
                })
    output.mkdir(parents=True, exist_ok=True)
    coefficients = pl.DataFrame(rows)
    coefficients.write_parquet(output / "daily_coefficients.parquet")
    coefficients.group_by("half_life_sessions", "penalty").agg(
        pl.len().alias("fit_days"),
        pl.col("observations").mean().alias("mean_observations"),
        *(pl.col(signal).mean().alias(f"mean_{signal}") for signal in SIGNALS),
        *(pl.col(signal).std().alias(f"std_{signal}") for signal in SIGNALS),
    ).sort("half_life_sessions", "penalty").write_csv(output / "coefficient_summary.csv")
    coefficients.filter(pl.col("date") == coefficients["date"].max()).sort(
        "half_life_sessions", "penalty"
    ).write_csv(output / "final_day_coefficients.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 1, 2))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    args = parser.parse_args()
    run(args.cache, args.output, args.start, args.end)


if __name__ == "__main__":
    main()
