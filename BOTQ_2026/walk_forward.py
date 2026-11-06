"""Backtest fixed ridge and equal-weight combinations of dated forecasts.

Forecasts must have one row per signal_date/barrid. The execution date is
the next market session, and the holding period ends one session later.
Returns for exits must be supplied in the asset history, including delistings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from BOTQ_2026.backtest import active_sharpe, plot_results
from BOTQ_2026.combination import SIGNALS, daily_ridge_inputs, fit_daily_ridge
from BOTQ_2026.portfolio import optimize_day


def build_panel(assets: pl.DataFrame, forecasts: pl.DataFrame, exits: pl.DataFrame | None = None) -> pl.DataFrame:
    """Align dates on a common exchange calendar, including for stocks that exit."""
    calendar = assets.select("date").unique().sort("date").with_columns(
        pl.col("date").shift(1).alias("signal_date"),
        pl.col("date").shift(-1).alias("label_available_date"),
    )
    members = assets.filter(pl.col("in_universe")).with_columns(
        (pl.col("market_cap") / pl.col("market_cap").sum().over("date")).alias("benchmark_weight")
    )
    # sf-quant stores daily returns in percentage points. CRSP fallbacks are decimals.
    realized = assets.select(
        pl.col("date").alias("label_available_date"),
        "barrid",
        (pl.col("return") / 100).alias("holding_return"),
    )
    if exits is not None:
        realized = realized.join(
            exits.select("label_available_date", "barrid", "delret", "crsp_return"),
            on=["label_available_date", "barrid"], how="full", coalesce=True, validate="1:1",
        ).with_columns(
            pl.coalesce("holding_return", "delret", "crsp_return").alias("holding_return")
        ).drop("delret", "crsp_return")
    panel = (
        members.join(calendar, on="date", validate="m:1")
        .drop_nulls(["signal_date", "label_available_date"])
        .join(forecasts, on=["signal_date", "barrid"], how="left", validate="m:1")
        .join(realized, on=["label_available_date", "barrid"], how="left", validate="m:1")
        .with_columns(
            pl.col("holding_return").is_null().sum().over("date").alias("carried_names"),
            pl.col("holding_return").fill_null(0),
        )
        .with_columns(
            (pl.col("benchmark_weight") * pl.col("holding_return")).sum().over("date").alias("benchmark_return")
        )
        .with_columns(
            (pl.col("holding_return") - pl.col("predicted_beta") * pl.col("benchmark_return"))
            .alias("forward_residual_return")
        )
    )
    return panel.sort(["date", "barrid"])


RIDGE_HALF_LIFE = 126  # Six months in trading sessions.
RIDGE_PENALTY = 0.2


def run(
    panel: pl.DataFrame,
    output: Path,
    *,
    end_date: date,
    workers: int = 1,
    sparse_solver: bool = False,
    single_signal: str | None = None,
) -> None:
    """Fit with matured labels and compare both combinations on identical dates."""
    if single_signal is None:
        ridge_inputs = daily_ridge_inputs(panel.select(
            "signal_date", "label_available_date", "forward_residual_return", *SIGNALS
        ))
    test = panel.filter(
        (pl.col("date") >= date(2010, 1, 1)) & (pl.col("label_available_date") <= end_date)
    ).partition_by("date", maintain_order=True)
    if not test:
        raise ValueError("No backtest sessions in the requested date range")

    def evaluate_day(day):
        execution = day["date"][0]
        if single_signal is None:
            fit = fit_daily_ridge(
                ridge_inputs, as_of=execution, half_life_days=RIDGE_HALF_LIFE,
                penalty=RIDGE_PENALTY,
            )
            scores = (day.select(SIGNALS).to_numpy() - fit.means) / fit.scales
            # Use the same inputs for both combinations. A missing forecast
            # contributes zero in standardized units.
            scores = np.where(np.isfinite(scores), scores, 0.0)
            alphas = {"ridge": scores @ fit.coefficients, "equal_weight": scores.mean(axis=1)}
            coefficients = {"date": execution, "portfolio": "ridge",
                            **dict(zip(SIGNALS, fit.coefficients))}
        else:
            scores = day[single_signal].to_numpy()
            alphas = {single_signal: np.where(np.isfinite(scores), scores, 0.0)}
            coefficients = None

        rows = []
        for method, alpha in alphas.items():
            alpha_panel = day.with_columns(pl.Series("combined_alpha", alpha))
            for long_only in ((False, True) if single_signal is None else (False,)):
                portfolio = f"{method}_{'long_only' if long_only else 'active'}"
                try:
                    weights = optimize_day(
                        execution, alpha_panel, long_only=long_only, sparse_solver=sparse_solver,
                    ).join(day.select("barrid", "holding_return"), on="barrid", validate="1:1")
                except Exception as exc:
                    raise RuntimeError(f"{portfolio} failed on {execution}") from exc
                realized = weights["holding_return"].to_numpy()
                rows.append(dict(
                    date=day["label_available_date"][0], portfolio=portfolio,
                    active_return=float(weights["active_weight"].to_numpy() @ realized),
                    portfolio_return=float(weights["total_weight"].to_numpy() @ realized),
                    benchmark_return=day["benchmark_return"][0],
                    carried_names=day["carried_names"][0],
                    predicted_tracking_error=weights["predicted_tracking_error"][0],
                    active_beta=float(weights["active_weight"].to_numpy() @ weights["predicted_beta"].to_numpy()),
                ))
        return execution, rows, coefficients

    returns, coefficients = [], []
    # Solve all variants for a date together so they share the same Barra load.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (execution, rows, coefficient_row) in enumerate(pool.map(evaluate_day, test), 1):
            returns.extend(rows)
            if coefficient_row is not None:
                coefficients.append(coefficient_row)
            if index % 250 == 0 or index == len(test):
                print(f"Backtest: {index}/{len(test)} sessions through {execution}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    performance = pl.DataFrame(returns)
    coefficient_history = pl.DataFrame(coefficients)
    performance.write_parquet(output / "returns.parquet")
    if single_signal is None:
        coefficient_history.write_parquet(output / "coefficients.parquet")
    summary = []
    for group in performance.partition_by("portfolio"):
        daily = group["active_return"].to_numpy()
        summary.append({
            "portfolio": group["portfolio"][0],
            "first_return": group["date"].min(),
            "last_return": group["date"].max(),
            "sessions": group.height,
            "portfolio_cumulative_return": float(np.prod(1 + group["portfolio_return"].to_numpy()) - 1),
            "benchmark_cumulative_return": float(np.prod(1 + group["benchmark_return"].to_numpy()) - 1),
            "active_sharpe": active_sharpe(daily),
            "realized_tracking_error": float(np.std(daily, ddof=1) * np.sqrt(252)),
            "max_predicted_tracking_error": group["predicted_tracking_error"].max(),
            "max_absolute_active_beta": group["active_beta"].abs().max(),
            "carried_name_sessions": group["carried_names"].sum(),
        })
    pl.DataFrame(summary).write_csv(output / "summary.csv")
    plot_results(performance, coefficient_history, output)


def main() -> None:
    import sf_quant.data as sfd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sparse-solver", action="store_true")
    parser.add_argument("--single-signal", choices=SIGNALS, help="Use one raw forecast as the sole alpha")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exits", type=Path)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--root", type=Path, default=Path.home())
    args = parser.parse_args()
    sfd.env(root=str(args.root), database="production")
    exits = pl.read_parquet(args.exits) if args.exits else None
    assets = (
        pl.scan_parquet(args.assets)
        .filter(pl.col("date").is_between(date(2005, 1, 1), args.end_date))
        .select("date", "barrid", "in_universe", "market_cap", "return", "predicted_beta")
        .collect()
    )
    run(
        build_panel(assets, pl.read_parquet(args.forecasts), exits),
        args.out, end_date=args.end_date, workers=args.workers,
        sparse_solver=args.sparse_solver, single_signal=args.single_signal,
    )


if __name__ == "__main__":
    main()
