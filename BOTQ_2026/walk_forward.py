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
from BOTQ_2026.dmd_attribution import dmd_style_scores, join_style_controls
from BOTQ_2026.model_candidates import candidate_alphas, centered_fits, portfolio_posteriors


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
    ridge_half_life: float = RIDGE_HALF_LIFE,
    ridge_penalty: float = RIDGE_PENALTY,
    ridge_only: bool = False,
    candidate_grid: bool = False,
    centered_grid: bool = False,
    dmd_inflation: bool = False,
    include_equal: bool = False,
    candidate_returns: pl.DataFrame | None = None,
) -> None:
    """Fit with matured labels and compare both combinations on identical dates."""
    signals = ("dmd", "inflation_beta") if dmd_inflation else SIGNALS
    if single_signal is None:
        ridge_inputs = daily_ridge_inputs(panel.select(
            "signal_date", "label_available_date", "forward_residual_return", *signals
        ), signals=signals)
    test_panel = panel.filter(
        (pl.col("date") >= date(2010, 1, 1)) & (pl.col("label_available_date") <= end_date)
    )
    if dmd_inflation and candidate_returns is None and not test_panel.is_empty():
        test_panel = pl.concat([
            join_style_controls(year) for year in
            test_panel.with_columns(pl.col("signal_date").dt.year().alias("_style_year"))
            .partition_by("_style_year", include_key=False)
        ]).sort("date", "barrid")
    test = test_panel.partition_by("date", maintain_order=True)
    if not test:
        raise ValueError("No backtest sessions in the requested date range")

    execution_dates = [day["date"][0] for day in test]
    fits = centered_fits(ridge_inputs, execution_dates) if centered_grid else None
    include_equal = include_equal or dmd_inflation
    if candidate_returns is not None:
        names = list(fits[execution_dates[0]]) if fits else None
        if names is not None and include_equal:
            names.append("equal_weight")
        model_names, posteriors = portfolio_posteriors(candidate_returns, execution_dates, names)

    def day_candidates(execution, scores):
        if fits is None:
            yield from candidate_alphas(ridge_inputs, execution, scores)
            return
        for name, (fit, _) in fits[execution].items():
            z = (scores - fit.means) / fit.scales
            yield name, np.where(np.isfinite(z), z, 0.0) @ fit.coefficients, fit.coefficients
        if include_equal:
            fit = fit_daily_ridge(ridge_inputs, as_of=execution, half_life_days=126, penalty=1.0)
            z = (scores - fit.means) / fit.scales
            z = np.where(np.isfinite(z), z, 0.0)
            weights = np.full(len(signals), 1 / len(signals))
            yield "equal_weight", z @ weights, weights
            if dmd_inflation and candidate_returns is None:
                weights = np.array([0.95, 0.05])
                yield "dmd95_inflation5", z @ weights, weights

    def evaluate_day(day):
        execution = day["date"][0]
        if candidate_returns is not None:
            candidates = {name: alpha for name, alpha, _ in
                          day_candidates(execution, day.select(signals).to_numpy())}
            matrix = np.column_stack([candidates[name] for name in model_names])
            # Standalone optimizers are scale invariant. Normalize dispersion
            # before blending so posterior weights control relative influence.
            dispersion = matrix.std(axis=0)
            matrix = np.divide(matrix, dispersion, out=np.zeros_like(matrix), where=dispersion > 0)
            alphas = {"bayesian": {variant: matrix @ weights[execution]
                                   for variant, weights in posteriors.items()}}
            coefficients = []
        elif candidate_grid or centered_grid:
            candidates = list(day_candidates(execution, day.select(signals).to_numpy()))
            alphas = {name: alpha for name, alpha, _ in candidates}
            coefficients = []
            for name, _, weights in candidates:
                row = {"date": execution, "portfolio": name, **dict(zip(signals, weights))}
                if fits is not None:
                    ridge = fits[execution].get(name)
                    row["ridge_center"] = ridge[1] if ridge else None
                coefficients.append(row)
            if dmd_inflation:
                residual, raw = dmd_style_scores(day)
                alphas["dmd_residual"] = residual
                alphas["dmd_matched_raw"] = raw
        elif single_signal is None:
            fit = fit_daily_ridge(
                ridge_inputs, as_of=execution, half_life_days=ridge_half_life,
                penalty=ridge_penalty,
            )
            scores = (day.select(signals).to_numpy() - fit.means) / fit.scales
            # Use the same inputs for both combinations. A missing forecast
            # contributes zero in standardized units.
            scores = np.where(np.isfinite(scores), scores, 0.0)
            alphas = {"ridge": scores @ fit.coefficients}
            if not ridge_only:
                alphas["equal_weight"] = scores.mean(axis=1)
            coefficients = [{"date": execution, "portfolio": "ridge",
                             **dict(zip(SIGNALS, fit.coefficients))}]
        else:
            scores = day[single_signal].to_numpy()
            alphas = {single_signal: np.where(np.isfinite(scores), scores, 0.0)}
            coefficients = []

        rows = []
        for method, alpha in alphas.items():
            for long_only in ((False, True) if single_signal is None else (False,)):
                variant = "long_only" if long_only else "active"
                portfolio = f"{method}_{variant}"
                score = alpha[variant] if candidate_returns is not None else alpha
                alpha_panel = day.with_columns(pl.Series("combined_alpha", score))
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
            coefficients.extend(coefficient_row)
            if index % 250 == 0 or index == len(test):
                print(f"Backtest: {index}/{len(test)} sessions through {execution}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    performance = pl.DataFrame(returns)
    coefficient_history = pl.DataFrame(coefficients)
    performance.write_parquet(output / "returns.parquet")
    if candidate_returns is not None:
        pl.DataFrame([
            {"date": day, "portfolio": variant, **dict(zip(model_names, weights))}
            for variant, series in posteriors.items() for day, weights in series.items()
        ]).write_parquet(output / "model_weights.parquet")
    if not coefficient_history.is_empty():
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
    parser.add_argument("--ridge-half-life", type=float, default=RIDGE_HALF_LIFE)
    parser.add_argument("--ridge-penalty", type=float, default=RIDGE_PENALTY)
    parser.add_argument("--ridge-only", action="store_true")
    parser.add_argument("--candidate-grid", action="store_true")
    parser.add_argument("--candidate-returns", type=Path, help="Fit portfolio-based Bayesian model weights")
    parser.add_argument("--centered-grid", action="store_true")
    parser.add_argument("--include-equal", action="store_true")
    parser.add_argument("--dmd-inflation", action="store_true",
                        help="Use DMD and inflation beta in the centered experiment")
    args = parser.parse_args()
    if args.dmd_inflation and not args.centered_grid:
        parser.error("--dmd-inflation requires --centered-grid")
    if args.centered_grid and (args.candidate_grid or args.single_signal or args.ridge_only):
        parser.error("--centered-grid is incompatible with other standalone run modes")
    if args.include_equal and not (args.centered_grid and args.candidate_returns):
        parser.error("--include-equal requires centered grid and merged candidate returns")
    if args.candidate_returns and (args.candidate_grid or args.single_signal or args.ridge_only):
        parser.error("--candidate-returns is a separate run mode")
    if args.candidate_grid and (args.single_signal or args.ridge_only):
        parser.error("--candidate-grid cannot be combined with --single-signal or --ridge-only")
    if not np.isfinite(args.ridge_half_life) or args.ridge_half_life <= 0:
        parser.error("--ridge-half-life must be finite and positive")
    if not np.isfinite(args.ridge_penalty) or args.ridge_penalty < 0:
        parser.error("--ridge-penalty must be finite and nonnegative")
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
        ridge_half_life=args.ridge_half_life, ridge_penalty=args.ridge_penalty,
        ridge_only=args.ridge_only, candidate_grid=args.candidate_grid,
        centered_grid=args.centered_grid, dmd_inflation=args.dmd_inflation,
        include_equal=args.include_equal,
        candidate_returns=pl.read_parquet(args.candidate_returns) if args.candidate_returns else None,
    )


if __name__ == "__main__":
    main()
