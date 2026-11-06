"""Run chronological portfolio validation and testing on dated signal forecasts.

Forecasts must have one row per signal_date/barrid. The execution date is
the next market session, and the holding period ends one session later.
Returns for exits must be supplied in the asset history, including delistings.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from BOTQ_2026.backtest import active_sharpe, plot_results
from BOTQ_2026.combination import SIGNALS, daily_ridge_inputs, fit_daily_ridge
from BOTQ_2026.portfolio import optimize_day
from BOTQ_2026.validation import select_ridge_parameters


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


def run(
    panel: pl.DataFrame,
    output: Path,
    *,
    end_date: date | None,
    validation_cache: dict | None = None,
    hmm_scores: pl.DataFrame | None = None,
    baseline_dir: Path | None = None,
    workers: int = 1,
    sparse_solver: bool = False,
    single_signal: str | None = None,
) -> None:
    """Fit using past labels, tune on 2007–09, and report through end_date."""
    if single_signal is None:
        history = panel.filter(pl.col("signal_date") >= date(2005, 1, 1)).select(
            "signal_date", "label_available_date", "forward_residual_return", *SIGNALS
        )
        ridge_inputs = daily_ridge_inputs(history)
    validation = []
    test = []
    for day in panel.partition_by("date", maintain_order=True):
        execution = day["date"][0]
        end = day["label_available_date"][0]
        if execution >= date(2007, 1, 1) and end < date(2010, 1, 1):
            validation.append(day)
        elif end_date is not None and execution >= date(2010, 1, 1) and end <= end_date:
            test.append(day)
    if not validation or (end_date is not None and not test):
        raise ValueError("Expected both validation and test sessions")
    beta_targets = {}
    if hmm_scores is not None:
        if end_date is None:
            raise ValueError("HMM beta targets are for the out-of-sample test")
        dates = pl.DataFrame({"signal_date": [day["signal_date"][0] for day in test]})
        targets = dates.join(
            hmm_scores.select("signal_date", "hmm_z_score"),
            on="signal_date", how="left", validate="1:1",
        )
        invalid = targets.filter(~pl.col("hmm_z_score").is_finite().fill_null(False))
        if invalid.height:
            raise ValueError(f"Missing or non-finite HMM z-score on {invalid['signal_date'][0]}")
        beta_targets = dict(targets.with_columns(
            (pl.col("hmm_z_score") / 4).clip(-0.75, 0.75)
        ).iter_rows())
    output.mkdir(parents=True, exist_ok=True)
    returns, coefficients, selected = [], [], {}
    variants = ("active",) if end_date is None else ("active", "long_only")
    if single_signal is not None:
        variants = (f"{single_signal}_active",)
    if beta_targets:
        variants += ("active_hmm_beta", "long_only_hmm_beta")
    if baseline_dir is not None:
        if not beta_targets or validation_cache is None:
            raise ValueError("Baseline reuse requires HMM scores and a validation cache")
        if json.loads((baseline_dir / "validation.json").read_text()) != validation_cache:
            raise ValueError("Baseline and HMM runs must use the same validation selection")
        baseline_returns = pl.read_parquet(baseline_dir / "returns.parquet")
        baseline_coefficients = pl.read_parquet(baseline_dir / "coefficients.parquet")
        for frame, expected_dates in (
            (baseline_returns, [day["label_available_date"][0] for day in test]),
            (baseline_coefficients, [day["date"][0] for day in test]),
        ):
            if set(frame["portfolio"].unique()) != {"active", "long_only"} or any(
                group["date"].sort().to_list() != expected_dates
                for group in frame.partition_by("portfolio")
            ):
                raise ValueError("Saved baseline does not cover the requested test sessions")
        variants = ("active_hmm_beta", "long_only_hmm_beta")
    selection = None if validation_cache is None else validation_cache["active"]
    for variant in variants:
        def evaluate(days, half_life, penalty, report=False):
            phase = "test" if report else "validation"
            print(f"{variant} {phase}: half-life={half_life}, penalty={penalty}, {len(days)} sessions", flush=True)
            def evaluate_day(day):
                execution = day["date"][0]
                if single_signal is not None:
                    # Use the published sign and preserve the universe when a score is missing.
                    scores = day[single_signal].to_numpy()
                    alpha = np.where(np.isfinite(scores), scores, 0.0)
                else:
                    fit = fit_daily_ridge(
                        ridge_inputs, as_of=execution, half_life_days=half_life,
                        penalty=penalty,
                    )
                    scores = day.select(SIGNALS).to_numpy().astype(float)
                    standardized = (scores - fit.means) / fit.scales
                    alpha = np.where(np.isfinite(standardized), standardized, 0) @ fit.coefficients
                target_beta = beta_targets[day["signal_date"][0]] if variant.endswith("_hmm_beta") else 0.0
                try:
                    weights = optimize_day(
                        execution, day.with_columns(pl.Series("combined_alpha", alpha)),
                        long_only=variant.startswith("long_only"),
                        target_active_beta=target_beta,
                        sparse_solver=sparse_solver,
                    ).join(day.select("barrid", "holding_return"), on="barrid", validate="1:1")
                except Exception as exc:
                    raise RuntimeError(f"{variant} {phase} failed on {execution}") from exc
                realized = weights["holding_return"].to_numpy()
                active = float(weights["active_weight"].to_numpy() @ realized)
                row = coefficients_row = None
                if report:
                    end = day["label_available_date"][0]
                    row = dict(date=end, portfolio=variant, active_return=active,
                                        portfolio_return=float(weights["total_weight"].to_numpy() @ realized),
                                        benchmark_return=day["benchmark_return"][0],
                                        carried_names=day["carried_names"][0],
                                        target_active_beta=target_beta,
                                        effective_active_beta_target=weights["effective_active_beta_target"][0],
                                        predicted_tracking_error=weights["predicted_tracking_error"][0],
                                        active_beta=float(weights["active_weight"].to_numpy() @
                                                          weights["predicted_beta"].to_numpy()))
                    if single_signal is None:
                        coefficients_row = dict(date=execution, portfolio=variant,
                                            **dict(zip(SIGNALS, fit.coefficients)))
                return execution, active, row, coefficients_row

            # Ridge fits only read matured labels and costs are zero, so
            # each date can be solved independently. Collect in date order.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for index, (execution, active, row, coefficients_row) in enumerate(
                    pool.map(evaluate_day, days), 1,
                ):
                    if report:
                        returns.append(row)
                        if coefficients_row is not None:
                            coefficients.append(coefficients_row)
                    if index % 250 == 0 or index == len(days):
                        print(f"{variant} {phase}: {index}/{len(days)} through {execution}", flush=True)
                    yield active

        if single_signal is not None:
            for _ in evaluate(test, None, None, report=True):
                pass
            continue
        if selection is None:
            selection = select_ridge_parameters(
                variant, lambda half, penalty, _: evaluate(validation, half, penalty)
            )
        selected["active"] = selection
        if end_date is None:
            continue
        best = selection["best"]
        for _ in evaluate(test, best["half_life"], best["penalty"], report=True):
            pass
    if single_signal is None:
        (output / "validation.json").write_text(json.dumps(selected, indent=2))
    if end_date is None:
        return
    performance = pl.DataFrame(returns)
    coefficient_history = pl.DataFrame(coefficients)
    if baseline_dir is not None:
        performance = pl.concat([baseline_returns, performance], how="diagonal_relaxed").with_columns(
            pl.col("target_active_beta", "effective_active_beta_target").fill_null(0)
        )
        coefficient_history = pl.concat([baseline_coefficients, coefficient_history])
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
    signal_mode = parser.add_mutually_exclusive_group()
    signal_mode.add_argument("--inflation-only", action="store_const", const="inflation_beta", dest="single_signal")
    signal_mode.add_argument("--single-signal", choices=SIGNALS, help="Use one raw forecast as the sole alpha")
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exits", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--hmm-scores", type=Path, help="Parquet with one as-of hmm_z_score per signal_date")
    parser.add_argument("--baseline-dir", type=Path, help="Reuse completed zero-beta results in an HMM comparison")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--end-date", type=date.fromisoformat)
    mode.add_argument("--validation-only", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.home())
    args = parser.parse_args()
    if args.single_signal and (args.validation_only or args.validation_cache or args.hmm_scores or args.baseline_dir):
        parser.error("Standalone signals require --end-date and runs without ridge or HMM options")
    sfd.env(root=str(args.root), database="production")
    exits = pl.read_parquet(args.exits) if args.exits else None
    validation_cache = json.loads(args.validation_cache.read_text()) if args.validation_cache else None
    assets = (
        pl.scan_parquet(args.assets)
        .filter(pl.col("date").is_between(date(2005, 1, 1), args.end_date or date(2009, 12, 31)))
        .select("date", "barrid", "in_universe", "market_cap", "return", "predicted_beta")
        .collect()
    )
    run(
        build_panel(assets, pl.read_parquet(args.forecasts), exits),
        args.out, end_date=args.end_date, validation_cache=validation_cache,
        hmm_scores=pl.read_parquet(args.hmm_scores) if args.hmm_scores else None,
        baseline_dir=args.baseline_dir, workers=args.workers,
        sparse_solver=args.sparse_solver, single_signal=args.single_signal,
    )


if __name__ == "__main__":
    main()
