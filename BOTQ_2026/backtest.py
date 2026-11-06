"""Daily next-close portfolio accounting and plots for BOTQ_2026.

Targets are set after close on date t and held from the next close, so weights
on t multiply the close-to-close return recorded on t+1. Trading costs are
zero by the user's instruction; reported performance is gross.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def active_sharpe(returns: np.ndarray, annualization: int = 252) -> float:
    """Annualized Sharpe of daily active returns, with zero risk-free spread."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(annualization) * r.mean() / r.std(ddof=1))


def plot_results(returns: pl.DataFrame, coefficients: pl.DataFrame, output: Path) -> None:
    """Save total wealth vs Russell, active wealth, and coefficients by variant."""
    output.mkdir(parents=True, exist_ok=True)
    data = returns.sort("date").to_pandas()
    if (data[["portfolio_return", "benchmark_return"]] <= -1).any().any():
        raise ValueError("Log returns require positive daily gross returns")
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in data.groupby("portfolio"):
        wealth = np.log1p(group["portfolio_return"]).cumsum()
        ax.plot(group["date"], wealth, label=f"{name} total")
    bench = data.groupby("date")["benchmark_return"].first()
    ax.plot(bench.index, np.log1p(bench).cumsum(), color="black", label="Russell 3000")
    ax.set(title="Portfolio and benchmark performance", ylabel="Cumulative log return")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "total_vs_russell.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in data.groupby("portfolio"):
        active_wealth = (np.log1p(group["portfolio_return"]) - np.log1p(group["benchmark_return"])).cumsum()
        ax.plot(group["date"], active_wealth, label=name)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Performance relative to the Russell 3000", ylabel="Cumulative excess log return")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output / "active_performance.png", dpi=160); plt.close(fig)

    if coefficients.is_empty():
        return
    c = coefficients.sort("date").to_pandas()
    count = c["portfolio"].nunique()
    fig, axes = plt.subplots(count, 1, figsize=(12, 4 * count), sharex=True, squeeze=False)
    for ax, (name, group) in zip(axes[:, 0], c.groupby("portfolio")):
        for signal in ("dmd", "sharpe_transformer", "inflation_beta"):
            ax.plot(group["date"], group[signal], label=signal)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"Ridge coefficients: {name}"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(output / "ridge_coefficients.png", dpi=160); plt.close(fig)
