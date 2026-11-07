"""Leakage-controlled, exponentially weighted ridge signal combination.

Read BYU_ORC_AGENTS.md before running large panels on the cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl


SIGNALS = ("dmd", "sharpe_transformer", "inflation_beta")


@dataclass(frozen=True)
class RidgeFit:
    as_of: date
    half_life_days: float
    penalty: float
    intercept: float
    coefficients: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    n_observations: int

    def predict(self, signal_matrix: np.ndarray) -> np.ndarray:
        x = np.asarray(signal_matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.coefficients.size:
            raise ValueError(f"Expected matrix with {self.coefficients.size} signal columns")
        return self.intercept + ((x - self.means) / self.scales) @ self.coefficients


def fit_ew_ridge(
    observations: pl.DataFrame,
    *,
    as_of: date,
    half_life_days: float,
    penalty: float,
    standardize: bool,
) -> RidgeFit:
    """Fit only matured labels known before ``as_of``.

    The input must contain ``signal_date``, ``label_available_date``, the three
    signal columns, and ``forward_residual_return``. The label-available date
    is the date on which the forward return is known; the caller must include
    any extra publication delay if its data source requires one.

    Row weights decay by trading-session distance from the latest eligible
    signal date, avoiding different weights for weekends. The ridge penalty
    applies to standardized coefficients; the intercept is not penalized.
    """
    if half_life_days <= 0 or penalty < 0:
        raise ValueError("half-life must be positive and penalty nonnegative")
    required = {"signal_date", "label_available_date", "forward_residual_return", *SIGNALS}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    history = (
        observations.filter(pl.col("label_available_date") < as_of)
        .drop_nulls(list(SIGNALS) + ["forward_residual_return"])
        .sort("signal_date")
    )
    if history.height < 30:
        raise ValueError(f"Only {history.height} matured observations before {as_of}")
    dates = history["signal_date"].to_list()
    sessions = {day: i for i, day in enumerate(sorted(set(dates)))}
    age = np.asarray([len(sessions) - 1 - sessions[day] for day in dates], dtype=float)
    weights = np.exp2(-age / half_life_days)
    x = history.select(SIGNALS).to_numpy().astype(float)
    y = history["forward_residual_return"].to_numpy().astype(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite training values")
    total_weight = weights.sum()
    means = np.average(x, axis=0, weights=weights) if standardize else np.zeros(x.shape[1])
    scales = (
        np.sqrt(np.average((x - means) ** 2, axis=0, weights=weights))
        if standardize
        else np.ones(x.shape[1])
    )
    scales = np.maximum(scales, 1e-12)
    xz = (x - means) / scales
    y_mean = np.average(y, weights=weights)
    x_mean = np.average(xz, axis=0, weights=weights)
    xc = xz - x_mean
    yc = y - y_mean
    gram = (xc * weights[:, None]).T @ xc / total_weight
    rhs = (xc * weights[:, None]).T @ yc / total_weight
    coefficients = np.linalg.solve(gram + penalty * np.eye(len(SIGNALS)), rhs)
    intercept = float(y_mean - x_mean @ coefficients)
    return RidgeFit(
        as_of=as_of,
        half_life_days=half_life_days,
        penalty=penalty,
        intercept=intercept,
        coefficients=coefficients,
        means=means,
        scales=scales,
        n_observations=history.height,
    )


def daily_ridge_inputs(observations: pl.DataFrame, signals=SIGNALS):
    """Compress labeled rows to daily sufficient statistics for repeated fits."""
    target = "forward_residual_return"
    columns = (*signals, target)
    products = [(a, b) for i, a in enumerate(signals) for b in columns[i:]]
    daily = (
        observations.filter(pl.all_horizontal(pl.col(c).is_finite() for c in columns))
        .group_by("signal_date", "label_available_date")
        .agg(pl.len().alias("n"), *(pl.col(c).sum() for c in columns),
             *((pl.col(a) * pl.col(b)).sum().alias(f"{a}/{b}") for a, b in products))
        .sort("signal_date")
    )
    return daily, products, signals


def fit_daily_ridge(inputs, *, as_of: date, half_life_days: float, penalty: float, center: float = 0.0) -> RidgeFit:
    """Fit trailing standardized ridge from the same moments as the row-level fit."""
    daily, products, signals = inputs
    history = daily.filter(pl.col("label_available_date") < as_of)
    n = history["n"].to_numpy()
    if n.sum() < 30:
        raise ValueError(f"Only {n.sum()} matured observations before {as_of}")
    weights = np.exp2(-np.arange(len(n) - 1, -1, -1) / half_life_days)
    total = weights @ n
    means = weights @ history.select(*signals, "forward_residual_return").to_numpy() / total
    weighted_products = weights @ history.select(*(f"{a}/{b}" for a, b in products)).to_numpy() / total
    covariance = np.zeros((len(signals), len(signals)))
    cross = np.zeros(len(signals))
    for (a, b), moment in zip(products, weighted_products):
        i = signals.index(a)
        if b in signals:
            j = signals.index(b)
            covariance[i, j] = covariance[j, i] = moment - means[i] * means[j]
        else:
            cross[i] = moment - means[i] * means[-1]
    scales = np.maximum(np.sqrt(np.maximum(covariance.diagonal(), 0)), 1e-12)
    gram = covariance / scales[:, None] / scales[None, :]
    coefficients = np.linalg.solve(gram + penalty * np.eye(len(signals)), cross / scales + penalty * center)
    return RidgeFit(as_of, half_life_days, penalty, means[-1], coefficients,
                    means[:-1], scales, int(n.sum()))
