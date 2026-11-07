"""Fixed coefficient rules for the walk-forward ensemble."""

from datetime import date
from itertools import product

import numpy as np

from BOTQ_2026.combination import SIGNALS, fit_daily_ridge

HALF_LIVES = (63, 252, 1260)
PENALTIES = (1.0, 0.05, 0.0025)


def candidate_alphas(inputs, as_of: date, scores: np.ndarray):
    """Yield each candidate's alpha and its coefficients on standardized inputs."""
    for half_life, penalty in product(HALF_LIVES, PENALTIES):
        fit = fit_daily_ridge(inputs, as_of=as_of, half_life_days=half_life, penalty=penalty)
        standardized = (scores - fit.means) / fit.scales
        standardized = np.where(np.isfinite(standardized), standardized, 0.0)
        if (half_life, penalty) == (HALF_LIVES[0], PENALTIES[0]):
            # Keep the same 126-session standardization as the earlier equal-weight run.
            equal_fit = fit_daily_ridge(inputs, as_of=as_of, half_life_days=126, penalty=1.0)
            equal_scores = (scores - equal_fit.means) / equal_fit.scales
            equal_scores = np.where(np.isfinite(equal_scores), equal_scores, 0.0)
            coefficients = np.full(len(SIGNALS), 1 / len(SIGNALS))
            yield "equal_weight", equal_scores @ coefficients, coefficients
        name = f"ridge_{half_life}_p{penalty:g}"
        yield name, standardized @ fit.coefficients, fit.coefficients


CENTERED_SPECS = ((63, 1.0), (1260, 0.0025))
CENTER_WARMUP = 252


def centered_fits(inputs, dates):
    """Fit in chronological order, initializing each series with ordinary ridge."""
    centers = dict.fromkeys(CENTERED_SPECS, 0.0)
    result = {}
    for index, day in enumerate(dates):
        result[day] = {}
        for half_life, penalty in CENTERED_SPECS:
            key = (half_life, penalty)
            center = centers[key]
            fit = fit_daily_ridge(inputs, as_of=day, half_life_days=half_life,
                                  penalty=penalty, center=center)
            if index < CENTER_WARMUP:
                centers[key] = float(fit.coefficients.max())
            result[day][f"centered_{half_life}_p{penalty:g}"] = (fit, center)
    return result


def portfolio_posteriors(returns, execution_dates, names=None):
    """Update uniform model odds with each candidate's active-return evidence.

    Compare a Gaussian positive-alpha model (1% annual active mean) to a
    zero-mean reference, both with 2.5% annual volatility. The daily log
    likelihood ratio is linear in the realized return. Use the same reference
    for all candidates, and normalize relative evidence across candidates.
    """
    import polars as pl
    from scipy.special import softmax

    if names is None:
        names = ["equal_weight", *(
            f"ridge_{half_life}_p{penalty:g}" for half_life, penalty in product(HALF_LIVES, PENALTIES)
        )]
    dates = np.asarray(execution_dates, dtype="datetime64[D]")
    posteriors = {}
    for variant in ("active", "long_only"):
        columns = [f"{name}_{variant}" for name in names]
        history = returns.filter(pl.col("portfolio").is_in(columns)).pivot(
            on="portfolio", index="date", values="active_return",
        ).sort("date")
        if set(history.columns) != {"date", *columns} or history.height != len(dates):
            raise ValueError(f"Incomplete candidate return history for {variant}")
        realized = history.select(columns).to_numpy()
        if not np.isfinite(realized).all():
            raise ValueError(f"Non-finite candidate returns for {variant}")
        # A return ending on the execution date is not known before that close.
        known = np.searchsorted(history["date"].to_numpy(), dates, side="left")
        cumulative = np.vstack([np.zeros(len(names)), realized.cumsum(axis=0)])
        posterior = softmax((0.01 / 0.025**2) * cumulative[known], axis=1)
        posteriors[variant] = dict(zip(execution_dates, posterior))
    return names, posteriors
