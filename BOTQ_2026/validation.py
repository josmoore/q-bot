"""Chronological selection of ridge hyperparameters by active Sharpe.

The active portfolio selects the settings used by both test variants. ``evaluate`` must
run the full validation dates in order and return their daily active returns;
the same dates cannot be reused as out-of-sample test dates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from BOTQ_2026.backtest import active_sharpe


DEFAULT_HALF_LIVES = (126.0, 252.0, 504.0)
DEFAULT_PENALTIES = (1e-3, 1e-1)


def select_ridge_parameters(
    portfolio: str,
    evaluate: Callable[[float, float, str], Iterable[float]],
    *,
    half_lives: tuple[float, ...] = DEFAULT_HALF_LIVES,
    penalties: tuple[float, ...] = DEFAULT_PENALTIES,
) -> dict:
    """Choose half-life and penalty by chronological validation active Sharpe.

    ``evaluate(half_life, penalty, portfolio)`` is expected to fit using only
    observations available before each validation date and return daily active
    returns from the constrained optimizer.
    """
    if portfolio not in {"active", "long_only"}:
        raise ValueError(f"Unknown portfolio variant {portfolio!r}")
    rows = []
    for half_life in half_lives:
        for penalty in penalties:
            daily = np.asarray(list(evaluate(half_life, penalty, portfolio)), dtype=float)
            score = active_sharpe(daily)
            rows.append({"portfolio": portfolio, "half_life": half_life,
                         "penalty": penalty, "active_sharpe": score})
    valid = [row for row in rows if np.isfinite(row["active_sharpe"])]
    if not valid:
        raise ValueError(f"No valid validation results for {portfolio}")
    best = max(valid, key=lambda row: row["active_sharpe"])
    return {"best": best, "grid": rows}
