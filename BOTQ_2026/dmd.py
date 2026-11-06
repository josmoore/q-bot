"""Hankel delay-coordinate DMD directional signal adapted from DMDBacktestV3.

Use split-adjusted close prices and only observations through the signal date.
This module returns a per-stock direction; the walk-forward layer owns model
selection and delayed execution alignment.
"""

from __future__ import annotations

import numpy as np


def dmd_predicts_up(
    prices: np.ndarray, horizon: int = 1, rank: int = 25, start_horizon: int = 0,
) -> bool:
    """Forecast direction between two future offsets from the last close."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if (n < 4 or start_horizon < 0 or horizon <= start_horizon
            or not np.isfinite(prices).all() or (prices <= 0).any()):
        return False
    # Match the source DMD implementation, which fits raw prices.
    logp = prices.copy()
    row_len = n // 2 + 1
    hankel = np.column_stack([logp[i:i + row_len] for i in range(n - row_len + 1)])
    x, y = hankel[:, :-1], hankel[:, 1:]
    try:
        u, singular, vh = np.linalg.svd(x, full_matrices=False)
        r = min(rank, len(singular))
        if r == 0 or singular[r - 1] <= np.finfo(float).eps:
            return False
        ur, vr = u[:, :r], vh[:r, :].T
        inv_s = np.diag(1.0 / singular[:r])
        reduced = ur.T @ y @ vr @ inv_s
        eigenvalues, modes = np.linalg.eig(reduced)
        good = np.isfinite(eigenvalues) & (np.abs(eigenvalues) > 1e-12)
        if not good.any():
            return False
        eigenvalues, modes = eigenvalues[good], modes[:, good]
        phi = y @ vr @ inv_s @ modes
        amplitudes = np.linalg.lstsq(phi, hankel, rcond=None)[0][:, 0]
        last = phi[0] @ (amplitudes * eigenvalues ** (hankel.shape[1] - 1 + start_horizon))
        future = phi[0] @ (amplitudes * eigenvalues ** (hankel.shape[1] - 1 + horizon))
        return bool(np.isfinite(last) and np.isfinite(future) and future > last)
    except (np.linalg.LinAlgError, FloatingPointError):
        return False


def directional_score(prices: np.ndarray, windows: tuple[int, ...] = tuple(range(8, 25))):
    """Return the best-window DMD direction, using only a supplied trailing series."""
    prices = np.asarray(prices, dtype=float)
    usable = [n for n in windows if n <= len(prices)]
    if not usable:
        return np.nan
    # The original strategy selects windows by trailing hit rate. That online
    # selector is implemented in walk_forward.py; this is the raw forecast.
    return 1.0 if dmd_predicts_up(prices[-max(usable):]) else -1.0


def walk_forward_stock_signal(
    prices: np.ndarray,
    *,
    windows: tuple[int, ...] = tuple(range(3, 25)),
    selection_lookback: int = 60,
    invert: bool = True,
) -> np.ndarray:
    """Daily online hit-rate selection adapted from DMDBacktestV3.

    The forecast covers the close after execution (t+1) to the next close
    (t+2). Window selection uses only those outcomes known by signal date t.
    Output is 0 until enough history exists, otherwise
    +1/-1; the source script's ``INVERT_SIGNALS=True`` convention is retained.
    """
    p = np.asarray(prices, dtype=float)
    t_count = len(p)
    pred = np.full((len(windows), t_count), np.nan)
    for wi, size in enumerate(windows):
        for t in range(size - 1, t_count):
            segment = p[t-size+1:t+1]
            if np.isfinite(segment).all():
                pred[wi, t] = 1.0 if dmd_predicts_up(
                    segment, horizon=2, start_horizon=1,
                ) else -1.0
    out = np.zeros(t_count, dtype=np.float32)
    for t in range(t_count):
        # The latest fully observed t+1 to t+2 return was forecast at t-2.
        hi = t - 2
        lo = hi - selection_lookback + 1
        if lo < 0 or hi < lo:
            continue
        future = p[2:t+1] / p[1:t] - 1.0
        actual = np.sign(future)
        actual[~np.isfinite(future)] = 0
        scores = []
        for wi in range(len(windows)):
            hist = pred[wi, lo:hi+1]
            truth = actual[lo:hi+1]
            valid = np.isfinite(hist) & (truth != 0)
            scores.append(np.mean(np.sign(hist[valid]) == truth[valid]) if valid.sum() == selection_lookback else -1.0)
        best = int(np.argmax(scores))
        if scores[best] >= 0 and np.isfinite(pred[best, t]):
            out[t] = pred[best, t] * (-1 if invert else 1)
    return out
