"""One combined-alpha optimization in active weight space.

The constraints and risk units are inherited from sf-quant. Read the
repository's BYU_ORC_AGENTS.md before running this on ORC.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np
import polars as pl


TARGET_TRACKING_ERROR = 0.025


@lru_cache(maxsize=800)
def _daily_risk(day: date, ids: tuple[str, ...]):
    """Reuse dated Barra inputs across portfolios."""
    from sf_quant.data import construct_factor_model_components

    return construct_factor_model_components(day, list(ids))


def optimize_day(
    day: date,
    panel: pl.DataFrame,
    *,
    long_only: bool,
    sparse_solver: bool = False,
) -> pl.DataFrame:
    """Optimize total and active weights for one dated Russell cross-section.

    ``panel`` needs unique ``barrid`` values and the columns ``combined_alpha``,
    ``benchmark_weight``, and ``predicted_beta``. Incomplete risk or signal
    inputs fail visibly rather than silently changing the benchmark universe.
    The beta target is measured on active weights relative to the benchmark.
    """
    from sf_quant.optimizer import (
        LongOnly,
        ZeroBeta,
        ZeroInvestment,
        active_mve_optimizer,
    )

    required = {"barrid", "combined_alpha", "benchmark_weight", "predicted_beta"}
    if missing := required - set(panel.columns):
        raise ValueError(f"{day}: missing columns {sorted(missing)}")
    if panel["barrid"].n_unique() != panel.height:
        raise ValueError(f"{day}: duplicate barrid values")
    if any(panel[c].null_count() for c in required):
        raise ValueError(f"{day}: missing optimization inputs")
    ids = panel["barrid"].to_list()
    alpha = panel["combined_alpha"].to_numpy().astype(float)
    benchmark = panel["benchmark_weight"].to_numpy().astype(float)
    beta = panel["predicted_beta"].to_numpy().astype(float)
    if not all(np.isfinite(x).all() for x in (alpha, benchmark, beta)):
        raise ValueError(f"{day}: non-finite optimization inputs")
    if not np.isclose(benchmark.sum(), 1.0, atol=1e-6):
        raise ValueError(f"{day}: benchmark weights sum to {benchmark.sum():.8f}")
    exposures, factor_covariance, specific_variance = _daily_risk(day, tuple(ids))
    constraints = [ZeroInvestment(), ZeroBeta()]
    if long_only:
        constraints.append(LongOnly())
    # A positive scaling preserves the optimizer and keeps tiny forecasts
    # from setting the numerical scale of the conic objective.
    scale = max(float(np.max(np.abs(alpha))), np.finfo(float).tiny)
    from cvxpy.error import SolverError

    try:
        if sparse_solver:
            from BOTQ_2026.sparse_optimizer import solve

            active, predicted_te = solve(
                alpha / scale, exposures, factor_covariance, specific_variance,
                benchmark, beta, long_only=long_only,
                risk_cap=TARGET_TRACKING_ERROR,
            )
        else:
            result = active_mve_optimizer(
                ids=ids,
                alphas=alpha / scale,
                factor_exposures=exposures,
                factor_covariance=factor_covariance,
                specific_risk=specific_variance,
                constraints=constraints,
                benchmark_weights=benchmark,
                target_active_risk=TARGET_TRACKING_ERROR,
                betas=beta,
            )
            active = result["active_weight"].to_numpy()
            predicted_te = float(result["active_risk"][0])
    except SolverError:
        active, predicted_te = _solve_soc(
            alpha / scale, exposures, factor_covariance, specific_variance,
            constraints, benchmark, beta,
        )

    total = active + benchmark
    if abs(active.sum()) > 1e-5 or abs(beta @ active) > 1e-5:
        raise ValueError(f"{day}: investment or active-beta constraint violated")
    if long_only and total.min() < -1e-6:
        raise ValueError(f"{day}: long-only constraint violated")
    if predicted_te > TARGET_TRACKING_ERROR + 1e-5:
        raise ValueError(f"{day}: tracking-error cap violated")
    return pl.DataFrame(
        {
            "date": [day] * len(ids),
            "barrid": ids,
            "benchmark_weight": benchmark,
            "active_weight": active,
            "total_weight": total,
            "predicted_beta": beta,
            "combined_alpha": alpha,
            "predicted_tracking_error": [predicted_te] * len(ids),
            "portfolio": ["long_only" if long_only else "active"] * len(ids),
        }
    )


def _solve_soc(
    alpha, exposures, covariance, specific_variance, constraints, benchmark, beta,
    *, risk_cap=TARGET_TRACKING_ERROR,
):
    """Maximize a linear objective with the Barra risk bound expressed as a norm."""
    import cvxpy as cp

    eig, vectors = np.linalg.eigh(covariance)
    root = vectors * np.sqrt(np.maximum(eig, 0))
    active = cp.Variable(len(alpha))
    factor = cp.Variable(exposures.shape[1])
    risk_vector = cp.hstack([root.T @ factor, cp.multiply(np.sqrt(specific_variance), active)])
    limits = [factor == exposures.T @ active, cp.norm(risk_vector) <= risk_cap]
    limits.extend(c(active, benchmark_weights=benchmark, betas=beta) for c in constraints)
    problem = cp.Problem(cp.Maximize(alpha @ active), limits)
    try:
        problem.solve(solver="CLARABEL", max_threads=1)
    except cp.error.SolverError:
        problem.solve(solver="SCS", eps=1e-7, max_iters=30000)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or active.value is None:
        raise ValueError(f"Fallback portfolio optimization failed: {problem.status}")
    weights = active.value
    risk = np.linalg.norm(np.r_[root.T @ (exposures.T @ weights), np.sqrt(specific_variance) * weights])
    return weights, float(risk)
