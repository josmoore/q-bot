"""Solve the sf-quant Barra risk problem without building a dense asset matrix."""

import clarabel
import numpy as np
from scipy import sparse


def solve(
    alpha, exposures, covariance, specific_variance, benchmark, beta,
    *, long_only, beta_target, risk_cap,
):
    """Maximize alpha under an exact second-order cone tracking-error cap.

    Decision variables are active weights and their factor exposures. Keeping
    factor exposures explicit preserves sparsity in the stock risk block.
    A None beta target leaves beta unconstrained for the HMM feasibility solve.
    """
    stocks, factors = exposures.shape
    size = stocks + factors
    eig, vectors = np.linalg.eigh(covariance)
    if eig.min() < -1e-10:
        raise ValueError("Barra factor covariance is not positive semidefinite")
    root = vectors * np.sqrt(np.maximum(eig, 0))
    equalities = [np.ones(stocks)]
    rhs = [0.0]
    if beta_target is not None:
        equalities.append(beta)
        rhs.append(beta_target)
    a = [
        sparse.bmat([
            [sparse.csc_matrix(equalities), sparse.csc_matrix((len(rhs), factors))],
            [-sparse.csc_matrix(exposures.T), sparse.eye(factors)],
        ]),
        sparse.csc_matrix((1, size)),
        sparse.bmat([
            [sparse.csc_matrix((factors, stocks)), -sparse.csc_matrix(root.T)],
            [-sparse.diags(np.sqrt(specific_variance)), sparse.csc_matrix((stocks, factors))],
        ]),
    ]
    b = [np.r_[rhs, np.zeros(factors)], np.r_[risk_cap, np.zeros(size)]]
    cones = [clarabel.ZeroConeT(len(rhs) + factors), clarabel.SecondOrderConeT(1 + size)]
    if long_only:
        a.append(sparse.hstack([-sparse.eye(stocks), sparse.csc_matrix((stocks, factors))]))
        b.append(benchmark)
        cones.append(clarabel.NonnegativeConeT(stocks))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    result = clarabel.DefaultSolver(
        sparse.csc_matrix((size, size)), np.r_[-alpha, np.zeros(factors)],
        sparse.vstack(a, format="csc"), np.concatenate(b), cones, settings,
    ).solve()
    if result.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
        raise ValueError(f"Sparse Barra optimizer failed: {result.status}")
    weights = np.asarray(result.x[:stocks])
    risk = np.linalg.norm(np.r_[root.T @ (exposures.T @ weights), np.sqrt(specific_variance) * weights])
    return weights, float(risk)
