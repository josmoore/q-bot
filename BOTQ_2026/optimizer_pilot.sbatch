#!/usr/bin/env bash
#SBATCH --job-name=botq-opt-pilot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
cd "$HOME/q-bot"
export PYTHONPATH="$HOME/libraries/sf-quant/src" POLARS_MAX_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$HOME/sf-quant-comb-alphas/.venv/bin/python" -u - <<'PY'
from datetime import date
from pathlib import Path
from time import perf_counter
import numpy as np
import polars as pl
import sf_quant.data as sfd
from BOTQ_2026.portfolio import _daily_risk, optimize_day
from BOTQ_2026.sparse_optimizer import solve

sfd.env(root=str(Path.home()), database="production")
assets = pl.scan_parquet(Path.home() / "nobackup/autodelete/botq2026/assets.parquet")
rng = np.random.default_rng(42)
for day in (date(2010, 1, 5), date(2015, 6, 15), date(2020, 3, 16), date(2024, 12, 30)):
    panel = assets.filter(pl.col("in_universe") & (pl.col("date") == day)).select(
        "barrid", "market_cap", "predicted_beta"
    ).sort("barrid").collect().with_columns(
        (pl.col("market_cap") / pl.col("market_cap").sum()).alias("benchmark_weight")
    )
    alpha = rng.normal(size=panel.height)
    alpha /= np.abs(alpha).max()
    panel = panel.with_columns(pl.Series("combined_alpha", alpha))
    benchmark = panel["benchmark_weight"].to_numpy()
    beta = panel["predicted_beta"].to_numpy()
    start = perf_counter()
    risk = _daily_risk(day, tuple(panel["barrid"]))
    load_seconds = perf_counter() - start
    for long_only in (False, True):
        start = perf_counter()
        reference = optimize_day(day, panel, long_only=long_only)
        reference_seconds = perf_counter() - start
        start = perf_counter()
        weights, te = solve(alpha, *risk, benchmark, beta, long_only=long_only,
                            risk_cap=0.025)
        sparse_seconds = perf_counter() - start
        reference_objective = alpha @ reference["active_weight"].to_numpy()
        np.testing.assert_allclose(alpha @ weights, reference_objective, rtol=1e-4, atol=1e-7)
        assert abs(weights.sum()) < 1e-5 and abs(beta @ weights) < 1e-5
        assert te <= 0.02501 and (not long_only or (weights + benchmark).min() >= -1e-6)
        print(day, long_only, f"risk_load={load_seconds:.3f}s",
              f"reference={reference_seconds:.3f}s", f"sparse={sparse_seconds:.3f}s")
PY
