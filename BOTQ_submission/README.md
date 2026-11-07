# BOTQ submission — completed analysis package

Only notebooks, this README, and dataset files belong in this folder.

**GitHub copy:** This public repository includes the executed notebooks and charts, but excludes the licensed analysis panel and Barra risk archives. The complete local submission package contains those files. See `dataset/README.md` before attempting a fresh run.

1. **01_Research_and_existing_evidence.ipynb**: existing experiments and demonstrable historical charts. These use a 2.5% risk cap and are explicitly historical.
2. **02_Final_model_and_backtest.ipynb**: self-contained implementation and dataset export; DMD residualized against the Barra momentum/reversal pair plus inflation beta; equal weighting and three fixed-prior ridge models, then portfolio-return Bayesian averaging.
3. **03_Final_charts_and_risk_checks.ipynb**: final performance, coefficient, model-weight, exposure, and risk figures.
4. **04_Penalty_independent_1pct_results.ipynb**: fresh 1% TE standalone and unrestricted active equal-weight results, style exposure charts, and diagnostics.

## Current status
The final 1% predicted-tracking-error backtest completed over 3,773 sessions from 2010 through 2024. All four notebooks have been executed. The selected penalties are 18, 5, and 2.5 for half-lives of 63, 252, and 1260 sessions. `historical_*` files are earlier 2.5% experiments and are labeled separately from final results.

## Environment and execution
Python 3.11+ with numpy, pandas, polars, pyarrow, scipy, matplotlib, clarabel, cvxpy, nbformat, nbclient, ipykernel, and sf-quant for the initial data export. Use the established BYU environment. Full export and backtests must execute in Slurm allocations, not on the login node. Open notebooks with this folder as the working directory. The implementation is embedded in notebook cells; no project Python files are needed. Preserve the exported stock ordering in annual NPZ risk archives.

## Portfolio definition
Both variants are fully invested total portfolios. Active weights sum to zero and have zero predicted market beta relative to the benchmark. Long-only additionally requires nonnegative total stock weights. The other variant permits shorts. The annual ex-ante tracking-error limit is 1%; realized tracking error can differ. Returns are gross of trading costs. Executions lag signal dates; all regression and BMA labels must have matured strictly before execution.

## Model design
Decay half-lives: 63, 252, 1260 sessions; penalties: 18, 5, 2.5, respectively. Each ridge fit uses a fixed common center of 10⁻⁵ for both standardized signal coefficients on every date. BMA uses past unrestricted active portfolio returns and applies the same date-specific model weights to both unrestricted and long-only forecasts. It normalizes candidate score dispersion before blending. Standalone DMD, matched raw DMD, and inflation beta are also evaluated long-only; ridge and equal-weight candidates are evaluated as unrestricted active portfolios only. Coefficients are predictive slopes, not capital allocations; signed normalized coefficients describe relative loading on standardized signals. Equal weight is 50/50 on standardized scores. BMA probabilities are weights on models, not stock holdings.

## Data dictionary and scope
Historical Parquet/CSV files contain dates, portfolio returns, benchmark returns, active returns, risk checks, coefficient histories, BMA weights, and DMD attribution results. `analysis_panel.parquet` contains dated stock IDs, raw DMD, inflation signal, forward holding returns, benchmark weights, predicted beta, and two Barra style exposures. `risk_YEAR.npz` contains ordered stock IDs and dated factor exposure, factor covariance, and specific-variance arrays for 2010–2024. `final_returns.parquet`, `final_coefficients.parquet`, `final_model_weights.parquet`, and `final_factor_exposures.parquet` contain the completed backtest. `baseline_returns.parquet` and `baseline_factor_exposures.parquet` are extracted from that same final run for notebook 04. `style_diagnostics.parquet` records DMD residualization coverage and explained variation.

The historical DMD comparison removes four controls (Barra and sf-quant); final DMD removes only the Barra pair. Report these as separate experiments. Forecast caches inherit the corrected t+1-to-t+2 DMD holding horizon and decimal-unit inflation construction. Forecast-generation routines are included for inspection; full upstream regeneration still requires raw prices and vintage CPI inputs. The dataset uses licensed financial sources; verify recipient data entitlements before distribution. No submission/upload has been performed.
