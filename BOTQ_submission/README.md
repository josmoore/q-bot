# Modal Anomaly | BOTQ resubmission

Open the notebooks from this folder in number order. Keep the dataset folder beside them.

1. **How We Got Our Results** gives the code for DMD, inflation beta, the daily sf-quant adjustment, the ridge rules, model averaging, the optimizer, and the backtest.
2. **Individual Signal Results** compares original and adjusted DMD and Betting Against Inflation Beta. It shows the stock-level attribution results.
3. **Combined Portfolio Results** compares both Bayesian blends under long-short and long-only constraints. It includes the benchmark and the active and total performance tables.
4. **Turnover, Trading Costs, and Factor Risk** shows turnover, cost sensitivity, drawdowns, and Barra exposures.

## Run the notebooks

Use Python 3 with Jupyter, numpy, pandas, polars, pyarrow, scipy, clarabel, and matplotlib. Select the Python 3 kernel for that environment. All four notebooks run from the supplied files and contain saved outputs. The full replay call in Notebook 1 is optional and can exceed 15 minutes. It requires the original licensed sf-quant momentum and reversal score snapshot, the supplied stock panel, and the yearly risk arrays. To generate the panel and risk arrays from source, use the code in Notebook 1 with licensed stock data and access to the risk service. The raw source files and separate sf-quant score files are outside this package.

The replay code writes candidate returns, model weights, portfolio returns, turnover, and optional daily factor exposure files to a separate folder. Notebook 1 includes a check that compares the replayed returns and turnover with the supplied results. The six-session pilot matched its own saved inputs. Exact agreement with the full comparison requires the original score snapshot and a full replay. Notebook 3 labels the Barra-adjusted reference charts. The supplied sf-quant comparison files contain portfolio outcomes but omit its daily model-weight history.

## Data files

- `regular_dmd_returns.parquet` and `regular_dmd_turnover.parquet` contain the ten portfolio series used in the presentation. There are 3,773 daily returns and 3,772 turnover measurements per portfolio.
- `regular_dmd_coverage.parquet` and the two `dmd_style_attribution_*` files contain the style coverage and stock-level attribution data.
- `regular_dmd_factor_exposures.parquet` contains the daily Barra exposure and forecast risk contribution of each factor for the four Bayesian portfolios.
- `daily_risk_free.csv` contains the daily Fama/French risk-free series for 2010–2024 from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Values are decimal returns.
- The two `barra_reference_*` files contain the coefficients and model weights for the labeled reference charts.
- `analysis_panel.parquet`, the fifteen annual `risk_YYYY.npz` files, and `risk_factor_names.csv` contain the supplied stock-level inputs and risk arrays.

## Test definitions

Both total portfolios are fully invested and target zero forecast active market beta. Predicted annual tracking error is capped at 1%. The long-only portfolio requires nonnegative total stock weights. The return charts and performance tables show gross results. Notebook 4 gives cost scenarios based on one-way turnover.
