# BOTQ portfolio results

The folder has three notebooks, one dataset folder, and this file. Open the notebooks from this folder. Read them in number order.

1. `01_How_we_got_our_results.ipynb` gives the signal, ridge, Bayesian model average, and portfolio code. It contains the code that makes the result files. The supplied panel and risk files are its inputs.
2. `02_Individual_signal_results.ipynb` reads the result files. It shows signal return charts and style and risk tables.
3. `03_Combined_results.ipynb` reads the result files. It shows combined return charts, ridge coefficients, model weights, and a risk table.

## Data

`analysis_panel.parquet` has dated stock scores, returns, benchmark weights, and Barra style scores. `risk_2010.npz` through `risk_2024.npz` have dated risk arrays. The final result files are `final_returns.parquet`, `final_coefficients.parquet`, `final_model_weights.parquet`, `final_factor_exposures.parquet`, and `style_diagnostics.parquet`. The files `dmd_style_attribution_daily.parquet`, `dmd_style_attribution_summary.csv`, and `dmd_style_attribution_coverage.csv` hold the DMD style and stock-payoff test. Notebook 1 contains the code that makes these files. The folder also includes checks and reference files.

The source stock prices, CPI vintages, and vendor risk service need licensed access. Notebook 1 shows the code that makes the dated scores and risk inputs from those sources. The source files are not in this folder. The supplied dated inputs let you rerun the complete portfolio experiment without model training or vendor access.

## Run

Use Python with numpy, pandas, polars, pyarrow, scipy, clarabel, matplotlib, and Jupyter. Notebook 1 needs at least 16 GB of memory and eight CPU cores for the full test. The full test can take more than 15 minutes. Its last code cell has a comment mark to prevent an accidental long run. Remove that mark to make the result files again. Notebooks 2 and 3 can run on a laptop.

Each portfolio is fully invested. Its forecast active beta is zero. Its annual forecast tracking error is at most 1%. The long-only portfolio cannot have negative stock weights. The test does not subtract trade costs. Confirm the data license before you share the input files.

The public GitHub copy contains the notebooks and this README. The licensed dataset is in the local submission folder for upload to Box. Place its `dataset` folder next to the notebooks before you run them.
