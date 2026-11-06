# Battle of the Quants 2026

Research code in this directory is developed on the `botq-2026` branch of the
existing [q-bot repository](https://github.com/Tarotar30127/q-bot), so it can be
reviewed and pushed upstream as a focused folder.

## Agreed scope

- Daily, dated Russell 3000 universe and benchmark data from `sf-quant`.
- Three stock-level forecasts: DMD, Sharpe Transformer, and inflation beta.
- Market-beta residual forward returns as the combination-regression target.
- Compare fixed exponentially weighted ridge (126 trading sessions, penalty 0.2)
  with an equal-weight mean of the same three standardized forecasts.
- One combined alpha followed by one constrained optimization per date.
- Two active portfolios: 2.5% predicted tracking-error cap and zero active beta;
  the second also requires long-only total weights.
- Charts for cumulative total and benchmark-relative log returns, plus the
  fitted ridge coefficients.
- Initial model training before 2010, followed by point-in-time retraining.
- The ridge target is the next trading day's market-beta residual return.
- Use sf-quant's dated Barra risk inputs for portfolio optimization.
- Daily rebalancing. Trading costs are set to zero in this first pass, as
  requested; results must not be described as cost-adjusted.

## Source code to adapt

- `../DMDBacktestV3.py`: DMD forecast and window-selection logic. Its same-close
  trade convention must be adapted to a feasible common execution schedule.
- `../Sharpe_Transformer.ipynb`: model, feature, and training logic. Its 100-stock
  selection and fixed cross-section assumptions must be adapted to the dated
  Russell universe.
- `sf-quant`: Polars data loaders, dated benchmark weights, Barra risk inputs,
  and `dynamic_mve_optimizer`. Keep this as a dependency rather than copying its
  internals unless the supercomputer installation requires a local editable copy.

The working copy on BYU ORC is `/home/josmoore/q-bot/BOTQ_2026`. Source code
belongs under the backed-up home directory; temporary model checkpoints and
large intermediate files belong under `~/nobackup/autodelete` and can be
deleted after twelve weeks of disuse. The dated source data is read from the
existing `~/groups/grp_quant/database/production/assets` annual partitions:
dated investable-universe membership, market-cap weights, total returns,
close, volume, predicted beta, and specific risk. OHLC features use
`~/groups/grp_quant/database/research/crsp_daily` (openprc, askhi, bidlo,
prc), joined on date and the first eight CUSIP characters. CRSP defines askhi
and bidlo as daily high/low trades or closing ask/bid when a trade price is
unavailable. CRSP v2's sampled high/low values were constant at close, so it
is not used. Market-context fields retain Jeffrey's Yahoo tickers: SPY, ^VIX,
^TNX, and ^IRX. CPI uses ALFRED vintage data (CPIAUCSL), selected as known on
the signal date. The optimizer and risk loaders use the current source checkout
at `~/libraries/sf-quant`, revision `6317431`, through `PYTHONPATH`; the shared
installed environment is not modified.

The signal date precedes execution by one trading session and the holding
return ends one session later. The 2010–2024 comparison tests both
combinations under each portfolio constraint set. Both use the same trailing
standardization and treat missing standardized scores as zero. The optional
`--single-signal` argument runs one raw forecast as an active portfolio.
The active chart sums the difference of portfolio and benchmark log returns.
Summary statistics use simple daily returns.

The Slurm entry points are `prepare_assets.sbatch`, `transformer_features.sbatch`,
`dmd.sbatch`, `transformer.sbatch`, and `backtest.sbatch`. The last requires
`--end-date YYYY-MM-DD` and reads the three forecast caches and the exit-return
audit. The 2025 Barra data is complete through the July 3 execution session
before a gap. Reported dates refer to the end of the holding period, so the
July 3 execution uses the July 7 return. Full-year results require additional
dated membership and risk data.

When the primary return is missing, the runner uses the verified CRSP fallback
where available. Otherwise it carries the position at the last observed value
for that session. Both portfolios and the benchmark apply the same return, and
the results table records the number of carried names each session.
