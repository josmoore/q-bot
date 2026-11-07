"""Verify the completed experiment and extract presentation baselines."""
from pathlib import Path

import numpy as np
import pandas as pd

repo = Path(__file__).resolve().parents[2]
data = (repo / 'BOTQ_submission/dataset') if (repo / 'BOTQ_2026').is_dir() else (Path(__file__).resolve().parents[1] / 'outputs/BOTQ_submission/dataset')
returns = pd.read_parquet(data / 'final_returns.parquet')
factors = pd.read_parquet(data / 'final_factor_exposures.parquet')
coefficients = pd.read_parquet(data / 'final_coefficients.parquet')
weights = pd.read_parquet(data / 'final_model_weights.parquet')

baseline_names = ('equal_weight', 'dmd_standalone', 'inflation_standalone', 'original_dmd_matched')
baseline = returns[returns.portfolio.str.startswith(baseline_names)].copy()
baseline_factors = factors[factors.portfolio.str.startswith(baseline_names)].copy()
baseline.to_parquet(data / 'baseline_returns.parquet', index=False)
baseline_factors.to_parquet(data / 'baseline_factor_exposures.parquet', index=False)

assert not returns.empty and not coefficients.empty and not weights.empty
assert returns.predicted_te.max() <= 0.010001
assert returns.active_beta.abs().max() < 1e-5
assert np.allclose(returns.total_investment, 1, atol=1e-5)
assert returns[returns.portfolio.str.endswith('_long_only')].minimum_total_weight.min() > -1e-6
assert set(coefficients.method) == {'ridge_63_p18', 'ridge_252_p5', 'ridge_1260_p2.5'}
assert np.allclose(coefficients.center, 1e-5)
assert set(weights.variant) == {'active', 'long_only'}
weight_columns = [c for c in weights if c not in ('date', 'variant')]
assert np.allclose(weights[weight_columns].sum(axis=1), 1)
wa = weights[weights.variant == 'active'].sort_values('date')[weight_columns].to_numpy()
wl = weights[weights.variant == 'long_only'].sort_values('date')[weight_columns].to_numpy()
assert np.allclose(wa, wl)
dates = returns.groupby('portfolio').date.nunique()
assert dates.nunique() == 1, dates.to_dict()
print('Sessions:', dates.iloc[0])
print('Portfolios:', sorted(dates.index))
print('Max predicted TE:', returns.predicted_te.max())
print('Max absolute active beta:', returns.active_beta.abs().max())
print('Baseline and risk checks verified')
