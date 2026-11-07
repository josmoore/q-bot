"""Short coefficient test with two signals and a fixed nonzero ridge prior."""
from datetime import date
from pathlib import Path
import numpy as np
import polars as pl

from BOTQ_2026.combination import daily_ridge_inputs, fit_daily_ridge

DATA=Path.home()/'nobackup/autodelete/botq2026/submission/dataset'
PRIOR=1e-5
SIGNALS=('dmd','inflation_beta')
panel=pl.read_parquet(DATA/'analysis_panel.parquet')
days=[]
for day in panel.partition_by('signal_date',maintain_order=True):
    raw=day['dmd'].to_numpy()
    styles=day.select('USSLOWL_MOMENTUM','USSLOWL_LTREVRSL').to_numpy()
    ok=np.isfinite(raw)&np.isfinite(styles).all(axis=1)
    if ok.sum()<30: continue
    y=raw[ok]-raw[ok].mean()
    x=styles[ok]-styles[ok].mean(axis=0)
    cleaned=np.zeros(day.height)
    cleaned[ok]=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
    days.append(day.with_columns(pl.Series('dmd',cleaned)).select('signal_date','label_available_date','forward_residual_return',*SIGNALS))
print('Residualized',len(days),'dates',flush=True)
inputs=daily_ridge_inputs(pl.concat(days),signals=SIGNALS)
dates=panel.filter(pl.col('date').is_between(date(2024,10,1),date(2024,12,31))).select('date').unique().sort('date')['date'].to_list()
rows=[]
for dt in dates:
    for half in (63,252,1260):
        for penalty in (5.,15.,50.):
            fit=fit_daily_ridge(inputs,as_of=dt,half_life_days=half,penalty=penalty,center=PRIOR)
            rows.append(dict(date=dt,half_life_sessions=half,penalty=penalty,prior=PRIOR,dmd=fit.coefficients[0],inflation_beta=fit.coefficients[1]))
print('Fitted',len(rows),'coefficient pairs',flush=True)
pl.DataFrame(rows).write_parquet(DATA/'fixed_prior_coefficients_2024Q4.parquet')
