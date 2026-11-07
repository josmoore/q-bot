"""Create the notebook-only submission; generator stays outside submission."""
from pathlib import Path
import json
import shutil

REPO = Path(__file__).resolve().parents[2]
IN_REPO = (REPO / 'BOTQ_2026').is_dir()
ROOT = REPO if IN_REPO else Path(__file__).resolve().parents[1]
OUT = ROOT / ('BOTQ_submission' if IN_REPO else 'outputs/BOTQ_submission')
SRC = ROOT / ('BOTQ_2026' if IN_REPO else 'outputs/q-bot/BOTQ_2026')
OUT.mkdir(exist_ok=True)
(OUT / 'dataset').mkdir(exist_ok=True)

def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s): return {'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]}
def save(name, cells):
    (OUT/name).write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells},indent=1))

common = """from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
DATA = Path('dataset')
assert DATA.exists(), 'Run from the submission directory'
plt.rcParams.update({'figure.figsize': (11, 5), 'axes.spines.top':False, 'axes.spines.right':False})
"""

for tag, folder in [('historical_candidates','botq2026-dmd-inflation-candidates'),('historical_bma','botq2026-dmd-inflation-bayesian'),('historical_inflation','botq2026-inflation-only'),('historical_dmd','botq2026-dmd-only'),('historical_attribution','botq2026-dmd-attribution')]:
    for f in (ROOT/'outputs'/folder).glob('*'):
        if f.suffix in ('.csv','.parquet'):
            shutil.copy2(f, OUT/'dataset'/f'{tag}_{f.name}')

save('01_Research_and_existing_evidence.ipynb',[
md('# DMD, inflation sensitivity, and style overlap\nThese are existing **2.5% predicted TE** experiments, not the final 1% experiment. Historical residual DMD removes all four style controls; the final experiment below removes only the Barra pair. Returns are gross of costs. Active Sharpe means mean active return divided by active volatility, annualized (an information ratio).'),code(common),
code("""r = pd.concat([pd.read_parquet(DATA/'historical_candidates_returns.parquet'),pd.read_parquet(DATA/'historical_bma_returns.parquet')])
for variant in ['active','long_only']:
    fig, ax = plt.subplots()
    for name, g in r[r.portfolio.str.endswith('_'+variant)].groupby('portfolio'):
        g=g.sort_values('date')
        ax.plot(g.date, (np.log1p(g.portfolio_return)-np.log1p(g.benchmark_return)).cumsum(), label=name)
    ax.set(title=f'Historical {variant}: 2.5% predicted TE',ylabel='Cumulative excess log return',xlabel='Date')
    ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1)); ax.grid(alpha=.2); plt.show()
"""),
md('## Original DMD versus style-residualized DMD\nMatched raw DMD and residualized DMD use identical coverage. This isolates the removal of styles from changes to the sample.'),
code("""for variant in ['active','long_only']:
    fig, ax=plt.subplots()
    for method in ['dmd_matched_raw','dmd_residual']:
        g=r[r.portfolio==f'{method}_{variant}'].sort_values('date')
        ax.plot(g.date,(np.log1p(g.portfolio_return)-np.log1p(g.benchmark_return)).cumsum(),label=method)
    ax.set(title=f'DMD comparison: {variant} (historical 2.5% TE)',ylabel='Cumulative excess log return');ax.legend();plt.show()
for signal in ['dmd','inflation']:
    g=pd.read_parquet(DATA/f'historical_{signal}_returns.parquet').sort_values('date')
    fig,ax=plt.subplots(); ax.plot(g.date,(np.log1p(g.portfolio_return)-np.log1p(g.benchmark_return)).cumsum())
    ax.set(title=f'Standalone {signal}: historical 2.5% TE',ylabel='Cumulative excess log return');plt.show()
a=pd.read_parquet(DATA/'historical_attribution_daily.parquet')
display(pd.read_csv(DATA/'historical_attribution_summary.csv'))
fig,ax=plt.subplots();a.groupby('controls').score_r2.mean().plot.bar(ax=ax)
ax.set(title='Fraction of DMD score variation explained by styles',ylabel='Mean daily R²');plt.show()
fig,ax=plt.subplots()
for name in ['barra','sf_quant','all']:
    g=a[a.controls==name].sort_values('signal_date')
    ax.plot(g.signal_date,g.remaining_payoff.cumsum(),label=name)
ax.set(title='DMD payoff remaining after style controls',ylabel='Cumulative payoff (additive, not portfolio wealth)');ax.legend();plt.show()
""")])

# Embed source as notebook code, with module namespaces for compatibility.
bootstrap="""import sys, types
pkg=types.ModuleType('BOTQ_2026');pkg.__path__=[];sys.modules['BOTQ_2026']=pkg
def register(name, source):
    module=types.ModuleType('BOTQ_2026.'+name)
    sys.modules[module.__name__]=module
    setattr(pkg,name,module)
    exec(compile(source, '<notebook:'+name+'>', 'exec'), module.__dict__)
    return module
"""
cells=[md('# Final experiment: two signals, four combination rules\nRun on an allocated BYU Slurm compute node for full panel preparation/backtesting. This notebook contains the implementation; it does not import the old repository. sf-quant and its licensed risk data must be available for the initial export. Once exported, the panel and dated risk snapshots support offline replay. Half-lives are 63, 252, and 1,260 trading sessions; these are exponential decay rates, not truncated windows.'),code(common+bootstrap)]
for name in ['combination','sparse_optimizer','backtest','portfolio','dmd_attribution','model_candidates','walk_forward','dmd','inflation','macro']:
    source=(SRC/f'{name}.py').read_text()
    cells += [md(f'## Implementation: {name}'),code(f"{name} = register({name!r}, r'''\n{source}\n''')")]
cells += [md('## Configuration and prior\nThe penalty is λ‖β − c·1‖² on standardized signals with fixed c = 10⁻⁵ on every date. Both signals share this center. The fitted coefficients can still differ. Equal-weight means equal weights on standardized signal scores. Zero beta is **active beta**, so total beta equals benchmark beta; total weights sum to one. BMA evidence comes only from unrestricted active portfolios; the same resulting model weights are used to form the long-only BMA portfolio.'),code("""PENALTIES = {63: 18, 252: 5, 1260: 2.5}
FIXED_PRIOR = 1e-5
TE_CAP = 0.01
END = __import__('datetime').date(2024,12,31)
portfolio.TARGET_TRACKING_ERROR = TE_CAP
SIGNALS = ('dmd','inflation_beta')
"""),md('## Export the actual analysis dataset and risk snapshots\nThis export uses corrected, previously generated point-in-time DMD and inflation forecasts. It exports the full benchmark universe, matured returns, and dated Barra styles. Missing style coverage is tracked, with unmatched DMD assigned zero. Do not distribute licensed underlying data beyond permitted recipients. Historical result files alone are insufficient to rerun the optimizer.'),code("""def prepare_dataset(cache=Path.home()/'nobackup/autodelete/botq2026/corrected-signals'):
    import sf_quant.data as sfd
    sfd.env(root=str(Path.home()), database='production')
    assets=pl.scan_parquet(cache/'assets.parquet').filter(pl.col('date').is_between(__import__('datetime').date(2005,1,1),END)).select('date','barrid','in_universe','market_cap','return','predicted_beta').collect()
    forecasts=pl.read_parquet(cache/'forecasts-through-2024.parquet').select('signal_date','barrid',*SIGNALS)
    panel=walk_forward.build_panel(assets,forecasts,pl.read_parquet(cache/'exit_return_audit.parquet'))
    parts=[]
    for year in panel.with_columns(pl.col('signal_date').dt.year().alias('_year')).partition_by('_year',include_key=False):
        styles=sfd.load_exposures(year['signal_date'].min(),year['signal_date'].max(),False,['USSLOWL_MOMENTUM','USSLOWL_LTREVRSL'])
        parts.append(year.join(styles.rename({'date':'signal_date'}),on=['signal_date','barrid'],how='left',validate='m:1'))
    panel=pl.concat(parts).sort('date','barrid')
    panel.write_parquet(DATA/'analysis_panel.parquet')
    print('Analysis panel saved:',panel.shape,flush=True)
    # Years are independent; limit concurrent archives to the Slurm CPU allocation.
    from concurrent.futures import ThreadPoolExecutor
    import os
    years=panel.filter(pl.col('date').dt.year()>=2010).with_columns(pl.col('date').dt.year().alias('_year')).partition_by('_year',include_key=False)
    def risk_year(year):
        destination=DATA/f'risk_{year["date"][0].year}.npz'
        if destination.exists(): return
        arrays={}
        for day in year.partition_by('date',maintain_order=True):
            key=str(day['date'][0]);ids=day['barrid'].to_list()
            e,f,s=sfd.construct_factor_model_components(day['date'][0],ids)
            arrays[key+'_ids']=np.asarray(ids,dtype=str)
            arrays[key+'_E']=e;arrays[key+'_F']=f;arrays[key+'_S']=s
        np.savez_compressed(destination,**arrays)
        print('Saved risk archive',destination,flush=True)
    with ThreadPoolExecutor(max_workers=min(4,int(os.environ.get('SLURM_CPUS_PER_TASK','4')))) as pool:
        list(pool.map(risk_year,years))
    return panel
# Run prepare_dataset() on the cluster to populate the final dataset.
"""),md('## Final backtest\nFits use only labels available strictly before execution. BMA updates from realized candidate active returns available strictly before execution. BMA uses the existing Gaussian evidence rule (1% annual alternative mean, 2.5% reference volatility); this is a specified evidence model, not proof that a candidate is the true model. Candidate scores are normalized before blending. Both variants are optimized at the new 1% cap, including standalone signals and matched original DMD.'),code("""def run_final(baselines_only=False):
    if not baselines_only:
        assert all(v is not None and v>=0 for v in PENALTIES.values()), 'Set final penalties first'
    panel=pl.read_parquet(DATA/'analysis_panel.parquet').sort('date','barrid')
    diagnostics=[];parts=[]
    for day in panel.partition_by('date',maintain_order=True):
        raw=day['dmd'].to_numpy(); styles=day.select('USSLOWL_MOMENTUM','USSLOWL_LTREVRSL').to_numpy()
        mask=np.isfinite(raw)&np.isfinite(styles).all(axis=1)
        if mask.sum()<30: raise ValueError('Insufficient style coverage')
        x=styles[mask]-styles[mask].mean(axis=0); y=raw[mask]-raw[mask].mean()
        residual=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
        clean=np.zeros(len(raw));matched=np.zeros(len(raw));clean[mask]=residual;matched[mask]=y
        diagnostics.append({'date':day['date'][0],'coverage':mask.mean(),'r2':1-residual@residual/(y@y) if y@y else np.nan})
        parts.append(day.with_columns(pl.Series('dmd',clean),pl.Series('raw_dmd',matched)))
    panel=pl.concat(parts)
    pl.DataFrame(diagnostics).write_parquet(DATA/'style_diagnostics.parquet')
    inputs=combination.daily_ridge_inputs(panel,signals=SIGNALS)
    days=panel.filter((pl.col('date').dt.year()>=2010)&(pl.col('label_available_date')<=END)).partition_by('date',maintain_order=True)
    dates=[d['date'][0] for d in days]
    fits={} if baselines_only else {dt:{f'ridge_{half}_p{penalty:g}':(combination.fit_daily_ridge(inputs,as_of=dt,half_life_days=half,penalty=penalty,center=FIXED_PRIOR),FIXED_PRIOR) for half,penalty in PENALTIES.items()} for dt in dates}
    import threading, os
    from concurrent.futures import ThreadPoolExecutor
    rows=[];coefs=[];factor_rows=[];worker_state=threading.local()
    workers=min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','4')))
    def evaluate(day, name, alpha, long_only):
        dt=day['date'][0];key=str(dt)
        if getattr(worker_state,'risk_year',None)!=dt.year:
            if getattr(worker_state,'archive',None) is not None: worker_state.archive.close()
            worker_state.archive=np.load(DATA/f'risk_{dt.year}.npz',allow_pickle=False)
            worker_state.risk_year=dt.year
        archive=worker_state.archive
        assert list(archive[key+'_ids'])==day['barrid'].to_list()
        E,F,S=[archive[key+'_'+s] for s in ('E','F','S')]
        b=day['benchmark_weight'].to_numpy();beta=day['predicted_beta'].to_numpy()
        a,te=sparse_optimizer.solve(alpha/max(np.max(np.abs(alpha)),1e-20),E,F,S,b,beta,long_only=long_only,risk_cap=TE_CAP)
        total=b+a; ret=day['holding_return'].to_numpy()
        assert abs(total.sum()-1)<1e-5 and abs(a@beta)<1e-5 and te<=TE_CAP+1e-6
        if long_only: assert total.min()>-1e-6
        variant='long_only' if long_only else 'active';label=name+'_'+variant
        exposures=E.T@a
        factor_rows.extend({'date':dt,'portfolio':label,'factor_index':j,'active_exposure':float(v),'variance_contribution':float(v*(F@exposures)[j])} for j,v in enumerate(exposures))
        return dict(date=day['label_available_date'][0],portfolio=label,active_return=float(a@ret),portfolio_return=float(total@ret),benchmark_return=float(b@ret),predicted_te=te,active_beta=float(a@beta),total_investment=float(total.sum()),minimum_total_weight=float(total.min()),specific_variance=float((a*a)@S),lagged_momentum_exposure=float(a@np.nan_to_num(day['USSLOWL_MOMENTUM'].to_numpy())),lagged_reversal_exposure=float(a@np.nan_to_num(day['USSLOWL_LTREVRSL'].to_numpy())))
    def alphas(day):
        dt=day['date'][0];scores=day.select(SIGNALS).to_numpy();out={}
        for name,(fit,center) in fits.get(dt,{}).items():
            z=np.nan_to_num((scores-fit.means)/fit.scales)
            out[name]=z@fit.coefficients
        f=combination.fit_daily_ridge(inputs,as_of=dt,half_life_days=126,penalty=1)
        z=np.nan_to_num((scores-f.means)/f.scales)
        out['equal_weight']=z.mean(axis=1)
        return out
    def candidate_day(day):
        dt=day['date'][0];candidates=alphas(day)
        coefficients=[dict(date=dt,method=name,center=center,**dict(zip(SIGNALS,fit.coefficients))) for name,(fit,center) in fits.get(dt,{}).items()]
        candidates.update(dmd_standalone=day['dmd'].to_numpy(),inflation_standalone=np.nan_to_num(day['inflation_beta'].to_numpy()),original_dmd_matched=day['raw_dmd'].to_numpy())
        result=[]
        for name,alpha in candidates.items():
            variants=(False,True) if name in ('dmd_standalone','inflation_standalone','original_dmd_matched') else (False,)
            for lo in variants: result.append(evaluate(day,name,alpha,lo))
        return result,coefficients
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,(result,coefficients) in enumerate(pool.map(candidate_day,days),1):
            rows.extend(result);coefs.extend(coefficients)
            if index%250==0: print('Backtest',index,'/',len(days),flush=True)
    history=pl.DataFrame(rows)
    if baselines_only:
        history.write_parquet(DATA/'baseline_returns.parquet')
        pl.DataFrame(factor_rows).write_parquet(DATA/'baseline_factor_exposures.parquet')
        print('Penalty-independent backtests complete',flush=True)
        return
    names=list(fits[dates[0]])+['equal_weight']
    from scipy.special import softmax
    columns=[name+'_active' for name in names]
    active_history=history.filter(pl.col('portfolio').is_in(columns)).pivot(on='portfolio',index='date',values='active_return').sort('date')
    assert set(active_history.columns)=={'date',*columns} and active_history.height==len(dates)
    realized=active_history.select(columns).to_numpy()
    assert np.isfinite(realized).all()
    known=np.searchsorted(active_history['date'].to_numpy(),np.asarray(dates,dtype='datetime64[D]'),side='left')
    cumulative=np.vstack([np.zeros(len(names)),realized.cumsum(axis=0)])
    active_post=dict(zip(dates,softmax((0.01/0.025**2)*cumulative[known],axis=1)))
    post={'active':active_post,'long_only':active_post}
    def bayes_day(day):
        dt=day['date'][0];a=alphas(day);matrix=np.column_stack([a[n] for n in names]);sd=matrix.std(axis=0)
        matrix=np.divide(matrix,sd,out=np.zeros_like(matrix),where=sd>0)
        return [evaluate(day,'bayesian',matrix@post['long_only' if lo else 'active'][dt],lo) for lo in (False,True)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,result in enumerate(pool.map(bayes_day,days),1):
            rows.extend(result)
            if index%250==0: print('Bayesian backtest',index,'/',len(days),flush=True)
    pl.DataFrame(rows).write_parquet(DATA/'final_returns.parquet')
    pl.DataFrame(coefs).write_parquet(DATA/'final_coefficients.parquet')
    pl.DataFrame(factor_rows).write_parquet(DATA/'final_factor_exposures.parquet')
    pl.DataFrame([dict(date=dt,variant=v,**dict(zip(names,w))) for v,series in post.items() for dt,w in series.items()]).write_parquet(DATA/'final_model_weights.parquet')
# run_final() after the dataset export.
""")]
save('02_Final_model_and_backtest.ipynb',cells)

save('03_Final_charts_and_risk_checks.ipynb',[md('# Final results and presentation charts\nRequires the final 1% experiment outputs from notebook 02. It fails visibly if they are absent; it never relabels historical results as final.'),code(common),code("""r=pd.read_parquet(DATA/'final_returns.parquet').sort_values('date')
c=pd.read_parquet(DATA/'final_coefficients.parquet').sort_values('date')
w=pd.read_parquet(DATA/'final_model_weights.parquet').sort_values('date')
f=pd.read_parquet(DATA/'final_factor_exposures.parquet')
assert r.predicted_te.max()<=.010001 and r.active_beta.abs().max()<1e-5
assert np.allclose(r.total_investment,1,atol=1e-5)
assert r[r.portfolio.str.endswith('_long_only')].minimum_total_weight.min()>-1e-6
metrics=[]
for name,g in r.groupby('portfolio'):
    metrics.append({'portfolio':name,'active_sharpe':np.sqrt(252)*g.active_return.mean()/g.active_return.std(),'realized_TE':np.sqrt(252)*g.active_return.std(),'max_predicted_TE':g.predicted_te.max()})
display(pd.DataFrame(metrics))
for variant in ['active','long_only']:
    sub=r[r.portfolio.str.endswith('_'+variant)]
    for kind in ['excess_log','total_log','drawdown','rolling_information_ratio','predicted_te']:
        fig,ax=plt.subplots()
        for name,g in sub.groupby('portfolio'):
            log=np.log1p(g.portfolio_return).cumsum()
            values={'excess_log':log-np.log1p(g.benchmark_return).cumsum(),'total_log':log,'drawdown':np.expm1(log-log.cummax()),'rolling_information_ratio':np.sqrt(252)*g.active_return.rolling(252).mean()/g.active_return.rolling(252).std(),'predicted_te':g.predicted_te}[kind]
            ax.plot(g.date,values,label=name)
        ax.set(title=f'{variant}: {kind}',xlabel='Date',ylabel=kind);ax.legend(fontsize=8,bbox_to_anchor=(1,1));ax.grid(alpha=.2);plt.show()
for variant in ['active','long_only']:
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for name in ['original_dmd_matched','dmd_standalone','inflation_standalone','bayesian']:
        g=r[r.portfolio==name+'_'+variant]
        for ax,col in zip(axes,['lagged_momentum_exposure','lagged_reversal_exposure']):
            ax.plot(g.date,g[col],label=name);ax.set(ylabel=col)
    axes[0].set_title(f'{variant}: exposures to signal-date Barra styles (missing exposures treated as zero)')
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    plt.show()
for name,g in c.groupby('method'):
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for s in ['dmd','inflation_beta']:
        axes[0].plot(g.date,g[s],label=s)
        axes[1].plot(g.date,g[s]/g[['dmd','inflation_beta']].abs().sum(axis=1),label=s)
    axes[0].plot(g.date,g.center,'k--',label='Shared prior center')
    axes[0].set(title=name,ylabel='Raw ridge coefficient');axes[1].set(ylabel='Signed coefficient / sum of absolute coefficients')
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    plt.show()
for variant,g in w.groupby('variant'):
    names=[x for x in g if x not in ['date','variant']]
    assert np.allclose(g[names].sum(axis=1),1)
    fig,ax=plt.subplots();ax.stackplot(g.date,*[g[n] for n in names],labels=names)
    ax.set(title=f'BMA model probabilities: {variant}',ylabel='Model weight',ylim=(0,1));ax.legend(bbox_to_anchor=(1,1));plt.show()
for variant in ['active','long_only']:
    sub=f[f.portfolio.str.endswith('_'+variant)]
    table=sub.groupby(['portfolio','factor_index']).active_exposure.mean().unstack()
    fig,ax=plt.subplots(figsize=(14,5));im=ax.imshow(table,aspect='auto',cmap='RdBu_r')
    ax.set_yticks(range(len(table)),table.index);ax.set(title=f'Mean active factor exposure: {variant}',xlabel='Factor index (risk archive order)')
    fig.colorbar(im,ax=ax);plt.show()
    fig,ax=plt.subplots()
    for name,g in sub.groupby('portfolio'):
        v=g.groupby('date').variance_contribution.sum()
        ax.plot(v.index,v,label=name)
    ax.set(title=f'Factor variance contribution: {variant}',ylabel='Annual predicted variance');ax.legend(fontsize=8,bbox_to_anchor=(1,1));plt.show()
d=pd.read_parquet(DATA/'style_diagnostics.parquet')
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
axes[0].plot(d.date,d.r2);axes[0].set(ylabel='DMD style R²',title='Barra momentum and reversal removed from original DMD')
axes[1].plot(d.date,d.coverage);axes[1].set(ylabel='Style coverage',xlabel='Date');plt.show()
""")])

save('04_Penalty_independent_1pct_results.ipynb',[
md('# Fresh 1% tracking-error standalone results\nThis notebook uses the standalone and equal-weight results extracted from the final run. Ridge and equal-weight candidates are evaluated only as unrestricted active portfolios; standalone DMD and inflation are also evaluated long-only. Original and residualized DMD use the same stocks. All returns are gross of costs.'),
code(common),
code("""r=pd.read_parquet(DATA/'baseline_returns.parquet').sort_values('date')
f=pd.read_parquet(DATA/'baseline_factor_exposures.parquet')
d=pd.read_parquet(DATA/'style_diagnostics.parquet')
assert r.predicted_te.max()<=.010001
assert r.active_beta.abs().max()<1e-5
assert np.allclose(r.total_investment,1,atol=1e-5)
assert r[r.portfolio.str.endswith('_long_only')].minimum_total_weight.min()>-1e-6
summary=[]
for name,g in r.groupby('portfolio'):
    summary.append({'portfolio':name,'sessions':len(g),'active_sharpe':np.sqrt(252)*g.active_return.mean()/g.active_return.std(),'realized_TE':np.sqrt(252)*g.active_return.std(),'max_predicted_TE':g.predicted_te.max(),'max_abs_beta':g.active_beta.abs().max()})
display(pd.DataFrame(summary).sort_values('portfolio'))
for variant in ['active','long_only']:
    fig,ax=plt.subplots()
    for name,g in r[r.portfolio.str.endswith('_'+variant)].groupby('portfolio'):
        g=g.sort_values('date')
        ax.plot(g.date,(np.log1p(g.portfolio_return)-np.log1p(g.benchmark_return)).cumsum(),label=name)
    ax.set(title=f'Fresh standalone and equal weight: {variant}, 1% predicted TE',ylabel='Cumulative excess log return',xlabel='Date')
    ax.legend(bbox_to_anchor=(1,1));ax.grid(alpha=.2);plt.show()
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for name,g in r[r.portfolio.str.endswith('_'+variant)].groupby('portfolio'):
        axes[0].plot(g.date,g.lagged_momentum_exposure,label=name)
        axes[1].plot(g.date,g.lagged_reversal_exposure,label=name)
    axes[0].set(ylabel='Barra momentum active exposure',title=f'Style exposures: {variant}')
    axes[1].set(ylabel='Barra reversal active exposure',xlabel='Date')
    for ax in axes: ax.legend(fontsize=8,bbox_to_anchor=(1,1));ax.grid(alpha=.2)
    plt.show()
fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
axes[0].plot(d.date,d.r2);axes[0].set(ylabel='Daily DMD score R²',title='Share explained by Barra momentum and reversal')
axes[1].plot(d.date,d.coverage);axes[1].set(ylabel='Matched stock coverage',xlabel='Date')
plt.show()
""")])
(OUT/'README.md').write_text('''# BOTQ submission — completed analysis package

Only notebooks, this README, and dataset files belong in this folder.

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
''')
print(OUT)
