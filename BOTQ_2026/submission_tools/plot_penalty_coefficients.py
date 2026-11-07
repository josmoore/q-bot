"""Compare higher ridge penalties for each 2024 decay half-life."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
IN_REPO=(REPO/'BOTQ_2026').is_dir()
ROOT=REPO if IN_REPO else Path(__file__).resolve().parents[1]
DATA=ROOT/('BOTQ_submission/dataset/fixed_prior_coefficients_2024Q4.parquet' if IN_REPO else 'outputs/BOTQ_submission/dataset/fixed_prior_coefficients_2024Q4.parquet')
OUT=ROOT/('botq2026-coefficient-charts' if IN_REPO else 'outputs/botq2026-coefficient-charts')
OUT.mkdir(exist_ok=True)
d=pd.read_parquet(DATA)
print(d.columns.tolist())
for half in (63,252,1260):
    panel=d[(d.half_life_sessions==half)&d.penalty.isin([5,15,50])].sort_values('date')
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for ax,signal in zip(axes,['dmd','inflation_beta']):
        for penalty in (5,15,50):
            g=panel[panel.penalty==penalty]
            ax.plot(g.date,g[signal],label=f'Penalty {penalty}',lw=1.5)
        ax.axhline(1e-5,color='0.35',lw=1,ls='--',label='Fixed prior: 1e-5')
        ax.set(ylabel=f'{signal} coefficient');ax.legend();ax.grid(alpha=.2)
    axes[0].set_title(f'2024 Q4 · Barra-residualized DMD + inflation · {half}-session half-life')
    axes[1].set_xlabel('Execution date')
    fig.tight_layout();fig.savefig(OUT/f'fixed_prior_penalties_half_life_{half}.png',dpi=180);plt.close(fig)
