"""Render presentation notebooks without adding helper files to submission."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

repo = Path(__file__).resolve().parents[2]
root = (repo / 'BOTQ_submission') if (repo / 'BOTQ_2026').is_dir() else (Path(__file__).resolve().parents[1] / 'outputs/BOTQ_submission')
parser = argparse.ArgumentParser()
parser.add_argument('notebook', choices=['01_Research_and_existing_evidence',
                                         '02_Final_model_and_backtest',
                                         '03_Final_charts_and_risk_checks',
                                         '04_Penalty_independent_1pct_results'])
args = parser.parse_args()
path = root / (args.notebook + '.ipynb')
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
nbformat.write(nb, path)
print(path)
