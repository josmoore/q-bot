"""Execute notebook definitions and penalty-independent work in Slurm."""
import json
import os
from pathlib import Path

root = Path.home() / 'nobackup/autodelete/botq2026/submission'
(root / 'dataset').mkdir(parents=True, exist_ok=True)
os.chdir(root)
source = Path.home() / 'q-bot/submission_setup.ipynb'
notebook = json.loads(source.read_text())
scope = {'__name__': '__submission__'}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), str(source), 'exec'), scope)
scope['prepare_dataset']()
scope['run_final'](baselines_only=True)
