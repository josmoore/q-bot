"""Execute the final notebook backtest on a Slurm compute node."""
import json
import os
from pathlib import Path

root = Path.home() / 'nobackup/autodelete/botq2026/submission'
os.chdir(root)
source = Path.home() / 'q-bot/submission_final.ipynb'
notebook = json.loads(source.read_text())
scope = {'__name__': '__submission__'}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), str(source), 'exec'), scope)
assert scope['PENALTIES'] == {63: 18, 252: 5, 1260: 2.5}
assert scope['FIXED_PRIOR'] == 1e-5
scope['run_final']()
