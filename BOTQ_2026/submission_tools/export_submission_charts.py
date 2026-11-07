"""Export embedded notebook figures for direct viewing outside submission."""
import base64
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
in_repo = (repo / 'BOTQ_2026').is_dir()
root = repo if in_repo else Path(__file__).resolve().parents[1]
out = root / ('botq2026-final-charts' if in_repo else 'outputs/botq2026-final-charts')
out.mkdir(exist_ok=True)
names = {
    '03_Final_charts_and_risk_checks': [
        'active_excess_log', 'active_total_log', 'active_drawdown',
        'active_rolling_information_ratio', 'active_predicted_te',
        'long_only_excess_log', 'long_only_total_log', 'long_only_drawdown',
        'long_only_rolling_information_ratio', 'long_only_predicted_te',
        'active_style_exposure', 'long_only_style_exposure',
        'ridge_coefficients_1260', 'ridge_coefficients_252', 'ridge_coefficients_63',
        'bma_weights_active', 'bma_weights_long_only',
        'active_factor_exposure', 'active_factor_variance',
        'long_only_factor_exposure', 'long_only_factor_variance',
        'dmd_style_fit',
    ],
    '04_Penalty_independent_1pct_results': [
        'standalone_active_excess_log', 'standalone_active_style_exposure',
        'standalone_long_only_excess_log', 'standalone_long_only_style_exposure',
        'standalone_dmd_style_fit',
    ],
}
for stem, labels in names.items():
    notebook_dir = root / ('BOTQ_submission' if in_repo else 'outputs/BOTQ_submission')
    notebook = json.loads((notebook_dir / (stem + '.ipynb')).read_text())
    images = [output['data']['image/png']
              for cell in notebook['cells'] for output in cell.get('outputs', [])
              if 'image/png' in output.get('data', {})]
    assert len(images) == len(labels), (stem, len(images), len(labels))
    for label, encoded in zip(labels, images):
        if isinstance(encoded, list):
            encoded = ''.join(encoded)
        (out / (label + '.png')).write_bytes(base64.b64decode(encoded))
print(out)
