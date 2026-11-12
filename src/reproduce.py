"""Re-run the published experiments using the packaged data, then check the results."""
import json
import math
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    'l2': ('tune_l2.py', ('selected_l2', 'rolling_n', 'december_n', 'pooled', 'december')),
    'model_comparison': ('compare_models.py', ('selected_offset', 'selected_xgboost', 'rolling_n',
                                               'december_n', 'pooled', 'december')),
    'trade_flow': ('compare_trade_flow.py', ('chosen_l2', 'feature_audit', 'rolling', 'december_review')),
}


def check(actual, expected, path):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'Fields differ at {path}')
        for key in expected:
            check(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'List size differs at {path}')
        for i, (item, reference) in enumerate(zip(actual, expected)):
            check(item, reference, f'{path}[{i}]')
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-7):
            raise ValueError(f'{path}: expected {expected}, got {actual}')
    elif actual != expected:
        raise ValueError(f'{path}: expected {expected}, got {actual}')


def main():
    for name, (script, keys) in EXPERIMENTS.items():
        print(f'Running {name}', flush=True)
        completed = subprocess.run([sys.executable, '-B', str(REPO / 'src' / script)],
                                   cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   encoding='utf-8', errors='replace', env=None)
        if completed.returncode:
            print(completed.stdout)
            raise SystemExit(completed.returncode)
        run_dir = REPO / 'runs' / name
        (run_dir / 'run.log').write_text(completed.stdout, encoding='utf-8')
        actual = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
        expected = json.loads((REPO / 'results' / name / 'metrics.json').read_text(encoding='utf-8'))
        for key in keys:
            check(actual[key], expected[key], f'{name}.{key}')
        print(f'{name}: sample counts, selected parameters and scores verified.', flush=True)


if __name__ == '__main__':
    main()
