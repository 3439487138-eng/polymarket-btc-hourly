from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import gzip
import json
import math
import sys

import numpy as np

import model_core as baseline

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'
ROOT = REPO / 'runs/l2'
REFERENCE = DATA / 'reference/expanded_baseline_metrics.json'

FEATURES = baseline.BINANCE_FEATURES + baseline.POLY_FEATURES
PENALTIES = (0.003, 0.01, 0.03, 0.1, 0.3)
VALID_MONTHS = (8, 9, 10, 11)
BJ = timezone(timedelta(hours=8))
sys.stdout.reconfigure(encoding='utf-8')


def timestamp(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fit(train, validation, penalty):
    fixed = baseline.RIDGE
    baseline.RIDGE = penalty
    try:
        return baseline.fit(train, validation, FEATURES)
    finally:
        baseline.RIDGE = fixed


def outcomes(rows):
    return np.array([r['target_up'] for r in rows])


def prices(rows):
    return np.array([r['poly_probability'] for r in rows])


def key(penalty):
    return f'{penalty:g}'


def load(arrow_path):
    if arrow_path:
        sys.path.append(arrow_path)
    import pyarrow.parquet as pq

    columns = ['event_id', 'date', 'start_bj', 'start_utc', 'split', 'target_up', 'feature_cut_s',
               'decision_s', 'poly_trade_timestamp_s', 'poly_probability', *FEATURES]
    rows = pq.read_table(DATA / 'model_features.parquet', columns=columns).to_pylist()
    markets = {row['event_id']: row for row in pq.read_table(DATA / 'market_coverage.parquet',
                columns=['event_id', 'closed_utc', 'winner']).to_pylist()}
    development = []
    december = []
    for row in rows:
        start = timestamp(row['start_utc'])
        market = markets[row['event_id']]
        row['closed_s'] = timestamp(market['closed_utc'])
        if not (row['date'] == row['start_bj'][:10] and row['date'].startswith('2025-')
                and start + 1440 <= row['poly_trade_timestamp_s'] < row['feature_cut_s'] == start + 1740
                and row['decision_s'] == start + 1800 and row['closed_s'] >= start + 3600
                and market['winner'] in ('Up', 'Down') and row['target_up'] == int(market['winner'] == 'Up')):
            raise ValueError(f'Input timing or label mismatch in event {row["event_id"]}')
        if row['split'] == 'train' and row['date'] < '2025-12-01':
            development.append(row)
        elif row['split'] == 'validation' and row['date'].startswith('2025-12'):
            december.append(row)
        else:
            raise ValueError('Unexpected split')
    if (len(development), len(december)) != (4410, 730) or len({row['start_utc'] for row in rows}) != len(rows):
        raise ValueError('The validated sample has changed')
    return sorted(development, key=lambda row: row['decision_s']), sorted(december, key=lambda row: row['decision_s'])


def rolling(development):
    folds, scored = [], []
    for month in VALID_MONTHS:
        start = datetime(2025, month, 1, tzinfo=BJ)
        end = datetime(2025, month + 1, 1, tzinfo=BJ)
        left, right = int(start.timestamp()), int(end.timestamp())
        train = [row for row in development if row['decision_s'] < left and row['closed_s'] <= left]
        valid = [row for row in development if left <= row['decision_s'] < right and row['closed_s'] <= int(datetime(2025, 12, 1, tzinfo=BJ).timestamp())]
        if len(train) < 500 or len(valid) < 500 or max(r['decision_s'] for r in train) >= min(r['decision_s'] for r in valid):
            raise ValueError(f'Invalid month: {month}')
        y = outcomes(valid)
        market = prices(valid)
        forecasts = {key(penalty): fit(train, valid, penalty)[1] for penalty in PENALTIES}
        folds.append({'month': f'2025-{month:02}', 'train': len(train), 'validation': len(valid),
                      'metrics': {name: baseline.scores(y, p) for name, p in forecasts.items()},
                      'market': baseline.scores(y, market)})
        scored.extend({'event_id': r['event_id'], 'date': r['date'], 'target_up': r['target_up'], 'market': float(market[i]),
                       'predictions': {name: float(value[i]) for name, value in forecasts.items()}}
                      for i, r in enumerate(valid))
    if len({row['event_id'] for row in scored}) != len(scored):
        raise ValueError('A validation sample was counted twice')
    labels = outcomes(scored)
    pooled = {key(penalty): baseline.scores(labels, np.array([row['predictions'][key(penalty)] for row in scored]))
              for penalty in PENALTIES}
    market_metrics = baseline.scores(labels, np.array([row['market'] for row in scored]))
    # Choose only from pre-December results. Prefer stronger regularization if exactly tied.
    chosen = min(PENALTIES, key=lambda penalty: (pooled[key(penalty)]['brier'], -penalty))
    return folds, scored, pooled, market_metrics, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arrow-path')
    args = parser.parse_args()
    development, december = load(args.arrow_path)
    folds, roll_rows, pooled, market_metrics, chosen = rolling(development)
    end = int(datetime(2025, 12, 1, tzinfo=BJ).timestamp())
    if any(row['closed_s'] > end for row in development):
        raise ValueError('A training label closed after the December boundary')
    tuned, p_tuned = fit(development, december, chosen)
    _, p_original = fit(development, december, 0.03)
    y, p_market = outcomes(december), prices(december)
    days = np.array([row['date'] for row in december])
    december_metrics = {'market': baseline.scores(y, p_market), 'original': baseline.scores(y, p_original),
                        'selected': baseline.scores(y, p_tuned)}
    prior_metrics = json.loads(REFERENCE.read_text(encoding='utf-8'))['metrics']
    if not all(math.isclose(value, prior_metrics[model][metric], rel_tol=1e-12, abs_tol=1e-12)
               for name, model in [('market', 'last'), ('original', 'joint')]
               for metric, value in december_metrics[name].items()):
        raise ValueError('December reference scores no longer reproduce the previous run')
    december_delta = {'market': baseline.delta(y, p_tuned, p_market, days),
                      'original': baseline.delta(y, p_tuned, p_original, days)}
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'grid': PENALTIES, 'selected_l2': chosen,
              'selection': 'Pooled Brier on rolling August-November 2025, tie prefers stronger L2',
              'folds': folds, 'rolling_n': len(roll_rows), 'pooled': pooled, 'rolling_market': market_metrics,
              'december_train_n': len(development), 'december_n': len(december), 'december': december_metrics,
              'december_delta': december_delta, 'previous_scores_reproduced': True,
              'feature_sha256': sha256((DATA / 'model_features.parquet').read_bytes()).hexdigest()}
    ROOT.mkdir(parents=True, exist_ok=True)
    save(ROOT / 'metrics.json', result)
    save(ROOT / 'model.json', {'l2_penalty': chosen, 'model': tuned, 'prediction_minute': 30,
                              'feature_cutoff_minute': 29, 'probability_clip': baseline.PROB_FLOOR,
                              'training_windows': len(development), 'feature_sha256': result['feature_sha256']})
    december_rows = [{'event_id': row['event_id'], 'date': row['date'], 'target_up': row['target_up'],
                      'market': float(p_market[i]), 'original': float(p_original[i]), 'selected': float(p_tuned[i])}
                     for i, row in enumerate(december)]
    with gzip.open(ROOT / 'predictions.json.gz', 'wt', encoding='utf-8') as output:
        json.dump({'rolling': roll_rows, 'december': december_rows}, output, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
