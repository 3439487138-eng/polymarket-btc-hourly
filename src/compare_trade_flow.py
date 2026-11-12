"""Compare an order-flow extension with the frozen 2025 L2 forecasts."""
import argparse
import gzip
import json
from datetime import datetime

import numpy as np

import tune_l2 as tune

ROOT = tune.REPO / 'runs/trade_flow'
PRIOR = tune.ROOT

FLOW = ('spot_trade_imbalance_5m', 'perp_trade_imbalance_5m')
FEATURES = tune.FEATURES + FLOW
MICROS = 1_000_000


def add_trade_flow(rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    windows = np.array([r['feature_cut_s'] for r in rows], dtype=np.int64)
    expected = (windows[:, None] - 60 * np.arange(5, 0, -1)) * MICROS
    volume_fields = ['volume_btc', 'taker_buy_volume_btc']
    audit = {}
    for venue, name in zip(('spot', 'perp'), FLOW):
        table = pq.read_table(tune.DATA / f'binance_{venue}_BTCUSDT_1m.parquet',
                              columns=['open_time_utc', 'close_time_utc', 'trade_count', *volume_fields])
        opens = table['open_time_utc'].cast(pa.int64()).to_numpy()
        if np.any(np.diff(opens) <= 0):
            raise ValueError(f'Unsorted or duplicate minute: {venue}')
        positions = np.searchsorted(opens, expected)
        if np.any(positions >= len(opens)) or not np.array_equal(opens[positions], expected):
            raise ValueError(f'At least one of the five required minutes is missing: {venue}')
        closes = table['close_time_utc'].cast(pa.int64()).to_numpy()[positions]
        resolution = 1 if venue == 'spot' else 1000
        if not np.array_equal(closes, expected + 60 * MICROS - resolution):
            raise ValueError(f'Unexpected candle close time: {venue}')
        if not np.all(closes < windows[:, None] * MICROS):
            raise ValueError(f'At least one candle is past the feature cutoff: {venue}')
        volume, buys = (table[field].to_numpy()[positions] for field in volume_fields)
        if (not np.isfinite(volume).all() or not np.isfinite(buys).all()
                or np.any(buys < 0) or np.any(volume < buys - 1e-9)):
            raise ValueError(f'Invalid taker/total volume: {venue}')
        total = volume.sum(axis=1)
        empty = total == 0
        trade_counts = table['trade_count'].to_numpy()[positions]
        if np.any(trade_counts[empty] != 0) or np.any(buys[empty] != 0):
            raise ValueError(f'Zero volume with reported trades: {venue}')
        imbalance = np.divide(2 * buys.sum(axis=1) - total, total,
                              out=np.zeros_like(total), where=~empty)
        if np.any(np.abs(imbalance) > 1 + 1e-10):
            raise ValueError(f'Imbalance out of range: {venue}')
        for row, value in zip(rows, imbalance):
            row[name] = float(value)
        audit[venue] = {'windows': len(imbalance), 'bars_checked': positions.size,
                        'min': float(imbalance.min()), 'max': float(imbalance.max()),
                        'zero_volume_events_neutral_value': [r['event_id'] for r, flag in zip(rows, empty) if flag]}
    return audit


def fit(train, valid, penalty):
    old = tune.baseline.RIDGE
    tune.baseline.RIDGE = penalty
    try:
        return tune.baseline.fit(train, valid, FEATURES)
    finally:
        tune.baseline.RIDGE = old


def evaluate(rows, market, original, forecasts):
    y = tune.outcomes(rows)
    days = np.array([r['date'] for r in rows])
    return {
        'n': len(y),
        'metrics': {name: tune.baseline.scores(y, p)
                    for name, p in {'market': market, 'original': original, **forecasts}.items()},
        'differences': {name: {
            'vs_original': tune.baseline.delta(y, p, original, days),
            'vs_market': tune.baseline.delta(y, p, market, days),
        } for name, p in forecasts.items()},
    }


def verify_alignment(rows, prior):
    current_keys = [(r['event_id'], r['date'], r['target_up'], r['poly_probability']) for r in rows]
    prior_keys = [(r['event_id'], r['date'], r['target_up'], r['market']) for r in prior]
    if current_keys != prior_keys:
        raise ValueError('Samples do not match the prior forecasts in order, outcome and market probability')


def records(rows, market, original, forecasts):
    return [{'event_id': r['event_id'], 'date': r['date'], 'target_up': r['target_up'],
             **{key: r[key] for key in FLOW}, 'market': float(market[i]), 'original': float(original[i]),
             'augmented': {key: float(values[i]) for key, values in forecasts.items()}}
            for i, r in enumerate(rows)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arrow-path')
    args = parser.parse_args()
    development, december = tune.load(args.arrow_path)
    audit = add_trade_flow(development + december)
    with gzip.open(PRIOR / 'predictions.json.gz', 'rt', encoding='utf-8') as source:
        reference = json.load(source)
    previous = json.loads((PRIOR / 'metrics.json').read_text(encoding='utf-8'))
    original_key = tune.key(previous['selected_l2'])
    december_start = int(datetime(2025, 12, 1, tzinfo=tune.BJ).timestamp())
    if any(r['closed_s'] > december_start for r in development):
        raise ValueError('A development outcome arrived after the December boundary')

    roll_rows, month_counts = [], {}
    monthly = {tune.key(penalty): [] for penalty in tune.PENALTIES}
    for month in tune.VALID_MONTHS:
        left = int(datetime(2025, month, 1, tzinfo=tune.BJ).timestamp())
        right = int(datetime(2025, month + 1, 1, tzinfo=tune.BJ).timestamp())
        train = [r for r in development if r['decision_s'] < left and r['closed_s'] <= left]
        valid = [r for r in development
                 if left <= r['decision_s'] < right and r['closed_s'] <= december_start]
        month_counts[f'2025-{month:02}'] = {'train': len(train), 'validation': len(valid)}
        roll_rows.extend(valid)
        for penalty in tune.PENALTIES:
            monthly[tune.key(penalty)].append(fit(train, valid, penalty)[1])

    verify_alignment(roll_rows, reference['rolling'])
    roll_market = tune.prices(roll_rows)
    roll_original = np.array([r['predictions'][original_key] for r in reference['rolling']])
    roll_predictions = {key: np.concatenate(parts) for key, parts in monthly.items()}
    rolling = evaluate(roll_rows, roll_market, roll_original, roll_predictions)
    chosen = min(tune.PENALTIES, key=lambda p: (rolling['metrics'][tune.key(p)]['brier'], -p))
    selected_key = tune.key(chosen)

    verify_alignment(december, reference['december'])
    december_market = tune.prices(december)
    december_original = np.array([r['selected'] for r in reference['december']])
    _, refit = tune.fit(development, december, previous['selected_l2'])
    np.testing.assert_allclose(refit, december_original, rtol=0, atol=1e-13)
    model, predictions = fit(development, december, chosen)
    december_predictions = {selected_key: predictions}
    reviewed = evaluate(december, december_market, december_original, december_predictions)

    result = {
        'feature_window_minutes': [24, 29],
        'feature_window_end_exclusive': True,
        'decision_minute': 30,
        'feature_audit': audit,
        'development_n': len(development),
        'original_l2': previous['selected_l2'],
        'chosen_l2': chosen,
        'folds': month_counts,
        'rolling': rolling,
        'december_review': reviewed,
        '2026_scored': False,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    tune.save(ROOT / 'metrics.json', result)
    tune.save(ROOT / 'model.json', {'l2_penalty': chosen, 'model': model,
                                    'feature_window_minutes': [24, 29], 'decision_minute': 30})
    with gzip.open(ROOT / 'predictions.json.gz', 'wt', encoding='utf-8') as out:
        json.dump({'rolling': records(roll_rows, roll_market, roll_original, roll_predictions),
                   'december': records(december, december_market, december_original, december_predictions)},
                  out, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
