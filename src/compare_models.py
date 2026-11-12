from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import gzip
import json
import math
import sys

import numpy as np

import tune_l2 as tune

ROOT = tune.REPO / 'runs/model_comparison'
PREVIOUS = tune.ROOT

BASELINE = tune.baseline
SPOT_FEATURES = BASELINE.BINANCE_FEATURES
OFFSET_GRID = tune.PENALTIES
DEPTHS = (2, 3)
RATES = (0.03, 0.08)
ROUNDS = (100, 200, 300)
XGB_FIXED = {'objective': 'binary:logistic', 'booster': 'gbtree', 'tree_method': 'hist',
             'base_score': 0.5, 'min_child_weight': 5, 'reg_lambda': 1.0, 'reg_alpha': 0.0,
             'subsample': 1.0, 'colsample_bytree': 1.0, 'seed': 20261110, 'nthread': 2}
sys.stdout.reconfigure(encoding='utf-8')


def matrix(rows, features):
    result = np.array([[row[name] for name in features] for row in rows], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('Non-finite model feature')
    return result


def offset_fit(train, valid, penalty):
    train_x = matrix(train, SPOT_FEATURES)
    valid_x = matrix(valid, SPOT_FEATURES)
    y = tune.outcomes(train)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(train)), (train_x - mean) / scale))
    offset = np.array([row['poly_logit_last'] for row in train])
    ridge = np.array([0, *([penalty] * train_x.shape[1])])
    coef = np.zeros(design.shape[1])

    def objective(weights):
        margin = offset + design @ weights
        return np.mean(np.logaddexp(0, margin) - y * margin) + np.dot(ridge * weights, weights) / 2

    for iteration in range(100):
        probability = BASELINE.sigmoid(offset + design @ coef)
        gradient = design.T @ (probability - y) / len(y) + ridge * coef
        if np.max(np.abs(gradient)) < 1e-9:
            break
        hessian = (design.T * (probability * (1 - probability))) @ design / len(y) + np.diag(ridge)
        step = np.linalg.solve(hessian, gradient)
        current = objective(coef)
        for k in range(24):
            fraction = 0.5 ** k
            candidate = coef - fraction * step
            if objective(candidate) <= current - 1e-4 * fraction * np.dot(gradient, step) + 1e-15:
                coef = candidate
                break
        else:
            raise ValueError('Offset optimizer line search failed')
    else:
        raise ValueError('Offset model did not converge')
    valid_offset = np.array([row['poly_logit_last'] for row in valid])
    prediction = BASELINE.sigmoid(valid_offset + coef[0] + ((valid_x - mean) / scale) @ coef[1:])
    return {'features': SPOT_FEATURES, 'mean': mean.tolist(), 'std': scale.tolist(), 'coef': coef.tolist(),
            'iterations': iteration, 'offset_feature': 'poly_logit_last', 'offset_coefficient': 1.0,
            'l2_penalty': penalty, 'market_probability_clip': BASELINE.PROB_FLOOR}, prediction


def tree_id(depth, rate, rounds):
    return f'{depth}/{rate:g}/{rounds}'


def dm(rows, with_labels=False):
    return xgb.DMatrix(matrix(rows, tune.FEATURES), label=tune.outcomes(rows) if with_labels else None,
                       feature_names=list(tune.FEATURES), nthread=XGB_FIXED['nthread'])


def tree_grid(train, valid):
    training, evaluation = dm(train, True), dm(valid)
    predictions = {}
    for depth in DEPTHS:
        for rate in RATES:
            model = xgb.train({**XGB_FIXED, 'max_depth': depth, 'eta': rate}, training, num_boost_round=max(ROUNDS))
            for rounds in ROUNDS:
                predictions[tree_id(depth, rate, rounds)] = model.predict(evaluation, iteration_range=(0, rounds)).astype(float)
    return predictions


def evaluate_predictions(y, predictions):
    return {key: BASELINE.scores(y, value) for key, value in predictions.items()}


def verify_reference(row, ref):
    if row['date'] != ref['date'] or row['target_up'] != ref['target_up'] or not math.isclose(row['poly_probability'], ref['market'], abs_tol=1e-12):
        raise ValueError(f'Previous comparison population changed: {row["event_id"]}')


def rolling(development, reference, prior_l2):
    december_boundary = int(datetime(2025, 12, 1, tzinfo=tune.BJ).timestamp())
    prior = {row['event_id']: row for row in reference}
    scored, folds = [], []
    for month in tune.VALID_MONTHS:
        left = int(datetime(2025, month, 1, tzinfo=tune.BJ).timestamp())
        right = int(datetime(2025, month + 1, 1, tzinfo=tune.BJ).timestamp())
        train = [row for row in development if row['decision_s'] < left and row['closed_s'] <= left]
        valid = [row for row in development if left <= row['decision_s'] < right and row['closed_s'] <= december_boundary]
        if not train or not valid or max(row['decision_s'] for row in train) >= min(row['decision_s'] for row in valid):
            raise ValueError('Invalid chronological fold')
        for row in valid:
            verify_reference(row, prior[row['event_id']])
        offsets = {tune.key(p): offset_fit(train, valid, p)[1] for p in OFFSET_GRID}
        trees = tree_grid(train, valid)
        labels = tune.outcomes(valid)
        market = tune.prices(valid)
        logistic = np.array([prior[row['event_id']]['predictions'][tune.key(prior_l2)] for row in valid])
        folds.append({'month': f'2025-{month:02}', 'train_n': len(train), 'n': len(valid),
                      'offset': evaluate_predictions(labels, offsets), 'xgboost': evaluate_predictions(labels, trees),
                      'market': BASELINE.scores(labels, market), 'l2_joint': BASELINE.scores(labels, logistic)})
        scored.extend({'event_id': row['event_id'], 'date': row['date'], 'target_up': row['target_up'],
                       'market': float(market[i]), 'l2_joint': float(logistic[i]),
                       'offset': {key: float(p[i]) for key, p in offsets.items()},
                       'xgboost': {key: float(p[i]) for key, p in trees.items()}}
                      for i, row in enumerate(valid))
        print(f'Fold {month}: {len(train)} train, {len(valid)} validation', flush=True)
    if len(scored) != len(prior) or {row['event_id'] for row in scored} != set(prior):
        raise ValueError('Rolling validation population changed')
    y = tune.outcomes(scored)
    pooled = {family: {key: BASELINE.scores(y, np.array([row[family][key] for row in scored]))
                       for key in scored[0][family]} for family in ('offset', 'xgboost')}
    pooled.update({name: BASELINE.scores(y, np.array([row[name] for row in scored])) for name in ('market', 'l2_joint')})
    return scored, folds, pooled


def main():
    global xgb
    parser = argparse.ArgumentParser()
    parser.add_argument('--arrow-path')
    args = parser.parse_args()
    import xgboost as xgb

    development, december = tune.load(args.arrow_path)
    prior_metrics = json.loads((PREVIOUS / 'metrics.json').read_text(encoding='utf-8'))
    prior_l2 = prior_metrics['selected_l2']
    with gzip.open(PREVIOUS / 'predictions.json.gz', 'rt', encoding='utf-8') as file:
        reference = json.load(file)
    scored, folds, pooled = rolling(development, reference['rolling'], prior_l2)
    offset_id = min(pooled['offset'], key=lambda k: (pooled['offset'][k]['brier'], -float(k)))
    chosen_tree = min(((depth, rate, rounds) for depth in DEPTHS for rate in RATES for rounds in ROUNDS),
                      key=lambda t: (pooled['xgboost'][tree_id(*t)]['brier'], t[0], t[2], t[1]))
    xgb_id = tree_id(*chosen_tree)
    if any(row['closed_s'] > int(datetime(2025, 12, 1, tzinfo=tune.BJ).timestamp()) for row in development):
        raise ValueError('Training labels crossed into December')
    depth, rate, rounds = chosen_tree
    params = {**XGB_FIXED, 'max_depth': depth, 'eta': rate}
    tree = xgb.train(params, dm(development, True), num_boost_round=rounds)
    test_matrix = dm(december)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'xgboost_model.json'
    tree.save_model(path)
    reloaded = xgb.Booster(model_file=path)
    p_tree = tree.predict(test_matrix).astype(float)
    if not np.array_equal(p_tree, reloaded.predict(test_matrix)) or reloaded.num_boosted_rounds() != rounds:
        raise ValueError('Saved XGBoost does not reproduce December predictions')
    correction, p_offset = offset_fit(development, december, float(offset_id))
    prior = {row['event_id']: row for row in reference['december']}
    if len(prior) != len(december):
        raise ValueError('December coverage changed')
    for row in december:
        verify_reference(row, prior[row['event_id']])
    labels = tune.outcomes(december)
    dates = np.array([row['date'] for row in december])
    p_market = tune.prices(december)
    p_l2 = np.array([prior[row['event_id']]['selected'] for row in december])
    forecasts = {'market': p_market, 'l2_joint': p_l2, 'offset': p_offset, 'xgboost': p_tree}
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'xgboost_version': xgb.__version__,
              'input_sha256': sha256((tune.DATA / 'model_features.parquet').read_bytes()).hexdigest(),
              'selection_metric': 'pooled rolling Brier on August-November 2025',
              'offset_grid': OFFSET_GRID, 'tree_grid': {'depths': DEPTHS, 'rates': RATES, 'rounds': ROUNDS},
              'fixed_xgboost': XGB_FIXED, 'selected_offset': offset_id, 'selected_xgboost': xgb_id,
              'selected_l2_baseline': prior_l2, 'rolling_n': len(scored), 'folds': folds, 'pooled': pooled,
              'december_train_n': len(development), 'december_n': len(december),
              'december': evaluate_predictions(labels, forecasts),
              'december_delta': {name: {ref: BASELINE.delta(labels, forecasts[name], forecasts[ref], dates)
                                         for ref in ('market', 'l2_joint')} for name in ('offset', 'xgboost')},
              'offset_model': correction}
    tune.save(ROOT / 'metrics.json', result)
    tune.save(ROOT / 'offset_model.json', correction)
    december_scores = [{'event_id': row['event_id'], 'date': row['date'], 'target_up': row['target_up'],
                        **{key: float(p[i]) for key, p in forecasts.items()}} for i, row in enumerate(december)]
    with gzip.open(ROOT / 'predictions.json.gz', 'wt', encoding='utf-8') as file:
        json.dump({'rolling': scored, 'december': december_scores}, file, ensure_ascii=False, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('xgboost_version', 'rolling_n', 'selected_offset', 'selected_xgboost',
                     'pooled', 'december', 'december_delta')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
