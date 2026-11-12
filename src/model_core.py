"""Shared L2 regression and scoring functions."""
import numpy as np

BINANCE_FEATURES = ('spot_return_29m_bp', 'spot_return_5m_bp', 'spot_realized_vol_29m_bp', 'perp_basis_bp', 'perp_basis_change_bp')
POLY_FEATURES = ('poly_logit_last',)
RIDGE = 0.03
PROB_FLOOR = 0.005
BOOTSTRAP = 5000
SEED = 20261110


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit(train, validation, features):
    x = np.array([[r[key] for key in features] for r in train])
    xv = np.array([[r[key] for key in features] for r in validation])
    assert np.isfinite(x).all() and np.isfinite(xv).all()
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1
    design = np.column_stack((np.ones(len(x)), (x - mean) / std))
    y = np.array([r['target_up'] for r in train])
    if np.unique(y).size != 2:
        raise ValueError('Training data must contain both outcomes')
    reg = np.diag([0] + [RIDGE] * len(features))
    coef = np.zeros(design.shape[1])
    for iteration in range(100):
        prob = sigmoid(design @ coef)
        gradient = design.T @ (prob - y) / len(y) + reg @ coef
        if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
            break
        hessian = (design.T * (prob * (1 - prob))) @ design / len(y) + reg
        step = np.linalg.solve(hessian, gradient)
        coef -= step
    else:
        raise ValueError('Logistic regression did not converge')
    preds = sigmoid(coef[0] + ((xv - mean) / std) @ coef[1:])
    model = {'features': features, 'mean': mean.tolist(), 'std': std.tolist(), 'coef': coef.tolist(), 'iterations': iteration}
    return model, preds


def scores(y, probs):
    log_p = np.clip(probs, PROB_FLOOR, 1 - PROB_FLOOR)
    return {'accuracy': float(np.mean((probs >= 0.5) == y)), 'brier': float(np.mean((probs - y) ** 2)),
            'log_loss': float(np.mean(-y * np.log(log_p) - (1 - y) * np.log(1 - log_p)))}


def delta(y, joint, comparison, days):
    losses = (joint - y) ** 2 - (comparison - y) ** 2
    unique = np.unique(days)
    groups = [losses[days == day] for day in unique]
    totals = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    indexes = np.random.default_rng(SEED).integers(len(groups), size=(BOOTSTRAP, len(groups)))
    draws = totals[indexes].sum(axis=1) / sizes[indexes].sum(axis=1)
    return {'delta_brier': float(losses.mean()), 'day_bootstrap_95_interval': np.quantile(draws, [0.025, 0.975]).tolist()}
