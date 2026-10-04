"""Model wrapper: calibrated, monotonic probabilities and a save/load round trip."""

import numpy as np
import pandas as pd
import pytest

from aurora.dataset import N_CLASSES, THRESHOLDS, exceedance_from_class_probs
from aurora.features import FEATURE_COLUMNS
from aurora.model import fit_temperature, load_models, nll, save_models, softmax, train_horizon


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    X.iloc[rng.choice(n, 200), 0] = np.nan  # LightGBM must cope with gaps
    y = np.clip(np.round(1 + X["kp_last"] + 0.5 * rng.normal(size=n)), 0, N_CLASSES - 1)
    return X, y.astype(int)


@pytest.fixture(scope="module")
def model(data):
    X, y = data
    return train_horizon(X[:2000], y[:2000], X[2000:], y[2000:], horizon=3)


def test_probabilities_valid_and_exceedance_monotonic(model, data):
    X, _ = data
    p = model.predict_proba(X)
    assert p.shape == (len(X), N_CLASSES)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    exceed = [exceedance_from_class_probs(p, t) for t in THRESHOLDS]
    for lower, higher in zip(exceed, exceed[1:], strict=False):
        assert (higher <= lower + 1e-12).all()


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(1)
    true_logits = rng.normal(size=(5000, N_CLASSES))
    y = np.array([rng.choice(N_CLASSES, p=row) for row in softmax(true_logits)])
    overconfident = 3 * true_logits
    t = fit_temperature(overconfident, y)
    assert t == pytest.approx(3, rel=0.2)
    assert nll(overconfident, y, t) < nll(overconfident, y)


def test_save_load_round_trip(model, data, tmp_path):
    X, _ = data
    save_models([model], tmp_path)
    loaded = load_models(tmp_path)[3]
    assert loaded.temperature == pytest.approx(model.temperature)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-6)
