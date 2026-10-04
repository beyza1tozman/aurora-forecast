"""Multiclass LightGBM per horizon + temperature scaling.

One booster per horizon predicts the Kp class {<=3, 4, 5, 6, 7+}. Exceedance
probabilities are tail sums (aurora.dataset.exceedance_from_class_probs), so
they are monotonic by construction. No class weights: they would inflate rare
classes and break calibration. Instead a single temperature T, fitted on the
validation set, rescales the logits: p = softmax(z / T).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar

from aurora.config import ROOT
from aurora.dataset import N_CLASSES
from aurora.features import FEATURE_COLUMNS

MODELS_DIR = ROOT / "models"

PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 0,
}
MAX_ROUNDS = 3000
EARLY_STOPPING = 100


@dataclass
class HorizonModel:
    horizon: int
    booster: lgb.Booster
    temperature: float = 1.0

    def predict_proba(self, X) -> np.ndarray:
        """Calibrated class probabilities, shape (n, N_CLASSES)."""
        logits = self.booster.predict(X[FEATURE_COLUMNS], raw_score=True)
        return softmax(logits / self.temperature)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of integer classes y under softmax(logits / T)."""
    p = softmax(logits / temperature)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None))))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising validation NLL (searched on log T for stability)."""
    res = minimize_scalar(
        lambda log_t: nll(logits, y, np.exp(log_t)), bounds=(-3, 3), method="bounded"
    )
    return float(np.exp(res.x))


def train_horizon(X_train, y_train, X_val, y_val, horizon: int, params=None) -> HorizonModel:
    """Train with early stopping on val, then fit the temperature on the same val set."""
    params = {**PARAMS, **(params or {})}
    train_set = lgb.Dataset(X_train[FEATURE_COLUMNS], label=y_train)
    val_set = lgb.Dataset(X_val[FEATURE_COLUMNS], label=y_val, reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    logits = booster.predict(X_val[FEATURE_COLUMNS], raw_score=True)
    temperature = fit_temperature(logits, np.asarray(y_val, dtype=int))
    return HorizonModel(horizon, booster, temperature)


def save_models(models: list[HorizonModel], directory: Path = MODELS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    calibration = {}
    for m in models:
        m.booster.save_model(
            directory / f"kp_h{m.horizon}.txt", num_iteration=m.booster.best_iteration
        )
        calibration[str(m.horizon)] = {
            "temperature": m.temperature,
            "best_iteration": m.booster.best_iteration,
        }
    meta = {"features": FEATURE_COLUMNS, "horizons": calibration}
    (directory / "calibration.json").write_text(json.dumps(meta, indent=2))


def load_models(directory: Path = MODELS_DIR) -> dict[int, HorizonModel]:
    meta = json.loads((directory / "calibration.json").read_text())
    if meta["features"] != FEATURE_COLUMNS:
        raise ValueError("saved models were trained on a different feature list")
    return {
        int(h): HorizonModel(
            int(h),
            lgb.Booster(model_file=str(directory / f"kp_h{h}.txt")),
            cal["temperature"],
        )
        for h, cal in meta["horizons"].items()
    }
