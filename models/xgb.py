"""XGBoost trainer and predictor (tabular, no sequence)."""

import json
import logging

import joblib
import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

from config.settings import MODEL_XGB
from .dataset import TARGET, TIME_VARYING_UNKNOWN

log = logging.getLogger(__name__)

_MODEL_PATH = MODEL_XGB / "xgb_model.joblib"
_SCALER_PATH = MODEL_XGB / "xgb_scaler.joblib"
_BEST_PARAMS_PATH = MODEL_XGB / "best_params.json"

# Suppress Optuna's per-trial verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in TIME_VARYING_UNKNOWN if c in df.columns]


def _load_best_params() -> dict | None:
    """Return saved Optuna params if they exist, else None."""
    if _BEST_PARAMS_PATH.exists():
        with open(_BEST_PARAMS_PATH) as f:
            return json.load(f)
    return None


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> XGBRegressor:
    cols = _feature_cols(train_df)

    X_train, y_train = train_df[cols].values, train_df[TARGET].values
    X_val, y_val = val_df[cols].values, val_df[TARGET].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Use best Optuna params if available, otherwise fall back to defaults
    saved = _load_best_params()
    if saved:
        log.info("Using Optuna-tuned params from %s", _BEST_PARAMS_PATH)
        params = saved
    else:
        params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )

    model = XGBRegressor(
        **params,
        early_stopping_rounds=50,
        eval_metric="rmse",
        n_jobs=-1,
        random_state=42,
        device="cpu",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    joblib.dump(model, _MODEL_PATH)
    joblib.dump(scaler, _SCALER_PATH)
    return model


def tune(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 50,
) -> dict:
    """
    Run Optuna hyperparameter search for XGBoost.
    Saves best params to MODEL_XGB/best_params.json.
    Returns the best params dict.
    """
    cols = _feature_cols(train_df)

    X_train, y_train = train_df[cols].values, train_df[TARGET].values
    X_val, y_val = val_df[cols].values, val_df[TARGET].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        }

        model = XGBRegressor(
            **params,
            early_stopping_rounds=50,
            eval_metric="rmse",
            n_jobs=-1,
            random_state=42,
            device="cpu",
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Best val RMSE recorded by XGBoost during early stopping
        val_rmse = model.best_score
        return float(val_rmse)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    log.info("Best Optuna params (val RMSE=%.6f): %s", study.best_value, best_params)

    with open(_BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2)
    log.info("Best params saved to %s", _BEST_PARAMS_PATH)

    return best_params


def load() -> tuple[XGBRegressor, StandardScaler]:
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"No XGBoost model found at {_MODEL_PATH}")
    return joblib.load(_MODEL_PATH), joblib.load(_SCALER_PATH)


def predict(
    model: XGBRegressor,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> pd.Series:
    cols = _feature_cols(df)
    X = scaler.transform(df[cols].values)
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="xgb_pred")
