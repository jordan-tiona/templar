"""XGBoost trainer and predictor (tabular, no sequence)."""

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

from config.settings import MODEL_XGB
from .dataset import TARGET, TIME_VARYING_UNKNOWN

_MODEL_PATH = MODEL_XGB / "xgb_model.joblib"
_SCALER_PATH = MODEL_XGB / "xgb_scaler.joblib"


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in TIME_VARYING_UNKNOWN if c in df.columns]


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

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
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
