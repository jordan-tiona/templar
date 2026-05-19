"""Temporal Fusion Transformer trainer and predictor."""

import logging
import time
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torch.utils.data import DataLoader

from config.settings import MODEL_TFT, SEQUENCE_LENGTH, PREDICTION_HORIZON
from .dataset import TARGET, TIME_VARYING_UNKNOWN, STATIC_CATEGORICALS

log = logging.getLogger(__name__)


class EpochLogger(Callback):
    """Logs epoch timing and loss to the standard logger after each validation pass."""

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        elapsed = time.time() - self._epoch_start
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss", float("nan")))
        val_loss = metrics.get("val_loss", float("nan"))
        epoch = trainer.current_epoch + 1
        remaining = (trainer.max_epochs - epoch) * elapsed
        log.info(
            "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f  elapsed: %.0fs  ~remaining: %.0fm",
            epoch, trainer.max_epochs,
            float(train_loss), float(val_loss),
            elapsed, remaining / 60,
        )


def _make_timeseries_dataset(
    df: pd.DataFrame,
    reference_dataset: TimeSeriesDataSet | None = None,
) -> TimeSeriesDataSet:
    feature_cols = [c for c in TIME_VARYING_UNKNOWN if c in df.columns]

    if reference_dataset is not None:
        return TimeSeriesDataSet.from_dataset(reference_dataset, df, predict=True, stop_randomization=True)

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=STATIC_CATEGORICALS,
        min_encoder_length=SEQUENCE_LENGTH // 2,
        max_encoder_length=SEQUENCE_LENGTH,
        min_prediction_length=1,
        max_prediction_length=PREDICTION_HORIZON,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(groups=STATIC_CATEGORICALS, transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


def _latest_checkpoint() -> str | None:
    checkpoints = sorted(MODEL_TFT.glob("*.ckpt"))
    return str(checkpoints[-1]) if checkpoints else None


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_size: int = 64,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    resume: bool = True,
) -> TemporalFusionTransformer:
    train_ds = _make_timeseries_dataset(train_df)
    val_ds = _make_timeseries_dataset(val_df, reference_dataset=train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    resume_ckpt = _latest_checkpoint() if resume else None
    if resume_ckpt:
        log.info("Resuming from checkpoint: %s", resume_ckpt)
    else:
        log.info("Starting TFT training from scratch")

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=MODEL_TFT,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        save_last=True,   # always keep latest for resuming
        mode="min",
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=8, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # uses GPU if available, falls back to CPU
        gradient_clip_val=0.1,
        callbacks=[checkpoint_cb, early_stop_cb, EpochLogger()],
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_loader, val_loader, ckpt_path=resume_ckpt)

    return tft


def load(checkpoint_path: str | None = None) -> TemporalFusionTransformer:
    if checkpoint_path is None:
        checkpoints = sorted(MODEL_TFT.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No TFT checkpoint found in {MODEL_TFT}")
        checkpoint_path = checkpoints[-1]

    return TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))


def predict(model: TemporalFusionTransformer, df: pd.DataFrame, train_df: pd.DataFrame) -> pd.Series:
    """
    Return median predicted 5-day forward return per row.
    df must have the same feature columns as train_df.
    """
    train_ds = _make_timeseries_dataset(train_df)
    pred_ds = _make_timeseries_dataset(df, reference_dataset=train_ds)
    loader = DataLoader(pred_ds, batch_size=128, shuffle=False, num_workers=2)

    raw_predictions, index = model.predict(loader, mode="prediction", return_index=True)
    # QuantileLoss returns multiple quantiles; take the median (index 3 of 7 default quantiles)
    median_idx = len(model.loss.quantiles) // 2
    preds = raw_predictions[:, median_idx].numpy()

    return pd.Series(preds, index=index.index, name="tft_pred")
