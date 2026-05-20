"""Temporal Fusion Transformer trainer and predictor."""

import logging
import time
import pandas as pd
import torch
import torch.serialization
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback

from config.settings import MODEL_TFT, SEQUENCE_LENGTH, PREDICTION_HORIZON, TFT_LEARNING_RATE
from .dataset import TARGET, TIME_VARYING_UNKNOWN, STATIC_CATEGORICALS

# Allow pytorch-forecasting classes when loading checkpoints in PyTorch 2.6+
torch.serialization.add_safe_globals([
    GroupNormalizer, NaNLabelEncoder, TorchNormalizer, EncoderNormalizer,
    pd.DataFrame,
])

log = logging.getLogger(__name__)

# Use Tensor Cores on Ampere+ GPUs (3080 Ti, etc.) for ~2x throughput with minimal precision loss
torch.set_float32_matmul_precision("medium")


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
    predict: bool = False,
) -> TimeSeriesDataSet:
    feature_cols = [c for c in TIME_VARYING_UNKNOWN if c in df.columns]

    if reference_dataset is not None:
        return TimeSeriesDataSet.from_dataset(reference_dataset, df, predict=predict, stop_randomization=True)

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
    # Prefer best checkpoint over last.ckpt
    best = [c for c in MODEL_TFT.glob("*.ckpt") if c.name != "last.ckpt"]
    return str(sorted(best)[-1]) if best else None


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    max_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = TFT_LEARNING_RATE,
    hidden_size: int = 96,
    attention_head_size: int = 4,
    dropout: float = 0.3,
    resume: bool = False,
) -> TemporalFusionTransformer:
    train_ds = _make_timeseries_dataset(train_df)

    # Val dataset needs encoder context from training tail; prepend it per symbol
    val_with_context = pd.concat([
        pd.concat([
            grp_train.iloc[-SEQUENCE_LENGTH:],
            val_df[val_df["symbol"] == sym],
        ])
        for sym, grp_train in train_df.groupby("symbol")
        if sym in val_df["symbol"].values
    ], ignore_index=True)
    val_ds = _make_timeseries_dataset(val_with_context, reference_dataset=train_ds)

    train_loader = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

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
        hidden_continuous_size=48,
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
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=12, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # uses GPU if available, falls back to CPU
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb, early_stop_cb, EpochLogger()],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(tft, train_loader, val_loader, ckpt_path=resume_ckpt)

    return tft


def load(checkpoint_path: str | None = None) -> TemporalFusionTransformer:
    if checkpoint_path is None:
        checkpoints = sorted(MODEL_TFT.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No TFT checkpoint found in {MODEL_TFT}")
        checkpoint_path = checkpoints[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # map_location remaps all tensors to the target device, but torchmetrics
    # caches its own device reference independently and fires cuda_init when
    # .to() is called on a CPU-only host with a GPU-saved checkpoint.
    # Fix: load to CPU, patch metric device refs, then move if needed.
    raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    raw["hyper_parameters"].pop("dataset_parameters", None)
    model = TemporalFusionTransformer(**raw["hyper_parameters"])
    model.load_state_dict(raw["state_dict"])
    # Reset torchmetrics device cache so .to(device) doesn't trigger cuda_init
    from torchmetrics import Metric
    for module in model.modules():
        if isinstance(module, Metric):
            module._device = torch.device("cpu")
    return model.to(device)


def predict(model: TemporalFusionTransformer, df: pd.DataFrame, train_df: pd.DataFrame) -> pd.Series:
    """
    Return median predicted forward return per row.
    df must have the same feature columns as train_df.
    """
    import logging as _logging
    for _name in ["lightning.pytorch.utilities.rank_zero", "lightning.pytorch.accelerators.cuda"]:
        _logging.getLogger(_name).setLevel(_logging.ERROR)

    train_ds = _make_timeseries_dataset(train_df)
    pred_ds = _make_timeseries_dataset(df, reference_dataset=train_ds, predict=True)
    loader = pred_ds.to_dataloader(train=False, batch_size=128, num_workers=0)

    result = model.predict(loader, mode="prediction", return_index=True)
    # pytorch-forecasting 1.7 returns a Prediction namedtuple:
    # [0] = predictions tensor (n_samples, n_quantiles)
    # [2] = index DataFrame
    raw_predictions = result[0]
    index = result[2]
    median_idx = raw_predictions.shape[1] // 2
    preds = raw_predictions[:, median_idx].numpy()

    return pd.Series(preds, index=index.index, name="tft_pred")
