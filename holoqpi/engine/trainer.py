"""Training loop for the joint reconstruction / segmentation / measurement task."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import Config
from ..losses import build_loss
from ..utils import (
    amp_dtype_from_name,
    count_parameters,
    format_metrics,
    get_logger,
    write_json,
)
from .evaluator import Evaluator, composite_score

LOGGER = get_logger(__name__)

_METRIC_DIRECTION = {
    "composite": "max",
    "seg_dice": "max",
    "seg_iou": "max",
    "seg_aji": "max",
    "seg_boundary_f1": "max",
    "phase_pearson_r": "max",
    "phase_ssim": "max",
    "phase_mae_rad": "min",
    "phase_rmse_rad": "min",
    "cls_accuracy": "max",
    "cls_macro_f1": "max",
    "dry_mass_mape": "min",
    "area_mape": "min",
}


def _build_grad_scaler(enabled: bool):
    """Construct a gradient scaler across PyTorch's two API generations.

    ``torch.amp.GradScaler(device, ...)`` arrived in 2.4; earlier 2.x releases
    only expose ``torch.cuda.amp.GradScaler``. Supporting both keeps the
    framework usable on a machine whose driver caps the installable version.
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: Config,
        loaders: dict[str, DataLoader],
        device: torch.device,
        run_dir: str | Path,
    ):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.val_loader = loaders.get("val")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        training = cfg.training
        self.epochs = training.epochs
        self.accumulation_steps = max(training.accumulation_steps, 1)
        self.grad_clip = training.grad_clip_norm
        self.log_every = training.log_every_n_steps

        self.criterion = build_loss(cfg).to(device)
        self.optimizer = self._build_optimizer(training)
        self.scheduler = self._build_scheduler(training)

        use_amp = training.mixed_precision and device.type == "cuda"
        self.amp_dtype = amp_dtype_from_name(training.amp_dtype)
        self.scaler = _build_grad_scaler(enabled=use_amp and self.amp_dtype == torch.float16)
        self.use_amp = use_amp

        self.evaluator = Evaluator(cfg, device) if self.val_loader is not None else None
        self.checkpoint_metric = training.checkpoint_metric
        self.direction = self._resolve_direction(training)
        self.best_value = -math.inf if self.direction == "max" else math.inf
        self.best_epoch = -1

        self.patience = training.early_stopping_patience
        self.epochs_without_improvement = 0
        self.history: list[dict] = []

        parameters = count_parameters(self.model)
        LOGGER.info(
            "trainable %.2fM / %.2fM parameters (%.1f%%)",
            parameters["trainable"] / 1e6, parameters["total"] / 1e6,
            100 * parameters["trainable_fraction"],
        )
        LOGGER.info("active physics terms: %s", self.criterion.active_physics_terms or "none")

    # -- setup ------------------------------------------------------------
    def _build_optimizer(self, training: Config) -> torch.optim.Optimizer:
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters:
            raise RuntimeError(
                "no trainable parameters. If LoRA is enabled, check that "
                "model.lora.always_trainable_patterns matches the decoder and head names."
            )
        name = training.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=training.learning_rate,
                weight_decay=training.weight_decay,
                betas=tuple(training.betas),
            )
        if name == "adam":
            return torch.optim.Adam(
                parameters, lr=training.learning_rate, betas=tuple(training.betas)
            )
        if name == "sgd":
            return torch.optim.SGD(
                parameters, lr=training.learning_rate,
                momentum=training.betas[0], weight_decay=training.weight_decay,
            )
        raise ValueError(f"unknown optimizer {training.optimizer!r}")

    def _build_scheduler(self, training: Config):
        name = training.scheduler
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(training.epochs - training.warmup_epochs, 1),
                eta_min=training.min_learning_rate,
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(training.epochs // 3, 1), gamma=0.1
            )
        raise ValueError(f"unknown scheduler {name!r}")

    def _resolve_direction(self, training: Config) -> str:
        if training.checkpoint_mode != "auto":
            return training.checkpoint_mode
        if self.checkpoint_metric not in _METRIC_DIRECTION:
            raise ValueError(
                f"cannot infer optimisation direction for checkpoint_metric "
                f"{self.checkpoint_metric!r}; set training.checkpoint_mode explicitly"
            )
        return _METRIC_DIRECTION[self.checkpoint_metric]

    # -- loop -------------------------------------------------------------
    def train(self) -> dict:
        warmup_epochs = self.cfg.training.warmup_epochs
        base_lr = self.cfg.training.learning_rate

        for epoch in range(1, self.epochs + 1):
            if epoch <= warmup_epochs and warmup_epochs > 0:
                for group in self.optimizer.param_groups:
                    group["lr"] = base_lr * epoch / warmup_epochs

            started = time.time()
            train_summary = self._train_one_epoch(epoch)
            train_seconds = time.time() - started

            record = {"epoch": epoch, "learning_rate": self.optimizer.param_groups[0]["lr"]}
            record.update({f"train_{k}": v for k, v in train_summary.items()})

            if self.evaluator is not None:
                validation_started = time.time()
                evaluation = self.evaluator.run(self.model, self.val_loader, loss_fn=self.criterion)
                validation_seconds = time.time() - validation_started
                metrics = evaluation["metrics"]
                record.update({f"val_{k}": v for k, v in metrics.items()})

                value = self._selection_value(metrics)
                record["val_selection"] = value
                if self._is_better(value):
                    self.best_value = value
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self._save_checkpoint("best_model.pt", epoch, metrics)
                    LOGGER.info("  new best %s=%.4f", self.checkpoint_metric, value)
                else:
                    self.epochs_without_improvement += 1

                # Report both halves: validation runs at full field size with the
                # per-cell measurement chain and routinely costs more than the
                # training epoch, so a single number would mislead any estimate
                # of how long the full schedule takes.
                total_seconds = train_seconds + validation_seconds
                record.update(
                    seconds=round(total_seconds, 1),
                    train_seconds=round(train_seconds, 1),
                    val_seconds=round(validation_seconds, 1),
                )
                LOGGER.info(
                    "epoch %d/%d (%.0fs = %.0fs train + %.0fs val) | %s",
                    epoch, self.epochs, total_seconds, train_seconds, validation_seconds,
                    format_metrics(
                        {
                            "train_loss": train_summary["total"],
                            "val_phase_mae": metrics.get("phase_mae_rad", float("nan")),
                            "val_dice": metrics.get("seg_dice", float("nan")),
                            "val_cls_acc": metrics.get("cls_accuracy", float("nan")),
                            "val_mass_mape": metrics.get("dry_mass_mape", float("nan")),
                        }
                    ),
                )
                if epoch == 1:
                    remaining = total_seconds * self.epochs / 60.0
                    LOGGER.info("  at this rate the full %d-epoch schedule takes about %.0f min",
                                self.epochs, remaining)
            else:
                record["seconds"] = round(train_seconds, 1)
                LOGGER.info("epoch %d/%d (%.0fs) | train_loss=%.4f",
                            epoch, self.epochs, train_seconds, train_summary["total"])

            self.history.append(record)
            write_json(self.history, self.run_dir / "history.json")

            if epoch > warmup_epochs and self.scheduler is not None:
                self.scheduler.step()

            if self.patience and self.epochs_without_improvement >= self.patience:
                LOGGER.info("early stopping after %d epochs without improvement", self.patience)
                break

        if self.cfg.training.save_last:
            self._save_checkpoint("last_model.pt", len(self.history), {})

        LOGGER.info("training finished; best epoch %d (%s=%.4f)",
                    self.best_epoch, self.checkpoint_metric, self.best_value)
        return {"best_epoch": self.best_epoch, "best_value": self.best_value,
                "history": self.history}

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals: dict[str, float] = {}
        steps = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            hologram = batch["hologram"].to(self.device, non_blocking=True)
            targets = {
                "phase": batch["phase"].to(self.device, non_blocking=True),
                "mask": batch["mask"].to(self.device, non_blocking=True),
                "condition": batch["condition"].to(self.device, non_blocking=True),
                # The forward-model term compares against the measurement itself,
                # so the input has to reach the loss as well as the network.
                "hologram": hologram,
            }

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                outputs = self.model(hologram)
                loss, components = self.criterion(outputs, targets)

            scaled = loss / self.accumulation_steps
            if self.scaler.is_enabled():
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (step + 1) % self.accumulation_steps == 0:
                if self.grad_clip:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            for key, value in components.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1

            if self.log_every and step % self.log_every == 0:
                LOGGER.info(
                    "  epoch %d step %d/%d loss=%.4f",
                    epoch, step, len(self.train_loader), components["total"],
                )

        return {key: value / max(steps, 1) for key, value in totals.items()}

    # -- checkpointing ----------------------------------------------------
    def _selection_value(self, metrics: dict) -> float:
        if self.checkpoint_metric == "composite":
            return composite_score(metrics, self.cfg.training.composite_metric)
        if self.checkpoint_metric not in metrics:
            raise KeyError(
                f"checkpoint_metric {self.checkpoint_metric!r} is not among the "
                f"computed metrics: {sorted(metrics)}"
            )
        return float(metrics[self.checkpoint_metric])

    def _is_better(self, value: float) -> bool:
        if not math.isfinite(value):
            return False
        return value > self.best_value if self.direction == "max" else value < self.best_value

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> Path:
        path = self.run_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.cfg.to_dict(),
                "checkpoint_metric": self.checkpoint_metric,
            },
            path,
        )
        return path


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device,
                    strict: bool = True) -> dict:
    """Restore weights, refusing to continue if the architecture disagrees.

    A silent partial load produces a partly random network that still runs and
    still emits plausible-looking masks, so the mismatch is raised instead.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    state = {key.replace("module.", "", 1): value for key, value in state.items()}

    result = model.load_state_dict(state, strict=False)
    expected = len(model.state_dict())
    matched = expected - len(result.missing_keys)

    LOGGER.info("checkpoint %s: %d/%d tensors matched", Path(path).name, matched, expected)
    if result.missing_keys:
        LOGGER.warning("  missing e.g. %s", result.missing_keys[:3])
    if result.unexpected_keys:
        LOGGER.warning("  unexpected e.g. %s", result.unexpected_keys[:3])

    if strict and matched / max(expected, 1) < 0.9:
        raise RuntimeError(
            f"only {100 * matched / expected:.1f}% of weights loaded from {path}. "
            "The model built from this config does not match the trained one; "
            "check model.encoder, model.frontend and model.lora before trusting any output."
        )
    return checkpoint
