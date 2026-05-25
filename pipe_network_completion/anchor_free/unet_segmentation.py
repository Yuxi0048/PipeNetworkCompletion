"""Utilities for raster U-Net sewer-corridor segmentation.

Workstream: Codex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Dataset


def make_unet_model(
    *,
    in_channels: int,
    encoder_name: str = "resnet18",
    encoder_weights: str | None = None,
    classes: int = 1,
) -> torch.nn.Module:
    """Create a standard U-Net from segmentation-models-pytorch."""

    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "segmentation-models-pytorch is required. Install it in the CUDA "
            "environment without changing the pinned torch/PyG stack."
        ) from exc
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=int(in_channels),
        classes=int(classes),
        activation=None,
    )


class NpzSegmentationDataset(Dataset):
    """Dataset for `.npz` tiles written by `prepare_unet_tiles.py`."""

    def __init__(
        self,
        index: pd.DataFrame,
        *,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        self.index = index.reset_index(drop=True).copy()
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = Path(self.index.loc[idx, "tile_path"])
        with np.load(path, allow_pickle=True) as data:
            x = data["x"].astype("float32")
            y = data["y"].astype("float32")[None, :, :]
        if self.mean is not None and self.std is not None:
            x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return torch.from_numpy(x), torch.from_numpy(y)


def load_tile_index(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    base = path.parent.resolve()
    table["tile_path"] = table["tile_path"].map(lambda value: str((base / str(value)).resolve()))
    return table


def split_index(index: pd.DataFrame, split: str) -> pd.DataFrame:
    return index[index["split"].astype(str) == split].reset_index(drop=True)


def compute_channel_stats(index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sums: np.ndarray | None = None
    sq_sums: np.ndarray | None = None
    count = 0
    for path_value in index["tile_path"]:
        with np.load(path_value, allow_pickle=True) as data:
            x = data["x"].astype("float64")
        if sums is None:
            sums = np.zeros(x.shape[0], dtype="float64")
            sq_sums = np.zeros(x.shape[0], dtype="float64")
        sums += x.reshape(x.shape[0], -1).sum(axis=1)
        sq_sums += np.square(x.reshape(x.shape[0], -1)).sum(axis=1)
        count += x.shape[1] * x.shape[2]
    if sums is None or sq_sums is None or count == 0:
        raise ValueError("Cannot compute channel stats from an empty training index.")
    mean = sums / count
    var = np.maximum((sq_sums / count) - np.square(mean), 1e-8)
    return mean.astype("float32"), np.sqrt(var).astype("float32")


def positive_class_weight(index: pd.DataFrame, *, max_weight: float = 100.0) -> float:
    positives = 0.0
    total = 0.0
    for path_value in index["tile_path"]:
        with np.load(path_value, allow_pickle=True) as data:
            y = data["y"].astype("float32")
        positives += float(y.sum())
        total += float(y.size)
    negatives = max(total - positives, 0.0)
    if positives <= 0.0:
        return 1.0
    return float(min(negatives / positives, max_weight))


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    intersection = torch.sum(prob * target)
    denominator = torch.sum(prob) + torch.sum(target)
    return 1.0 - (2.0 * intersection + eps) / (denominator + eps)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, *, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce + dice_loss_from_logits(logits, target)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).detach().cpu()
        y_prob.append(torch.sigmoid(logits).numpy().reshape(-1))
        y_true.append(y.numpy().reshape(-1))
    return np.concatenate(y_true), np.concatenate(y_prob)


def binary_pixel_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    y_true_int = (y_true >= 0.5).astype("int32")
    y_pred = (y_prob >= float(threshold)).astype("int32")
    tp = float(((y_true_int == 1) & (y_pred == 1)).sum())
    fp = float(((y_true_int == 0) & (y_pred == 1)).sum())
    fn = float(((y_true_int == 1) & (y_pred == 0)).sum())
    union = tp + fp + fn
    metrics = {
        "threshold": float(threshold),
        "positive_fraction": float(y_true_int.mean()) if y_true_int.size else 0.0,
        "precision": float(precision_score(y_true_int, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_int, y_pred, zero_division=0)),
        "iou": tp / union if union > 0.0 else 0.0,
    }
    if np.unique(y_true_int).size == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_int, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true_int, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def save_stats(path: Path, *, mean: np.ndarray, std: np.ndarray, channel_names: list[str]) -> None:
    payload = {
        "channel_names": list(channel_names),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
