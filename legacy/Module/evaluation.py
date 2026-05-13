from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


@dataclass(frozen=True)
class BinaryMetrics:
    auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    mcc: float


def compute_binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> BinaryMetrics:
    predictions = (scores > threshold).astype(int)
    auc = float("nan")
    if np.unique(labels).size > 1:
        auc = float(roc_auc_score(labels, scores))
    return BinaryMetrics(
        auc=auc,
        f1=float(f1_score(labels, predictions, zero_division=0)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        accuracy=float(accuracy_score(labels, predictions)),
        mcc=float(matthews_corrcoef(labels, predictions)),
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: str | torch.device,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> BinaryMetrics:
    model.eval()
    preds = []
    ground_truths = []

    for batch_index, sampled_data in enumerate(tqdm(loader, desc="Evaluating")):
        if max_batches is not None and batch_index >= max_batches:
            break
        with torch.no_grad():
            sampled_data = sampled_data.to(device)
            preds.append(model(sampled_data).detach().cpu())
            ground_truths.append(
                sampled_data["MH", "link", "MH"].edge_label.detach().cpu()
            )

    if not preds:
        raise ValueError("No batches were evaluated.")

    scores = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(ground_truths, dim=0).numpy()
    return compute_binary_metrics(scores, labels, threshold=threshold)
