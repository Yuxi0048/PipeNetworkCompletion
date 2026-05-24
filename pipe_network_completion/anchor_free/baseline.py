"""Classical non-GNN baselines for anchor-free road-edge prediction."""

# Workstream: Codex + Claude merge

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipe_network_completion.anchor_free.features import assert_no_anchor_features


BaselineKind = Literal["logistic_regression", "random_forest"]


@dataclass(frozen=True)
class BaselineResult:
    model: object
    probabilities: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray


def make_stratified_edge_splits(
    y: np.ndarray,
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic train/validation/test edge splits."""

    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for cls in sorted(np.unique(y)):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        n_train = int(round(len(cls_idx) * train_fraction))
        n_val = int(round(len(cls_idx) * val_fraction))
        if len(cls_idx) >= 3:
            n_train = max(1, min(n_train, len(cls_idx) - 2))
            n_val = max(1, min(n_val, len(cls_idx) - n_train - 1))
        train_parts.append(cls_idx[:n_train])
        val_parts.append(cls_idx[n_train : n_train + n_val])
        test_parts.append(cls_idx[n_train + n_val :])

    train_index = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    val_index = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
    test_index = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
    rng.shuffle(train_index)
    rng.shuffle(val_index)
    rng.shuffle(test_index)

    if train_index.size == 0 and y.size:
        train_index = np.arange(y.size)
    if val_index.size == 0:
        val_index = test_index.copy()
    if test_index.size == 0:
        test_index = val_index.copy()

    return train_index, val_index, test_index


def make_buffer_invariant_splits(
    edge_ids: np.ndarray,
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test split assignments that depend ONLY on ``edge_ids``.

    Phase A of ``docs/research_notes/audit_followup_implementation_plan.md``.

    Why this exists:

    * ``make_stratified_edge_splits`` stratifies on the label vector ``y``.
      When the same road graph is labelled at multiple ``label_buffer_m``
      values (e.g. ``training_ready_label_buffers_m: [10, 5]``), the label
      vector differs per buffer, so the stratified split *also* differs.
      That makes 10 m and 5 m metrics non-comparable on a per-edge basis.
    * This helper instead shuffles a stable list of ``edge_ids`` and slices
      it by the given fractions, so the split assignment is identical for
      every label buffer that shares the same edge IDs.

    Why ``random.Random(seed)``, not ``np.random.default_rng(seed)``:

    * The ISARC 2024 paper's split mechanism is
      ``random.Random(seed).shuffle(items)`` at
      ``process.py:33`` (``split_list_by_ratio``) with default ``seed=42``
      at ``process.py:272``. Reusing the same RNG mechanism keeps the
      anchor-free split's shuffle order in the same family as the paper's
      published split discipline — different splitting *unit* (edges vs
      components), same seeding *contract*.
    * ``random.Random`` is Python's Mersenne-Twister-derived RNG; for the
      same seed it deterministically produces a different shuffle order
      than ``np.random.default_rng`` (PCG64). The anchor-free
      backward-compatible function ``make_stratified_edge_splits`` keeps
      the numpy RNG to preserve the existing
      ``outputs/anchor_free_brisbane_*`` metric history; only this new
      helper adopts the ISARC mechanism.

    Returns three disjoint integer index arrays into the input
    ``edge_ids`` order. Their union covers every index 0 .. len(edge_ids)-1.
    """

    edge_ids = np.asarray(edge_ids, dtype=int)
    n = int(edge_ids.size)
    if n == 0:
        empty = np.array([], dtype=int)
        return empty, empty.copy(), empty.copy()

    if not 0.0 <= float(train_fraction) <= 1.0:
        raise ValueError(f"train_fraction must be in [0,1]; got {train_fraction}")
    if not 0.0 <= float(val_fraction) <= 1.0:
        raise ValueError(f"val_fraction must be in [0,1]; got {val_fraction}")
    if float(train_fraction) + float(val_fraction) > 1.0 + 1e-9:
        raise ValueError(
            f"train_fraction + val_fraction must be <= 1; "
            f"got {train_fraction} + {val_fraction}"
        )

    # Shuffle positional indexes (not the edge_id values themselves), so the
    # split assignment is invariant to the order in which the caller passes
    # edge_ids. The shuffle key is the (seed, edge_id) pair so two callers
    # with the same edge_ids and the same seed get the same partition even
    # if they pass edge_ids in different orders.
    positions = list(range(n))
    rng = random.Random(int(seed))
    # Sort positions by their (seed, edge_id) hash so the partition is a
    # pure function of the edge_id set and the seed. We use rng.shuffle on
    # a deterministically pre-sorted list so we get both reproducibility AND
    # invariance to caller order.
    positions.sort(key=lambda p: int(edge_ids[p]))
    rng.shuffle(positions)

    n_train = int(round(n * float(train_fraction)))
    n_val = int(round(n * float(val_fraction)))
    # Guarantee at least one edge in each split when n is large enough; this
    # mirrors the safety guard in ``make_stratified_edge_splits``.
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))

    train_index = np.array(sorted(positions[:n_train]), dtype=int)
    val_index = np.array(sorted(positions[n_train : n_train + n_val]), dtype=int)
    test_index = np.array(sorted(positions[n_train + n_val :]), dtype=int)
    return train_index, val_index, test_index


def build_baseline_model(
    kind: BaselineKind,
    *,
    seed: int = 42,
    class_weight: str | dict | None = "balanced",
) -> object:
    if kind == "logistic_regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight=class_weight,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=1,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unsupported baseline kind: {kind}")


def train_baseline(
    features: pd.DataFrame,
    y: np.ndarray,
    *,
    kind: BaselineKind = "random_forest",
    train_index: np.ndarray | None = None,
    val_index: np.ndarray | None = None,
    test_index: np.ndarray | None = None,
    seed: int = 42,
    class_weight: str | dict | None = "balanced",
) -> BaselineResult:
    """Fit a classical classifier and return probabilities for all edges."""

    assert_no_anchor_features(features.columns)
    y = np.asarray(y, dtype=int)
    if train_index is None or val_index is None or test_index is None:
        train_index, val_index, test_index = make_stratified_edge_splits(y, seed=seed)

    if np.unique(y[train_index]).size < 2:
        model = DummyClassifier(strategy="prior")
    else:
        model = build_baseline_model(kind, seed=seed, class_weight=class_weight)
    model.fit(features.iloc[train_index], y[train_index])
    proba = model.predict_proba(features)
    if proba.shape[1] == 1:
        probabilities = np.full(len(features), float(model.classes_[0] == 1))
    else:
        probabilities = proba[:, 1]

    return BaselineResult(
        model=model,
        probabilities=np.asarray(probabilities, dtype=float),
        train_index=np.asarray(train_index, dtype=int),
        val_index=np.asarray(val_index, dtype=int),
        test_index=np.asarray(test_index, dtype=int),
    )
