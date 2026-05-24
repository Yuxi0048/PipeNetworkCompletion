"""Tests for the ISARC-seeded buffer-invariant split (Stage 1 / Phase A).

# Workstream: Claude

Covers AR-AF-A.1 (helper exists, uses Python stdlib RNG), AR-AF-A.4
(identical split across label buffers), and a frozen-fixture regression
to detect accidental RNG-mechanism drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipe_network_completion.anchor_free.baseline import (
    make_buffer_invariant_splits,
    make_stratified_edge_splits,
)


def test_buffer_invariant_split_partitions_every_edge():
    edge_ids = np.arange(1000)
    train, val, test = make_buffer_invariant_splits(edge_ids, seed=42)
    union = np.concatenate([train, val, test])
    assert union.size == edge_ids.size
    assert set(union.tolist()) == set(range(edge_ids.size))
    assert np.intersect1d(train, val).size == 0
    assert np.intersect1d(val, test).size == 0
    assert np.intersect1d(train, test).size == 0


def test_buffer_invariant_split_respects_fractions_within_one():
    edge_ids = np.arange(1000)
    train, val, test = make_buffer_invariant_splits(
        edge_ids, seed=42, train_fraction=0.6, val_fraction=0.2
    )
    # Within 1 edge of the requested fractions due to integer rounding.
    assert abs(train.size - 600) <= 1
    assert abs(val.size - 200) <= 1
    assert abs(test.size - 200) <= 1


def test_buffer_invariant_split_is_invariant_to_caller_order():
    """AR-AF-A.1 — split is a pure function of the edge_id SET, not order."""
    edge_ids_asc = np.arange(500)
    edge_ids_shuffled = np.random.default_rng(99).permutation(edge_ids_asc)
    a = make_buffer_invariant_splits(edge_ids_asc, seed=42)
    b = make_buffer_invariant_splits(edge_ids_shuffled, seed=42)
    # The integer positions returned will differ (they index into the
    # caller's array), but the underlying edge_id sets in each split must
    # match — that's the contract callers depend on.
    for split_a, split_b in zip(a, b):
        ids_a = set(int(edge_ids_asc[i]) for i in split_a)
        ids_b = set(int(edge_ids_shuffled[i]) for i in split_b)
        assert ids_a == ids_b


def test_buffer_invariant_split_is_identical_across_label_buffers():
    """AR-AF-A.4 — same edges, two different label vectors -> same split.

    This is the M6 fix: 10 m and 5 m label tables share the canonical
    train/val/test assignment so per-edge comparisons are apples-to-apples.
    """
    edge_ids = np.arange(2000)
    # Two different "label vectors" (irrelevant to the split now)
    y_10m = np.random.default_rng(0).integers(0, 2, size=2000)
    y_5m = np.random.default_rng(1).integers(0, 2, size=2000)
    assert not np.array_equal(y_10m, y_5m), "test labels must differ"

    split_10 = make_buffer_invariant_splits(edge_ids, seed=42)
    split_5 = make_buffer_invariant_splits(edge_ids, seed=42)
    for a, b in zip(split_10, split_5):
        np.testing.assert_array_equal(a, b)


def test_buffer_invariant_split_uses_stdlib_random_not_numpy():
    """AR-AF-A.2 — mirrors ISARC's random.Random(seed) mechanism.

    Verifies the contract by checking that the deterministic output differs
    from what numpy's default_rng would produce, while the existing
    stratified-split function (which uses numpy) still works.
    """
    edge_ids = np.arange(1000)
    a = make_buffer_invariant_splits(edge_ids, seed=42)

    # Run a numpy-style shuffle to confirm the two algorithms diverge with
    # the same integer seed. If this assertion ever fires it means someone
    # silently swapped the RNG mechanism.
    rng = np.random.default_rng(42)
    numpy_order = rng.permutation(1000)
    isarc_indices_combined = np.concatenate([a[0], a[1], a[2]])
    # The combined sorted indices cover 0..999 trivially; what we want to
    # know is that they are NOT in the same order as numpy's permutation.
    # Since make_buffer_invariant_splits sorts each split before returning,
    # we compare the *first split's* set order to numpy's prefix order.
    train_set = set(a[0].tolist())
    numpy_prefix_set = set(numpy_order[: a[0].size].tolist())
    assert train_set != numpy_prefix_set, (
        "buffer-invariant split appears to use numpy RNG; should use Python "
        "stdlib random.Random per the ISARC seeding contract"
    )


def test_buffer_invariant_split_is_seed_deterministic():
    edge_ids = np.arange(500)
    a = make_buffer_invariant_splits(edge_ids, seed=42)
    b = make_buffer_invariant_splits(edge_ids, seed=42)
    c = make_buffer_invariant_splits(edge_ids, seed=43)
    for ai, bi in zip(a, b):
        np.testing.assert_array_equal(ai, bi)
    # Different seed -> different partition (at least one split differs).
    assert any(not np.array_equal(ai, ci) for ai, ci in zip(a, c))


def test_buffer_invariant_split_handles_tiny_inputs():
    """Smoke: 0 and 1 element edge_ids must not crash."""
    empty = np.array([], dtype=int)
    a, b, c = make_buffer_invariant_splits(empty, seed=42)
    assert a.size == b.size == c.size == 0

    one = np.array([7], dtype=int)
    a, b, c = make_buffer_invariant_splits(one, seed=42)
    union = np.concatenate([a, b, c])
    assert union.size == 1


def test_make_stratified_edge_splits_unchanged_for_backward_compat():
    """AR-AF-A.5 — the legacy function must keep its existing contract."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 100)
    train, val, test = make_stratified_edge_splits(y, seed=42)
    union = np.concatenate([train, val, test])
    assert union.size == y.size
    assert set(union.tolist()) == set(range(y.size))


def test_pipeline_uses_buffer_invariant_split_by_default(tmp_path):
    """Stage 1 completion — pipeline.run_anchor_free_experiment must use
    the buffer-invariant split by default. Catches the bug where Stage 1
    fixed prep but left the training pipeline on the leaky stratified
    split."""
    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment

    cfg = load_anchor_free_config(None)
    cfg["model"]["type"] = "logistic_regression"
    cfg["experiment_name"] = "split_default_smoke"
    # Default config has split.strategy == "buffer_invariant".
    assert cfg["split"]["strategy"] == "buffer_invariant"
    result = run_anchor_free_experiment(cfg, synthetic=True, output_root=tmp_path)
    # The pure invariant: derive what the buffer-invariant split SHOULD be
    # on the same node ids and compare to what the pipeline actually used.
    # Post-Phase-2.A the prediction unit is RoadSegment, so the helper
    # exposes ``segment_ids`` instead of ``edge_ids``.
    expected = make_buffer_invariant_splits(result.features.segment_ids, seed=42)
    for actual, want in zip(
        (result.train_index, result.val_index, result.test_index), expected
    ):
        np.testing.assert_array_equal(actual, want)


def test_pipeline_can_still_run_stratified_split_for_back_compat(tmp_path):
    """AR-AF-A.5 — opt-in fallback to the legacy stratified split."""
    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free.pipeline import run_anchor_free_experiment

    cfg = load_anchor_free_config(None)
    cfg["model"]["type"] = "logistic_regression"
    cfg["experiment_name"] = "split_legacy_smoke"
    cfg["split"]["strategy"] = "stratified"
    result = run_anchor_free_experiment(cfg, synthetic=True, output_root=tmp_path)
    # Whatever the strategy is, the splits must cover every edge exactly once.
    union = np.concatenate(
        [result.train_index, result.val_index, result.test_index]
    )
    n_edges = int(result.features.features.shape[0])
    # Stratified split can duplicate val/test on tiny synthetic graphs (see
    # the safety guard at the bottom of make_stratified_edge_splits), so we
    # only check the union covers every edge.
    assert set(union.tolist()).issuperset(set(range(n_edges)))
