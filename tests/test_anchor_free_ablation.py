"""Tests for anchor-free ablation configuration hygiene."""

# Workstream: Codex

from __future__ import annotations

from scripts.run_anchor_free_ablation import _context_flags, _variant_config


def test_road_only_ablation_disables_all_context_layers():
    base = {
        "experiment_name": "anchor_free_test",
        "model": {"type": "gnn"},
        "graph": {
            "use_buildings": True,
            "use_building_points": True,
            "use_built_up": True,
            "use_dem": True,
        },
        "decoder": {"type": "threshold"},
    }
    config = _variant_config(
        base,
        variant_name="Anchor-free road-only Random Forest",
        model_type="random_forest",
        context_flags=_context_flags(
            use_buildings=False,
            use_building_points=False,
            use_built_up=False,
            use_dem=False,
        ),
        decoder_type="threshold",
    )
    assert config["model"]["type"] == "random_forest"
    assert config["decoder"]["type"] == "threshold"
    assert config["graph"]["use_buildings"] is False
    assert config["graph"]["use_building_points"] is False
    assert config["graph"]["use_built_up"] is False
    assert config["graph"]["use_dem"] is False
