"""Configuration helpers for anchor-free experiments."""

# Workstream: Codex + Claude merge

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "anchor_free_isarc2024",
    "mode": "anchor_free",
    "task_type": "sewer",
    "seed": 42,
    "data": {
        "roads_path": "data/raw/gis/roads/Roads_ExportFeatures.shp",
        "buildings_path": "",
        "building_points_path": "",
        "built_up_path": "",
        "watercourse_drainage_lines_path": "",
        "watercourse_corridor_centrelines_path": "",
        "watercourse_corridors_path": "",
        "utility_truth_path": "data/raw/gis/sewer/SewerGravityMa_ExportFeature2.shp",
        "dem_path": "",
        "sources_sinks_path": "",
    },
    "graph": {
        "target_crs": "EPSG:3857",
        "snap_tolerance_m": 1.0,
        "road_class_columns": ["road_class", "OVL2_CAT", "CLASS", "highway"],
        "building_buffer_m": 50.0,
        "building_point_buffer_m": 50.0,
        "built_up_buffer_m": 50.0,
        "watercourse_buffer_m": 100.0,
        "road_density_buffer_m": 100.0,
        "dem_sample_spacing_m": 30.0,
        "dem_max_samples_per_edge": 64,
        "label_buffer_m": 10.0,
        "label_overlap_threshold": 0.25,
        "use_buildings": True,
        "use_building_points": True,
        "use_built_up": True,
        "use_dem": True,
        "use_watercourses": False,
        "watercourse_context_complete": False,
    },
    # Stage 1 of audit_followup_implementation_plan.md: the training
    # pipeline now defaults to the ISARC-seeded buffer-invariant split.
    # Override to "stratified" to reproduce pre-Stage-1 metric history.
    "split": {
        "strategy": "buffer_invariant",
        "train_fraction": 0.6,
        "val_fraction": 0.2,
    },
    "model": {
        "type": "gnn",
        "hidden_dim": 64,
        "num_layers": 3,
        "dropout": 0.1,
        "lr": 0.001,
        "epochs": 100,
        "class_weight": "balanced",
        "device": "auto",
        # Stage 2 ablation: set to False to test whether absolute road-node
        # (x, y) coordinates are acting as a location-memorisation shortcut.
        "include_node_coords": True,
    },
    "decoder": {
        "type": "threshold",
        "threshold": 0.5,
        "lambda_length": 0.001,
        "lambda_engineering": 1.0,
        "loop_budget_fraction": 0.05,
    },
    "evaluation": {
        "threshold_grid": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "building_service_buffer_m": 50.0,
        "extra_label_buffers_m": [5.0],
    },
}


def deep_update(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value == "":
        return {}
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        key, sep, raw_value = content.partition(":")
        if not sep:
            raise ValueError(f"Unsupported config line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_scalar(raw_value)
        parent[key.strip()] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root


def _empty_maps_to_none(value: Any) -> Any:
    if isinstance(value, dict):
        if not value:
            return None
        return {key: _empty_maps_to_none(child) for key, child in value.items()}
    return value


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return dict(loaded or {})
    except ModuleNotFoundError:
        return _empty_maps_to_none(_load_simple_yaml(text))


def load_anchor_free_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return deepcopy(DEFAULT_CONFIG)
    loaded = load_yaml_mapping(Path(path))
    return deep_update(DEFAULT_CONFIG, loaded)


def write_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
