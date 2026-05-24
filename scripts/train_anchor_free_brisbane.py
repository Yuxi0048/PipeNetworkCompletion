"""Train the anchor-free road-edge GNN on real Brisbane data.

# Workstream: Claude

Drives ``run_anchor_free_experiment`` against the full Brisbane road network
(~41k segments, ~244k truth lines) with optional building / built-up / DEM
context. The patch applied is a sjoin-based replacement for the legacy edge
labeler and the active RoadSegment labeler; Codex's sindex-based density loop
is fast enough to use as-is.

Examples:
    # road-only (no buildings, no built-up, no DEM)
    python scripts/train_anchor_free_brisbane.py --no-buildings --no-built-up --no-dem

    # full feature set
    python scripts/train_anchor_free_brisbane.py

    # ablation: features but no DEM
    python scripts/train_anchor_free_brisbane.py --no-dem
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------------
# Fast sjoin-based replacements for the legacy edge labeler and the active
# RoadSegment labeler. These use geopandas sjoin to restrict per-candidate work
# to truth lines that actually intersect each candidate buffer.
# -----------------------------------------------------------------------------
def _fast_label_road_edges_from_utility_lines(
    graph, utility_truth_gdf, *, label_buffer_m=10.0, label_overlap_threshold=0.25
):
    from pipe_network_completion.anchor_free.labels import RoadEdgeLabels

    edges = graph.edges.copy()
    if edges.empty:
        empty = gpd.GeoDataFrame(
            {"edge_id": [], "y": [], "overlap_length": [], "overlap_ratio": []},
            geometry=[],
            crs=graph.crs,
        )
        return RoadEdgeLabels(empty)

    truth = utility_truth_gdf
    if graph.crs and truth.crs and str(truth.crs) != str(graph.crs):
        truth = truth.to_crs(graph.crs)

    buffered = gpd.GeoDataFrame(
        {"edge_id": edges["edge_id"].astype(int).values},
        geometry=edges.geometry.buffer(float(label_buffer_m)).values,
        crs=graph.crs,
    )
    truth_min = gpd.GeoDataFrame(
        {"_ti": np.arange(len(truth))},
        geometry=truth.geometry.values,
        crs=graph.crs,
    )
    joined = gpd.sjoin(truth_min, buffered, predicate="intersects", how="inner")

    edge_to_buf = dict(zip(buffered["edge_id"].values, buffered.geometry.values))
    truth_geoms = truth.geometry.values
    overlap = {int(eid): 0.0 for eid in edges["edge_id"].astype(int).values}
    for ti, eid in zip(joined["_ti"].values, joined["edge_id"].values):
        eid = int(eid)
        inter = truth_geoms[int(ti)].intersection(edge_to_buf[eid])
        if not inter.is_empty:
            overlap[eid] = overlap[eid] + float(inter.length)

    lengths = edges["length_m"].astype(float).values
    edge_ids = edges["edge_id"].astype(int).values
    overlap_lengths = np.array([overlap[int(e)] for e in edge_ids])
    overlap_ratios = np.minimum(overlap_lengths / np.maximum(lengths, 1e-12), 1.0)
    y = (overlap_ratios >= float(label_overlap_threshold)).astype(int)

    records = []
    for i, edge in enumerate(edges.itertuples(index=False)):
        records.append(
            {
                "edge_id": int(edge.edge_id),
                "y": int(y[i]),
                "overlap_length": float(overlap_lengths[i]),
                "overlap_ratio": float(overlap_ratios[i]),
                "geometry": edge.geometry,
            }
        )
    return RoadEdgeLabels(
        gpd.GeoDataFrame(records, geometry="geometry", crs=edges.crs)
    )


def _fast_label_road_segments_from_utility_lines(
    graph, utility_truth_gdf, *, label_buffer_m=10.0, label_overlap_threshold=0.25
):
    from pipe_network_completion.anchor_free.labels import RoadSegmentLabels

    segments = graph.road_segments.copy()
    if segments.empty:
        empty = gpd.GeoDataFrame(
            {"segment_id": [], "y": [], "overlap_length": [], "overlap_ratio": []},
            geometry=[],
            crs=getattr(graph, "crs", None),
        )
        return RoadSegmentLabels(empty)

    truth = utility_truth_gdf
    if graph.crs and truth.crs and str(truth.crs) != str(graph.crs):
        truth = truth.to_crs(graph.crs)

    buffered = gpd.GeoDataFrame(
        {"segment_id": segments["segment_id"].astype(int).values},
        geometry=segments.geometry.buffer(float(label_buffer_m)).values,
        crs=graph.crs,
    )
    truth_min = gpd.GeoDataFrame(
        {"_ti": np.arange(len(truth))},
        geometry=truth.geometry.values,
        crs=graph.crs,
    )
    joined = gpd.sjoin(truth_min, buffered, predicate="intersects", how="inner")

    segment_to_buf = dict(zip(buffered["segment_id"].values, buffered.geometry.values))
    truth_geoms = truth.geometry.values
    overlap = {int(sid): 0.0 for sid in segments["segment_id"].astype(int).values}
    for ti, sid in zip(joined["_ti"].values, joined["segment_id"].values):
        sid = int(sid)
        inter = truth_geoms[int(ti)].intersection(segment_to_buf[sid])
        if not inter.is_empty:
            overlap[sid] = overlap[sid] + float(inter.length)

    lengths = segments["length_m"].astype(float).values
    segment_ids = segments["segment_id"].astype(int).values
    overlap_lengths = np.array([overlap[int(sid)] for sid in segment_ids])
    overlap_ratios = np.minimum(overlap_lengths / np.maximum(lengths, 1e-12), 1.0)
    y = (overlap_ratios >= float(label_overlap_threshold)).astype(int)

    records = []
    for i, segment in enumerate(segments.itertuples(index=False)):
        records.append(
            {
                "segment_id": int(segment.segment_id),
                "y": int(y[i]),
                "overlap_length": float(overlap_lengths[i]),
                "overlap_ratio": float(overlap_ratios[i]),
                "geometry": segment.geometry,
            }
        )
    return RoadSegmentLabels(
        gpd.GeoDataFrame(records, geometry="geometry", crs=segments.crs)
    )


def _validating_compatible_gdf(gdf, graph):
    """Repair invalid polygon geometries before Codex's intersection loops.

    Real GIS data (especially OS / OpenStreetMap-derived polygons) routinely
    contains self-intersections, repeated vertices, and other GEOS-invalid
    topologies. Codex's ``_polygon_context_stats`` and ``_building_context_stats``
    do per-edge polygon.intersection(buffer), which raises GEOSException on
    invalid inputs. We repair once at load time using ``make_valid`` so the
    downstream loops can stay simple.
    """
    from shapely.validation import make_valid

    if gdf is None or gdf.empty:
        return None
    if graph.crs and gdf.crs and str(gdf.crs) != str(graph.crs):
        gdf = gdf.to_crs(graph.crs)
    invalid_mask = ~gdf.geometry.is_valid
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        gdf = gdf.copy()
        # make_valid keeps the polygon's topology when possible and returns a
        # GeometryCollection only for the most pathological cases; the
        # downstream code already filters on geom_type.
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(
            make_valid
        )
        print(f"    [patch] repaired {n_invalid} invalid polygon geometries")
    return gdf


def apply_fast_patches() -> None:
    from pipe_network_completion.anchor_free import features as af_features
    from pipe_network_completion.anchor_free import labels as af_labels
    from pipe_network_completion.anchor_free import pipeline as af_pipeline

    af_features._compatible_gdf = _validating_compatible_gdf
    af_labels.label_road_edges_from_utility_lines = (
        _fast_label_road_edges_from_utility_lines
    )
    af_pipeline.label_road_edges_from_utility_lines = (
        _fast_label_road_edges_from_utility_lines
    )
    af_labels.label_road_segments_from_utility_lines = (
        _fast_label_road_segments_from_utility_lines
    )
    af_pipeline.label_road_segments_from_utility_lines = (
        _fast_label_road_segments_from_utility_lines
    )


# -----------------------------------------------------------------------------
# Real-data paths.
# -----------------------------------------------------------------------------
DEFAULT_ROADS = "data/raw/gis/roads/Roads_ExportFeatures.shp"
DEFAULT_TRUTH = [
    "data/raw/gis/sewer/SewerGravityMa_ExportFeature1.shp",
    "data/raw/gis/sewer/SewerGravityMa_ExportFeature2.shp",
]
DEFAULT_BUILDINGS = "data/processed/context/study_area/building_areas_study_area.geojson"
# Codex's review (P2 of current_codebase_review_codex.md) flagged that this
# script previously omitted building POINTS even though they are a
# meaningfully different signal from building POLYGONS. Both are wired now.
DEFAULT_BUILDING_POINTS = "data/processed/context/study_area/building_points_study_area.geojson"
DEFAULT_BUILT_UP = "data/processed/context/study_area/build_up_areas_study_area.geojson"
DEFAULT_DEM = "data/processed/context/study_area/brisbane_dem_h_1sec_epsg28356.tif"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train anchor-free GNN on Brisbane real data."
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0"])
    p.add_argument("--target-crs", default="EPSG:28356")
    p.add_argument("--label-buffer-m", type=float, default=10.0)
    p.add_argument("--label-overlap-threshold", type=float, default=0.25)
    p.add_argument("--density-buffer-m", type=float, default=100.0)
    p.add_argument("--building-buffer-m", type=float, default=50.0)
    p.add_argument("--built-up-buffer-m", type=float, default=50.0)
    p.add_argument("--building-point-buffer-m", type=float, default=50.0)
    p.add_argument("--dem-spacing-m", type=float, default=30.0)
    p.add_argument("--decoder", choices=["threshold", "connected"], default="threshold")

    feats = p.add_argument_group("feature toggles")
    feats.add_argument("--no-buildings", action="store_true")
    feats.add_argument("--no-building-points", action="store_true")  # Codex P2
    feats.add_argument("--no-built-up", action="store_true")
    feats.add_argument("--no-dem", action="store_true")

    paths = p.add_argument_group("data paths (override real defaults if needed)")
    paths.add_argument("--roads", default=DEFAULT_ROADS)
    paths.add_argument("--truth", default=DEFAULT_TRUTH)
    paths.add_argument("--buildings", default=DEFAULT_BUILDINGS)
    paths.add_argument("--building-points", default=DEFAULT_BUILDING_POINTS)  # Codex P2
    paths.add_argument("--built-up", default=DEFAULT_BUILT_UP)
    paths.add_argument("--dem", default=DEFAULT_DEM)

    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Defaults to 'anchor_free_brisbane_<feature_combo>'.",
    )
    return p.parse_args()


def _print_metrics_vs_baseline(metrics: dict, *, split_prefix: str = "test_") -> None:
    """Print model vs all-positive baseline side-by-side with Δ column.

    Phase C of docs/research_notes/audit_followup_implementation_plan.md.
    Stops a small lift over the prevalence ceiling from being misread as a
    strong result.
    """
    prev = metrics.get(f"{split_prefix}positive_prevalence")
    if prev is not None:
        try:
            print(f"  positive_prevalence : {float(prev):.4f}")
        except (TypeError, ValueError):
            pass
    pairs = [
        ("roc_auc", "all_positive_roc_auc"),
        ("pr_auc", "all_positive_pr_auc"),
        ("f1", "all_positive_f1"),
        ("precision", "all_positive_precision"),
        ("recall", "all_positive_recall"),
        ("brier_score", "all_positive_brier_score"),
        ("balanced_accuracy", "all_positive_balanced_accuracy"),
    ]
    header = f"  {'metric':>20s} {'model':>10s} {'all_pos':>10s} {'Δ':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for model_key, baseline_key in pairs:
        m = metrics.get(f"{split_prefix}{model_key}")
        b = metrics.get(f"{split_prefix}{baseline_key}")
        if m is None or b is None:
            continue
        try:
            mv, bv = float(m), float(b)
            delta = mv - bv
            print(
                f"  {model_key:>20s} {mv:>10.4f} {bv:>10.4f} {delta:>+10.4f}"
            )
        except (TypeError, ValueError):
            continue


def _derive_experiment_name(args) -> str:
    if args.experiment_name:
        return args.experiment_name
    parts = ["road"]
    if not args.no_buildings:
        parts.append("buildings")
    if not args.no_building_points:
        parts.append("bpoints")
    if not args.no_built_up:
        parts.append("builtup")
    if not args.no_dem:
        parts.append("dem")
    return "anchor_free_brisbane_" + "_".join(parts)


def main() -> int:
    args = parse_args()
    apply_fast_patches()

    from pipe_network_completion.anchor_free.config import load_anchor_free_config
    from pipe_network_completion.anchor_free.pipeline import (
        run_anchor_free_experiment,
    )

    cfg = load_anchor_free_config(REPO_ROOT / "configs" / "anchor_free_isarc2024.yaml")
    cfg["experiment_name"] = _derive_experiment_name(args)

    cfg["data"]["roads_path"] = args.roads
    cfg["data"]["utility_truth_path"] = args.truth
    cfg["data"]["buildings_path"] = "" if args.no_buildings else args.buildings
    cfg["data"]["building_points_path"] = (
        "" if args.no_building_points else args.building_points
    )
    cfg["data"]["built_up_path"] = "" if args.no_built_up else args.built_up
    cfg["data"]["dem_path"] = "" if args.no_dem else args.dem

    cfg["graph"]["target_crs"] = args.target_crs
    cfg["graph"]["road_class_columns"] = ["OVL2_CAT", "OVL_CAT"]
    cfg["graph"]["label_buffer_m"] = float(args.label_buffer_m)
    cfg["graph"]["label_overlap_threshold"] = float(args.label_overlap_threshold)
    cfg["graph"]["road_density_buffer_m"] = float(args.density_buffer_m)
    cfg["graph"]["building_buffer_m"] = float(args.building_buffer_m)
    cfg["graph"]["building_point_buffer_m"] = float(args.building_point_buffer_m)
    cfg["graph"]["built_up_buffer_m"] = float(args.built_up_buffer_m)
    cfg["graph"]["dem_sample_spacing_m"] = float(args.dem_spacing_m)
    cfg["graph"]["use_buildings"] = not args.no_buildings
    cfg["graph"]["use_building_points"] = not args.no_building_points
    cfg["graph"]["use_built_up"] = not args.no_built_up
    cfg["graph"]["use_dem"] = not args.no_dem

    cfg["model"]["type"] = "gnn"
    cfg["model"]["device"] = args.device
    cfg["model"]["epochs"] = int(args.epochs)
    cfg["model"]["hidden_dim"] = int(args.hidden)
    cfg["model"]["num_layers"] = int(args.num_layers)

    cfg["decoder"]["type"] = args.decoder

    enabled = [name for name, on in (
        ("buildings",       not args.no_buildings),
        ("building_points", not args.no_building_points),
        ("built_up",        not args.no_built_up),
        ("dem",             not args.no_dem),
    ) if on]
    print(f"==> Brisbane anchor-free training [{cfg['experiment_name']}]")
    print(f"    epochs={args.epochs}, hidden={args.hidden}, layers={args.num_layers}")
    print(f"    device={args.device}, decoder={args.decoder}, crs={args.target_crs}")
    print(f"    extra features: {', '.join(enabled) if enabled else '(road only)'}")
    print()

    t0 = time.time()
    result = run_anchor_free_experiment(
        cfg, synthetic=False, output_root=args.output_root
    )
    runtime = time.time() - t0

    print()
    print(f"==> done in {runtime:.1f}s")
    print(f"    output dir : {result.output_dir}")
    print(f"    n_segments = {len(result.graph.road_segments)}")
    print(f"    n_features = {result.features.features.shape[1]}")
    print(f"    pos_rate   = {float(result.labels.y.mean()):.4f}")
    print(f"    decoded    = {len(result.decoded.road_segments)} segments")
    print()
    print("Held-out edge metrics (test) vs trivial baselines:")
    _print_metrics_vs_baseline(result.metrics.values, split_prefix="test_")
    print()
    print("Decoded network metrics, all candidate edges:")
    for k in (
        "length_precision",
        "length_recall",
        "length_f1",
        "connected_component_count",
        "predicted_total_length",
        "true_total_length",
        "runtime_sec",
    ):
        v = result.metrics.values.get(k)
        if v is None:
            continue
        try:
            print(f"  {k:>30s}: {float(v):.4f}")
        except (TypeError, ValueError):
            print(f"  {k:>30s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
