"""Analyze non-parcel cadastral support regions for sewer-main alignment.

Workstream: Codex

This script does not train a model. It estimates whether non-parcel cadastral
evidence defines useful support regions for sewer mains by comparing:

- truth recall: fraction of sampled sewer-main length inside a support region;
- background area fraction: fraction of random study-area points inside the
  same support region;
- enrichment: truth recall divided by background area fraction.

Parcel polygons are intentionally not used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_GDAL_DATA = Path(sys.prefix) / "Library" / "share" / "gdal"
if "GDAL_DATA" not in os.environ and DEFAULT_GDAL_DATA.exists():
    os.environ["GDAL_DATA"] = str(DEFAULT_GDAL_DATA)

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform as transform_geometry


DEFAULT_TRUTH = (
    ("gravity_trunk", REPO_ROOT / "data" / "raw" / "gis" / "sewer" / "SewerGravityMa_ExportFeature1.shp"),
    ("gravity_main", REPO_ROOT / "data" / "raw" / "gis" / "sewer" / "SewerGravityMa_ExportFeature2.shp"),
    ("pressure_main", REPO_ROOT / "data" / "raw" / "gis" / "sewer" / "SewerPressureM_ExportFeature.shp"),
)
DEFAULT_SPLIT_TRUTH = (
    ("train", REPO_ROOT / "data" / "processed" / "split_shapefiles" / "train.shp"),
    ("val", REPO_ROOT / "data" / "processed" / "split_shapefiles" / "val.shp"),
    ("test", REPO_ROOT / "data" / "processed" / "split_shapefiles" / "test.shp"),
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_layer(path: Path, target_crs: str, *, columns: list[str] | None = None) -> gpd.GeoDataFrame:
    kwargs = {"columns": columns} if columns is not None else {}
    data = gpd.read_file(path, **kwargs)
    if data.crs is None:
        raise ValueError(f"Layer has no CRS: {path}")
    data = data.to_crs(target_crs)
    data = data[data.geometry.notna() & ~data.geometry.is_empty].copy()
    return data


def _line_parts(geom):
    if geom is None or geom.is_empty:
        return
    if geom.geom_type in {"LineString", "LinearRing"}:
        yield geom
    elif hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from _line_parts(part)


def _load_raw_main_truth(
    target_crs: str,
    study_area: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    frames = []
    for source_name, path in DEFAULT_TRUTH:
        if not path.exists():
            continue
        frame = _read_layer(path, target_crs)
        frame["truth_source"] = source_name
        frame = gpd.clip(frame, study_area)
        frame = frame[frame.geometry.length > 0.0].copy()
        frames.append(frame[["truth_source", "geometry"]])
    if not frames:
        return gpd.GeoDataFrame({"truth_source": []}, geometry=[], crs=target_crs)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=target_crs)


def _load_split_truth(target_crs: str) -> gpd.GeoDataFrame:
    frames = []
    for split_name, path in DEFAULT_SPLIT_TRUTH:
        if not path.exists():
            continue
        frame = _read_layer(path, target_crs)
        frame["truth_source"] = f"paper_split_{split_name}"
        frame = frame[frame.geometry.length > 0.0].copy()
        frames.append(frame[["truth_source", "geometry"]])
    if not frames:
        return gpd.GeoDataFrame({"truth_source": []}, geometry=[], crs=target_crs)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=target_crs)


def _sample_truth_lines(truth: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    rows: list[dict[str, object]] = []
    spacing = max(float(spacing_m), 1.0)
    sample_id = 0
    for row in truth.itertuples(index=False):
        for line in _line_parts(row.geometry):
            length = float(line.length)
            if length <= 0.0:
                continue
            n = max(1, int(np.ceil(length / spacing)))
            weight = length / n
            for i in range(n):
                distance = min((i + 0.5) * weight, length)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_kind": "truth",
                        "truth_source": row.truth_source,
                        "weight_m": float(weight),
                        "geometry": line.interpolate(distance),
                    }
                )
                sample_id += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=truth.crs)


def _sample_split_truth_lines(target_crs: str, spacing_m: float) -> gpd.GeoDataFrame:
    rows: list[dict[str, object]] = []
    spacing = max(float(spacing_m), 1.0)
    target = CRS.from_user_input(target_crs)
    sample_id = 0
    for split_name, path in DEFAULT_SPLIT_TRUTH:
        with fiona.open(path) as src:
            transformer = Transformer.from_crs(CRS.from_user_input(src.crs), target, always_xy=True)
            for feature in src:
                if not feature.get("geometry"):
                    continue
                geom = transform_geometry(transformer.transform, shape(feature["geometry"]))
                for line in _line_parts(geom):
                    length = float(line.length)
                    if length <= 0.0:
                        continue
                    n = max(1, int(np.ceil(length / spacing)))
                    weight = length / n
                    for i in range(n):
                        distance = min((i + 0.5) * weight, length)
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "sample_kind": "truth",
                                "truth_source": f"paper_split_{split_name}",
                                "weight_m": float(weight),
                                "geometry": line.interpolate(distance),
                            }
                        )
                        sample_id += 1
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)


def _sample_background(
    study_area: gpd.GeoDataFrame,
    n_points: int,
    seed: int,
) -> gpd.GeoDataFrame:
    geom = study_area.geometry.iloc[0]
    xmin, ymin, xmax, ymax = geom.bounds
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    batch = max(int(n_points * 1.25), 1000)
    sample_id = 0
    while len(rows) < n_points:
        xs = rng.uniform(xmin, xmax, batch)
        ys = rng.uniform(ymin, ymax, batch)
        points = [Point(float(x), float(y)) for x, y in zip(xs, ys)]
        for point in points:
            if geom.covers(point):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_kind": "background",
                        "truth_source": "",
                        "weight_m": 1.0,
                        "geometry": point,
                    }
                )
                sample_id += 1
                if len(rows) >= n_points:
                    break
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=study_area.crs)


def _nearest_distances(
    samples: gpd.GeoDataFrame,
    support_path: Path,
    *,
    target_crs: str,
    max_distance_m: float,
    label: str,
) -> pd.Series:
    print(f"loading support layer: {label}", flush=True)
    support = _read_layer(support_path, target_crs, columns=[])
    support = support[support.geometry.notna() & ~support.geometry.is_empty].copy()
    support = support[["geometry"]].reset_index(drop=True)
    support["support_id"] = support.index.astype("int64")
    joined = gpd.sjoin_nearest(
        samples[["sample_id", "geometry"]],
        support,
        how="left",
        max_distance=float(max_distance_m),
        distance_col=f"dist_to_{label}_m",
    )
    distances = joined.groupby("sample_id")[f"dist_to_{label}_m"].min()
    return samples["sample_id"].map(distances).astype(float)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> list[float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not mask.any():
        return [float("nan") for _ in quantiles]
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / float(np.sum(weights))
    return [float(np.interp(q, cdf, values)) for q in quantiles]


def _variant_masks(samples: pd.DataFrame) -> dict[str, pd.Series]:
    road_buffers = [5, 10, 20, 30, 50]
    natural_buffers = [10, 20, 30, 50, 100]
    address_radii = [20, 30, 50, 80, 100]
    masks: dict[str, pd.Series] = {}

    for value in road_buffers:
        masks[f"cadastral_road_{value}m"] = samples["dist_to_cadastral_road_m"] <= value
    for value in natural_buffers:
        masks[f"natural_boundary_{value}m"] = samples["dist_to_natural_boundary_m"] <= value
    for value in address_radii:
        masks[f"address_point_{value}m"] = samples["dist_to_address_point_m"] <= value

    combos = [
        ("road20_or_natural30", 20, None, 30),
        ("road30_or_natural50", 30, None, 50),
        ("road20_or_address50", 20, 50, None),
        ("road30_or_address80", 30, 80, None),
        ("all_nonparcel_road20_addr50_nat30", 20, 50, 30),
        ("all_nonparcel_road30_addr80_nat50", 30, 80, 50),
    ]
    for name, road_m, address_m, natural_m in combos:
        mask = samples["dist_to_cadastral_road_m"] <= road_m
        if address_m is not None:
            mask = mask | (samples["dist_to_address_point_m"] <= address_m)
        if natural_m is not None:
            mask = mask | (samples["dist_to_natural_boundary_m"] <= natural_m)
        masks[name] = mask
    return masks


def _summarize_variants(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = samples[samples["sample_kind"] == "truth"].copy()
    background = samples[samples["sample_kind"] == "background"].copy()
    total_truth_weight = float(truth["weight_m"].sum())
    rows = []
    source_rows = []

    for name, mask_all in _variant_masks(samples).items():
        truth_mask = mask_all.loc[truth.index].fillna(False)
        background_mask = mask_all.loc[background.index].fillna(False)
        covered_truth = float(truth.loc[truth_mask, "weight_m"].sum())
        recall = covered_truth / total_truth_weight if total_truth_weight > 0.0 else float("nan")
        area_fraction = float(background_mask.mean()) if len(background_mask) else float("nan")
        enrichment = recall / area_fraction if area_fraction and area_fraction > 0.0 else float("inf")
        selectivity = 1.0 - area_fraction if np.isfinite(area_fraction) else float("nan")
        support_score = (
            2.0 * recall * selectivity / (recall + selectivity)
            if recall + selectivity > 0.0 and np.isfinite(selectivity)
            else float("nan")
        )
        rows.append(
            {
                "variant": name,
                "truth_recall": recall,
                "covered_truth_length_m": covered_truth,
                "total_truth_length_sampled_m": total_truth_weight,
                "background_area_fraction": area_fraction,
                "enrichment_vs_background": enrichment,
                "selectivity_1_minus_area_fraction": selectivity,
                "support_score_recall_selectivity_hmean": support_score,
                "truth_sample_count": int(len(truth)),
                "background_sample_count": int(len(background)),
            }
        )
        for source, group in truth.groupby("truth_source"):
            source_mask = mask_all.loc[group.index].fillna(False)
            source_total = float(group["weight_m"].sum())
            source_covered = float(group.loc[source_mask, "weight_m"].sum())
            source_rows.append(
                {
                    "variant": name,
                    "truth_source": source,
                    "truth_recall": source_covered / source_total if source_total > 0.0 else float("nan"),
                    "covered_truth_length_m": source_covered,
                    "total_truth_length_sampled_m": source_total,
                    "truth_sample_count": int(len(group)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(source_rows)


def _distance_quantiles(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for sample_kind, group in samples.groupby("sample_kind"):
        weights = group["weight_m"].to_numpy(dtype=float)
        for distance_column in [
            "dist_to_cadastral_road_m",
            "dist_to_address_point_m",
            "dist_to_natural_boundary_m",
        ]:
            values = group[distance_column].to_numpy(dtype=float)
            qs = _weighted_quantile(values, weights, quantiles)
            row = {
                "sample_kind": sample_kind,
                "distance_column": distance_column,
                "finite_fraction": float(np.isfinite(values).mean()),
            }
            row.update({f"q{int(q * 100):02d}_m": value for q, value in zip(quantiles, qs)})
            rows.append(row)
    return pd.DataFrame(rows)


def _write_plot(summary: pd.DataFrame, output_path: Path) -> None:
    plot_data = summary.sort_values("truth_recall")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.scatter(
        plot_data["background_area_fraction"],
        plot_data["truth_recall"],
        s=42,
        c=plot_data["enrichment_vs_background"].clip(upper=10.0),
        cmap="viridis",
    )
    for row in plot_data.itertuples(index=False):
        if row.variant.startswith("all_nonparcel") or row.variant in {
            "cadastral_road_20m",
            "cadastral_road_30m",
            "address_point_50m",
            "natural_boundary_50m",
        }:
            ax.annotate(row.variant, (row.background_area_fraction, row.truth_recall), fontsize=7)
    ax.plot([0, 1], [0, 1], color="0.7", linewidth=1, linestyle="--")
    ax.set_xlabel("Background area fraction")
    ax.set_ylabel("Sewer-main truth recall")
    ax.set_title("Non-parcel cadastral support enrichment")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cadastral-dir",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "processed"
        / "context"
        / "study_area"
        / "cadastral_sewer_extent_exact_epsg28356",
    )
    parser.add_argument("--target-crs", default="EPSG:28356")
    parser.add_argument("--truth-sample-spacing-m", type=float, default=50.0)
    parser.add_argument(
        "--truth-source",
        choices=("split_shapefiles", "raw_mains"),
        default="split_shapefiles",
        help="split_shapefiles matches the anchor-based paper target; raw_mains loads gravity+pressure main files.",
    )
    parser.add_argument("--background-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "cadastral_nonparcel_support_regions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cadastral_dir = args.cadastral_dir if args.cadastral_dir.is_absolute() else REPO_ROOT / args.cadastral_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary = _read_layer(cadastral_dir / "study_area_boundary_epsg28356.geojson", args.target_crs)
    if args.truth_source == "raw_mains":
        truth = _load_raw_main_truth(args.target_crs, boundary)
        truth_layers = {name: _relative(path) for name, path in DEFAULT_TRUTH}
        print(f"loaded raw truth lines: {len(truth)}", flush=True)
        truth_samples = _sample_truth_lines(truth, spacing_m=float(args.truth_sample_spacing_m))
    else:
        truth_layers = {name: _relative(path) for name, path in DEFAULT_SPLIT_TRUTH}
        print("sampling split-shapefile truth lines", flush=True)
        truth_samples = _sample_split_truth_lines(args.target_crs, float(args.truth_sample_spacing_m))
    print(f"truth samples: {len(truth_samples)}", flush=True)
    background_samples = _sample_background(boundary, int(args.background_samples), int(args.seed))
    print(f"background samples: {len(background_samples)}", flush=True)
    samples = gpd.GeoDataFrame(
        pd.concat([truth_samples, background_samples], ignore_index=True),
        geometry="geometry",
        crs=args.target_crs,
    )

    support_paths = {
        "cadastral_road": cadastral_dir / "cadastral_roads_epsg28356.fgb",
        "address_point": cadastral_dir / "address_points_epsg28356.fgb",
        "natural_boundary": cadastral_dir / "natural_boundaries_epsg28356.fgb",
    }
    max_distances = {
        "cadastral_road": 50.0,
        "address_point": 100.0,
        "natural_boundary": 100.0,
    }
    for label, path in support_paths.items():
        print(f"nearest distances: {label} <- {path.name}", flush=True)
        samples[f"dist_to_{label}_m"] = _nearest_distances(
            samples,
            path,
            target_crs=args.target_crs,
            max_distance_m=max_distances[label],
            label=label,
        )

    sample_table = pd.DataFrame(samples.drop(columns="geometry"))
    summary, by_source = _summarize_variants(sample_table)
    quantiles = _distance_quantiles(sample_table)

    summary = summary.sort_values(
        ["truth_recall", "enrichment_vs_background"],
        ascending=[False, False],
    ).reset_index(drop=True)
    summary.to_csv(output_dir / "support_region_summary.csv", index=False)
    by_source.to_csv(output_dir / "support_region_by_truth_source.csv", index=False)
    quantiles.to_csv(output_dir / "support_distance_quantiles.csv", index=False)
    sample_table.to_parquet(output_dir / "support_sample_distances.parquet", index=False)
    _write_plot(summary, output_dir / "support_region_recall_vs_area.png")

    manifest = {
        "workstream": "Codex",
        "description": "Non-parcel cadastral support region analysis; no model training.",
        "cadastral_dir": _relative(cadastral_dir),
        "excluded_layers": ["dcdb_parcels"],
        "included_layers": {key: _relative(path) for key, path in support_paths.items()},
        "truth_source": args.truth_source,
        "truth_layers": truth_layers,
        "target_crs": args.target_crs,
        "truth_sample_spacing_m": float(args.truth_sample_spacing_m),
        "background_samples": int(args.background_samples),
        "seed": int(args.seed),
        "truth_sample_count": int((sample_table["sample_kind"] == "truth").sum()),
        "background_sample_count": int((sample_table["sample_kind"] == "background").sum()),
        "outputs": {
            "summary": _relative(output_dir / "support_region_summary.csv"),
            "by_truth_source": _relative(output_dir / "support_region_by_truth_source.csv"),
            "distance_quantiles": _relative(output_dir / "support_distance_quantiles.csv"),
            "sample_distances": _relative(output_dir / "support_sample_distances.parquet"),
            "plot": _relative(output_dir / "support_region_recall_vs_area.png"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.head(12).to_string(index=False))
    print(f"wrote {_relative(output_dir)}")
    print("No model training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
