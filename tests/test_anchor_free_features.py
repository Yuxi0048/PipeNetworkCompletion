"""Tests for the anchor-free road-edge feature pipeline and anchor guard."""

# Workstream: Claude

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from pipe_network_completion.anchor_free.features import (
    FORBIDDEN_ANCHOR_TOKENS,
    assert_no_anchor_features,
    build_road_edge_features,
    standardize_features,
)
from pipe_network_completion.anchor_free.road_graph import build_road_candidate_graph
from pipe_network_completion.anchor_free.synthetic import (
    make_synthetic_anchor_free_data,
)


def _graph():
    data = make_synthetic_anchor_free_data()
    return data, build_road_candidate_graph(data.roads, target_crs="EPSG:3857")


def test_assert_no_anchor_features_rejects_explicit_token():
    with pytest.raises(ValueError):
        assert_no_anchor_features(["road_length", "manhole_distance_m"])


def test_assert_no_anchor_features_rejects_mh_abbreviation():
    with pytest.raises(ValueError):
        assert_no_anchor_features(["road_length", "near_mh_count"])


def test_assert_no_anchor_features_accepts_road_only_features():
    assert_no_anchor_features(
        [
            "length_m",
            "bearing_sin",
            "bearing_cos",
            "degree_sum",
            "local_road_density_100m",
            "building_count_50m",
        ]
    )


def test_forbidden_tokens_cover_required_categories():
    for required in (
        "manhole",
        "valve",
        "pole",
        "transformer",
        "cabinet",
        "anchor",
        "utility_node",
        "facility_node",
        "surveyed_node",
    ):
        assert required in FORBIDDEN_ANCHOR_TOKENS


def test_features_have_one_row_per_edge_and_no_anchor_names():
    data, graph = _graph()
    table = build_road_edge_features(
        graph,
        buildings_gdf=data.buildings,
        road_class_columns="road_class",
    )
    # One row per candidate edge.
    assert table.features.shape[0] == len(graph.edges)
    # Guard must pass; otherwise the constructor would already have raised.
    assert_no_anchor_features(table.feature_names)
    # The basic geometric and demand-context features should be present.
    expected_basics = {"length_m", "bearing_sin", "bearing_cos", "degree_sum"}
    assert expected_basics.issubset(set(table.feature_names))


def test_building_polygon_context_adds_area_features():
    data, graph = _graph()
    buildings = data.buildings.copy()
    buildings["geometry"] = buildings.geometry.buffer(8.0)
    buildings["dimension_m2"] = buildings.geometry.area
    buildings["ground_elevation"] = 10.0

    table = build_road_edge_features(
        graph,
        buildings_gdf=buildings,
        road_class_columns="road_class",
        building_buffer_m=50.0,
    )

    assert "building_footprint_area_sum_50m" in table.feature_names
    assert "building_footprint_area_density_50m" in table.feature_names
    assert "building_ground_elevation_mean_50m" in table.feature_names
    assert float(table.features["building_footprint_area_sum_50m"].max()) > 0.0


def test_building_point_context_adds_point_features():
    data, graph = _graph()
    points = data.buildings.copy()
    points["function"] = ["House" if i % 2 == 0 else "Shed" for i in range(len(points))]

    table = build_road_edge_features(
        graph,
        building_points_gdf=points,
        road_class_columns="road_class",
        building_buffer_m=50.0,
    )

    assert "nearest_building_point_distance_m" in table.feature_names
    assert "building_point_count_50m" in table.feature_names
    assert "building_point_density_50m" in table.feature_names
    assert "building_point_function_house_count_50m" in table.feature_names
    assert float(table.features["building_point_count_50m"].max()) > 0.0


def test_standardize_features_treats_near_constant_columns_as_constant():
    frame = pd.DataFrame(
        {
            "almost_constant": [1.0, 1.0 + 1e-14, 1.0 - 1e-14, 1.0],
            "varying": [0.0, 1.0, 2.0, 3.0],
        }
    )
    scaled, _, std = standardize_features(frame, train_index=np.array([0, 1, 2]))
    assert float(std["almost_constant"]) == 1.0
    assert float(scaled["almost_constant"].abs().max()) < 1e-10
    assert np.isfinite(scaled.to_numpy()).all()


def test_dem_features_are_added_from_projected_raster():
    gdal = pytest.importorskip("osgeo.gdal")
    osr = pytest.importorskip("osgeo.osr")

    data, graph = _graph()
    dem_dir = Path("outputs") / "test_anchor_free_features"
    dem_dir.mkdir(parents=True, exist_ok=True)
    dem_path = dem_dir / "synthetic_dem.tif"
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(dem_path), 64, 64, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((-50.0, 10.0, 0.0, 450.0, 0.0, -10.0))
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(3857)
    dataset.SetProjection(spatial_ref.ExportToWkt())
    raster = np.add.outer(np.arange(64), np.arange(64)).astype("float32")
    dataset.GetRasterBand(1).WriteArray(raster)
    dataset.FlushCache()
    dataset = None

    table = build_road_edge_features(
        graph,
        buildings_gdf=data.buildings,
        dem_path=dem_path,
        road_class_columns="road_class",
    )

    expected = {
        "elevation_u_m",
        "elevation_v_m",
        "elevation_mean_m",
        "elevation_delta_uv_m",
        "slope_uv",
        "dem_valid_fraction",
    }
    assert expected.issubset(set(table.feature_names))
    assert np.isfinite(table.features[list(expected)].to_numpy()).all()
    assert float(table.features["dem_valid_fraction"].min()) > 0.0
