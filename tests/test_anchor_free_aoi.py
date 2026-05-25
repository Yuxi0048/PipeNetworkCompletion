from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Point

from pipe_network_completion.anchor_free.aoi import (
    AOIThresholds,
    assign_aoi_splits,
    clip_vector_to_aoi,
    make_non_overlapping_grid_aois,
    select_viable_aois,
    summarize_aoi_content,
)


def test_non_overlapping_grid_aois_respect_gap():
    aois = make_non_overlapping_grid_aois(
        (0, 0, 260, 120),
        tile_size_m=100,
        gap_m=20,
        crs="EPSG:3857",
    )

    assert len(aois) == 2
    assert not aois.geometry.iloc[0].intersects(aois.geometry.iloc[1])
    assert aois.geometry.iloc[0].distance(aois.geometry.iloc[1]) >= 20


def test_aoi_summary_selection_and_split_are_area_level():
    aois = make_non_overlapping_grid_aois(
        (0, 0, 340, 100),
        tile_size_m=100,
        gap_m=20,
        crs="EPSG:3857",
    )
    roads = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 50), (100, 50)]),
                LineString([(120, 50), (220, 50)]),
                LineString([(240, 50), (340, 50)]),
            ]
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    truth = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 60), (100, 60)]),
                LineString([(120, 60), (220, 60)]),
                LineString([(240, 60), (340, 60)]),
            ]
        },
        geometry="geometry",
        crs="EPSG:3857",
    )
    buildings = gpd.GeoDataFrame(
        {"geometry": [Point(10, 10), Point(130, 10), Point(250, 10)]},
        geometry="geometry",
        crs="EPSG:3857",
    )

    summary = summarize_aoi_content(
        aois,
        roads=roads,
        utility_truth=truth,
        building_points=buildings,
    )
    selected = select_viable_aois(
        summary,
        thresholds=AOIThresholds(
            min_road_length_m=90,
            min_truth_length_m=90,
            min_building_points=1,
        ),
        max_aois=3,
        min_gap_m=20,
    )
    selected = assign_aoi_splits(selected, seed=7, train_fraction=0.5, val_fraction=0.25)

    assert len(selected) == 3
    assert set(selected["split"]) == {"train", "val", "test"}
    assert selected["aoi_id"].is_unique


def test_clip_vector_to_aoi_returns_only_inside_geometry():
    aoi = make_non_overlapping_grid_aois(
        (0, 0, 100, 100),
        tile_size_m=100,
        crs="EPSG:3857",
    ).iloc[[0]]
    lines = gpd.GeoDataFrame(
        {"geometry": [LineString([(-10, 50), (50, 50)]), LineString([(200, 0), (300, 0)])]},
        geometry="geometry",
        crs="EPSG:3857",
    )

    clipped = clip_vector_to_aoi(lines, aoi)

    assert len(clipped) == 1
    assert clipped.length.sum() == 50
