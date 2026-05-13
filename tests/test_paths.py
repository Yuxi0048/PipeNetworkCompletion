"""Path constants must point inside the repo and match the documented layout.

If anyone renames a folder under data/, models/, or results/, these tests
fail and force the rename to be propagated through `paths.py` and the docs.
"""

from __future__ import annotations

from pipe_network_completion import paths


def test_repo_root_resolves_to_repo() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").exists()
    assert (paths.REPO_ROOT / "environment.yml").exists()


def test_data_dirs_are_inside_repo() -> None:
    for path in [
        paths.DATA_DIR,
        paths.RAW_DATA_DIR,
        paths.RAW_GIS_DIR,
        paths.RAW_SEWER_DIR,
        paths.RAW_ROADS_DIR,
        paths.RAW_MH_ROAD_DIR,
        paths.INTERIM_DATA_DIR,
        paths.PROCESSED_DATA_DIR,
        paths.GRAPH_DATA_DIR,
        paths.SPLIT_SHAPEFILE_DIR,
        paths.EXPERIMENT_DATA_DIR,
        paths.RESULTS_DIR,
        paths.METRICS_DIR,
        paths.MODELS_DIR,
        paths.CHECKPOINT_DIR,
    ]:
        assert path.is_absolute()
        assert paths.REPO_ROOT in path.parents or path == paths.REPO_ROOT


def test_documented_subtree_exists() -> None:
    # docs/DATA_LAYOUT.md documents these folders; they should be tracked even
    # when empty so a fresh clone matches the documented layout.
    for directory in [
        paths.RAW_SEWER_DIR,
        paths.RAW_ROADS_DIR,
        paths.INTERIM_DATA_DIR,
        paths.GRAPH_DATA_DIR,
        paths.CHECKPOINT_DIR,
        paths.METRICS_DIR,
    ]:
        assert directory.exists(), f"missing documented directory: {directory}"
