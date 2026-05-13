from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_GIS_DIR = RAW_DATA_DIR / "gis"
RAW_SEWER_DIR = RAW_GIS_DIR / "sewer"
RAW_ROADS_DIR = RAW_GIS_DIR / "roads"
RAW_MH_ROAD_DIR = RAW_DATA_DIR / "mh_road"

INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPH_DATA_DIR = PROCESSED_DATA_DIR / "graphs"
SPLIT_SHAPEFILE_DIR = PROCESSED_DATA_DIR / "split_shapefiles"
EXPERIMENT_DATA_DIR = DATA_DIR / "experiments"

RESULTS_DIR = REPO_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"

MODELS_DIR = REPO_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

