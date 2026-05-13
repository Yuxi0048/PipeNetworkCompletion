from __future__ import annotations

import argparse
import importlib
import platform
import pickle
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.paths import CHECKPOINT_DIR, GRAPH_DATA_DIR, METRICS_DIR

EXPECTED_REPO_FILES = [
    "legacy/Inductive.ipynb",
    "legacy/linemap.py",
    "legacy/Module/dataset.py",
    "legacy/Module/LocationEncoder.py",
    "process.py",
    "scripts/build_graphs.py",
    "pipe_network_completion/dataset.py",
    "pipe_network_completion/location_encoder.py",
    "pipe_network_completion/model.py",
    "pipe_network_completion/evaluation.py",
]

EXPECTED_DATA_ARTIFACTS = [
    str(GRAPH_DATA_DIR.relative_to(REPO_ROOT) / "train_data.pkl"),
    str(GRAPH_DATA_DIR.relative_to(REPO_ROOT) / "val_data.pkl"),
    str(GRAPH_DATA_DIR.relative_to(REPO_ROOT) / "test_data.pkl"),
    str(
        CHECKPOINT_DIR.relative_to(REPO_ROOT)
        / "model1212_hiddensize_128_drop_00.pt"
    ),
    str(METRICS_DIR.relative_to(REPO_ROOT) / "model_metrics1212.csv"),
]


def import_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return getattr(module, "__version__", "installed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the replication environment.")
    parser.add_argument(
        "--load-data",
        action="store_true",
        help="Also load the prepared HeteroData pickles and print graph metadata.",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Require downloaded data/checkpoint artifacts to be present.",
    )
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    required_modules = [
        "numpy",
        "pandas",
        "geopandas",
        "sklearn",
        "torch",
        "torch_geometric",
        "torch_sparse",
    ]
    for module_name in required_modules:
        try:
            print(f"{module_name}: {import_version(module_name)}")
        except Exception as exc:
            print(f"{module_name}: MISSING ({exc})")
            return 1

    try:
        print(f"pyg_lib: {import_version('pyg_lib')}")
    except Exception as exc:
        print(f"pyg_lib: optional missing ({exc})")

    for module_name in [
        "pipe_network_completion.location_encoder",
        "pipe_network_completion.dataset",
        "pipe_network_completion.model",
        "pipe_network_completion.evaluation",
    ]:
        importlib.import_module(module_name)
        print(f"{module_name}: import ok")

    expected = list(EXPECTED_REPO_FILES)
    if args.require_artifacts or args.load_data:
        expected.extend(EXPECTED_DATA_ARTIFACTS)

    missing = [artifact for artifact in expected if not (REPO_ROOT / artifact).exists()]
    if missing:
        print("Missing expected repo artifacts:")
        for artifact in missing:
            print(f"  - {artifact}")
        return 1

    if args.load_data:
        for split_name in ["train", "val", "test"]:
            data_path = GRAPH_DATA_DIR / f"{split_name}_data.pkl"
            with data_path.open("rb") as handle:
                data = pickle.load(handle)
            edge_count = data["MH", "link", "MH"].edge_label_index.shape[1]
            print(
                f"{split_name}: MH={data['MH'].num_nodes}, "
                f"Road={data['Road'].num_nodes}, "
                f"labeled_MH_edges={edge_count}"
            )

    print("Environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
