"""GPU-aware environment check for the CUDA conda env.

Runs the same module-import checks as ``scripts/verify_environment.py`` and
additionally:

* prints torch / CUDA build info
* confirms ``torch.cuda.is_available()``
* lists every visible CUDA device with name, capability, and free memory
* runs one tiny tensor op on each device to catch driver / runtime mismatch
* exercises ``torch_geometric.nn.SAGEConv`` and the anchor-free road-edge GNN
  on GPU so the full PyG stack is validated end-to-end
"""

# Workstream: Claude + Codex merge

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _import_version(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, "__version__", "installed")


def _check_modules() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    required = [
        "numpy",
        "pandas",
        "geopandas",
        "sklearn",
        "torch",
        "torch_geometric",
        "torch_sparse",
        "torch_scatter",
    ]
    for name in required:
        try:
            print(f"{name}: {_import_version(name)}")
        except Exception as exc:
            print(f"{name}: MISSING ({exc})")
            return 1
    try:
        print(f"pyg_lib: {_import_version('pyg_lib')}")
    except Exception as exc:
        print(f"pyg_lib: optional missing ({exc})")
    return 0


def _check_gpu(strict: bool) -> int:
    import torch

    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if not available:
        msg = "CUDA is not available; check driver, conda env, and pytorch-cuda pin."
        if strict:
            print(f"ERROR: {msg}")
            return 1
        print(f"WARN: {msg}")
        return 0

    n_devices = torch.cuda.device_count()
    print(f"torch.cuda.device_count(): {n_devices}")
    for i in range(n_devices):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        gib = 1024 ** 3
        print(
            f"  cuda:{i} {props.name} cc={props.major}.{props.minor} "
            f"total={total / gib:.1f} GiB free={free / gib:.1f} GiB"
        )
        # Smoke test: one tiny tensor op per device.
        x = torch.randn(1024, 1024, device=f"cuda:{i}")
        y = (x @ x).sum().item()
        print(f"    smoke matmul OK (sum={y:.2f})")
    return 0


def _check_pyg_on_gpu() -> int:
    import torch
    from torch_geometric.nn import SAGEConv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyG smoke test device: {device}")
    n_nodes, in_ch, out_ch = 32, 8, 16
    x = torch.randn(n_nodes, in_ch, device=device)
    edge_index = torch.randint(0, n_nodes, (2, 64), device=device)
    conv = SAGEConv(in_ch, out_ch).to(device)
    out = conv(x, edge_index)
    assert out.shape == (n_nodes, out_ch)
    print(f"  SAGEConv forward OK shape={tuple(out.shape)}")
    return 0


def _check_anchor_free_gnn_on_gpu() -> int:
    import torch

    from pipe_network_completion.anchor_free.features import (
        build_road_edge_features,
    )
    from pipe_network_completion.anchor_free.labels import (
        label_road_edges_from_utility_lines,
    )
    from pipe_network_completion.anchor_free.model import (
        build_pyg_road_edge_data,
        train_road_edge_gnn,
    )
    from pipe_network_completion.anchor_free.road_graph import (
        build_road_candidate_graph,
    )
    from pipe_network_completion.anchor_free.synthetic import (
        make_synthetic_anchor_free_data,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Anchor-free GNN smoke test device: {device}")
    data = make_synthetic_anchor_free_data()
    graph = build_road_candidate_graph(data.roads, target_crs="EPSG:3857")
    features = build_road_edge_features(
        graph, buildings_gdf=data.buildings, road_class_columns="road_class"
    )
    labels = label_road_edges_from_utility_lines(graph, data.utility_truth)
    pyg = build_pyg_road_edge_data(graph, features, labels=labels.y)
    import numpy as np

    train_idx = np.arange(len(labels.y))
    result = train_road_edge_gnn(
        pyg,
        train_index=train_idx,
        epochs=2,
        hidden_dim=16,
        num_layers=2,
        seed=0,
        device=device,
    )
    print(
        "  anchor-free GNN OK "
        f"(n_edges={len(labels.y)}, final_loss={result.losses[-1]:.4f})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU-aware environment check for the CUDA env."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if CUDA is not available (default: warn only).",
    )
    parser.add_argument(
        "--skip-anchor-free",
        action="store_true",
        help="Skip the anchor-free GNN smoke test (use after a clean install).",
    )
    args = parser.parse_args()

    if _check_modules() != 0:
        return 1
    if _check_gpu(strict=args.strict) != 0:
        return 1
    if _check_pyg_on_gpu() != 0:
        return 1
    if not args.skip_anchor_free:
        if _check_anchor_free_gnn_on_gpu() != 0:
            return 1
    print("GPU environment verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
