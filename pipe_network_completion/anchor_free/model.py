"""Small road-edge GNN for anchor-free utility-corridor classification."""

# Workstream: Codex + Claude merge

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from pipe_network_completion.anchor_free.features import (
    RoadEdgeFeatureTable,
    assert_no_anchor_features,
)
from pipe_network_completion.anchor_free.road_graph import RoadCandidateGraph


@dataclass(frozen=True)
class GNNTrainingResult:
    model: "RoadEdgeGNN"
    probabilities: np.ndarray
    losses: list[float]
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    device: str


class RoadEdgeGNN(torch.nn.Module):
    """Road-node message passing with an edge decoder for candidate corridors."""

    def __init__(
        self,
        *,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.node_encoder = torch.nn.Linear(node_input_dim, hidden_dim)
        self.convs = torch.nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(max(int(num_layers), 1))]
        )
        decoder_input_dim = hidden_dim * 4 + edge_input_dim
        self.edge_decoder = torch.nn.Sequential(
            torch.nn.Linear(decoder_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = F.relu(self.node_encoder(data.x.float()))
        for conv in self.convs:
            x_next = conv(x, data.edge_index)
            x = F.dropout(F.relu(x_next), p=self.dropout, training=self.training)

        edge_index = data.edge_label_index
        h_u = x[edge_index[0]]
        h_v = x[edge_index[1]]
        edge_features = data.edge_label_attr.float()
        decoder_input = torch.cat(
            [h_u, h_v, torch.abs(h_u - h_v), h_u * h_v, edge_features],
            dim=1,
        )
        return self.edge_decoder(decoder_input).reshape(-1)


def _node_feature_tensor(
    graph: RoadCandidateGraph, *, include_coords: bool = True
) -> torch.Tensor:
    """Return per-node feature tensor for the road-edge GNN.

    Stage 2 of audit_followup_implementation_plan.md adds the
    ``include_coords`` toggle so we can ablate whether absolute road-node
    ``x, y`` coordinates act as a location-memorisation shortcut. When
    ``include_coords=False`` only the (centred / scaled) degree feature is
    kept, so the GNN must rely on neighbourhood structure and edge
    features rather than absolute position.
    """
    feature_dim = 3 if include_coords else 1
    if graph.nodes.empty:
        return torch.zeros((0, feature_dim), dtype=torch.float32)

    nodes = graph.nodes.sort_values("node_id").copy()
    degree = nodes[["degree"]].to_numpy(dtype=float)
    degree_scale = degree.std(axis=0, keepdims=True)
    degree_scale[degree_scale == 0.0] = 1.0
    degree = (degree - degree.mean(axis=0, keepdims=True)) / degree_scale

    if not include_coords:
        return torch.tensor(degree, dtype=torch.float32)

    coords = nodes[["x", "y"]].to_numpy(dtype=float)
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = coords.std(axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    coords = coords / scale
    return torch.tensor(np.column_stack([coords, degree]), dtype=torch.float32)


def torch_device_report() -> dict:
    """Return the PyTorch/CUDA device state for CLI preflight reporting."""

    devices = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ]
    return {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": devices,
    }


def resolve_torch_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when PyTorch can actually use it."""

    if device is None:
        device = "auto"
    if isinstance(device, torch.device):
        return device
    requested = str(device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested for the anchor-free GNN, but this PyTorch "
            "environment reports torch.cuda.is_available() == False."
        )
    return resolved


def build_pyg_road_edge_data(
    graph: RoadCandidateGraph,
    edge_features: RoadEdgeFeatureTable | pd.DataFrame | np.ndarray,
    *,
    labels: np.ndarray | None = None,
    include_node_coords: bool = True,
) -> Data:
    """Convert the road candidate graph into a PyG ``Data`` object.

    Stage 2 of audit_followup_implementation_plan.md: ``include_node_coords``
    (default True) controls whether the GNN sees absolute road-node ``x, y``.
    Set to False to ablate the location-memorisation shortcut.
    """

    edges = graph.edges.sort_values("edge_id").copy()
    edge_ids = edges["edge_id"].astype(int).to_numpy()

    if isinstance(edge_features, RoadEdgeFeatureTable):
        feature_frame = edge_features.features.loc[edge_ids]
    elif isinstance(edge_features, pd.DataFrame):
        assert_no_anchor_features(edge_features.columns)
        feature_frame = edge_features.loc[edge_ids]
    else:
        feature_frame = pd.DataFrame(edge_features, index=edge_ids)

    assert_no_anchor_features(feature_frame.columns)
    x = _node_feature_tensor(graph, include_coords=include_node_coords)
    edge_label_index = torch.tensor(edges[["u", "v"]].to_numpy(dtype=int).T, dtype=torch.long)
    message_edges = np.concatenate(
        [
            edges[["u", "v"]].to_numpy(dtype=int),
            edges[["v", "u"]].to_numpy(dtype=int),
        ],
        axis=0,
    )
    data = Data(
        x=x,
        edge_index=torch.tensor(message_edges.T, dtype=torch.long),
        edge_label_index=edge_label_index,
        edge_label_attr=torch.tensor(feature_frame.to_numpy(dtype=float), dtype=torch.float32),
        edge_id=torch.tensor(edge_ids, dtype=torch.long),
    )
    if labels is not None:
        data.edge_label = torch.tensor(np.asarray(labels, dtype=float), dtype=torch.float32)
    return data


def _positive_weight(labels: torch.Tensor, train_index: np.ndarray) -> torch.Tensor:
    train_labels = labels[torch.tensor(train_index, dtype=torch.long)]
    positives = torch.clamp(train_labels.sum(), min=1.0)
    negatives = torch.clamp(train_labels.numel() - train_labels.sum(), min=1.0)
    return negatives / positives


def train_road_edge_gnn(
    data: Data,
    *,
    train_index: np.ndarray,
    val_index: np.ndarray | None = None,
    test_index: np.ndarray | None = None,
    seed: int = 42,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    lr: float = 0.001,
    epochs: int = 100,
    weight_decay: float = 0.0,
    device: str | torch.device | None = "auto",
) -> GNNTrainingResult:
    """Train the road-edge GNN with BCEWithLogitsLoss."""

    if not hasattr(data, "edge_label"):
        raise ValueError("data.edge_label is required for GNN training.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_torch_device(device)
    data = data.to(device)
    train_index = np.asarray(train_index, dtype=int)
    val_index = np.asarray(val_index if val_index is not None else [], dtype=int)
    test_index = np.asarray(test_index if test_index is not None else [], dtype=int)

    model = RoadEdgeGNN(
        node_input_dim=int(data.x.shape[1]),
        edge_input_dim=int(data.edge_label_attr.shape[1]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = _positive_weight(data.edge_label, train_index).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_tensor = torch.tensor(train_index, dtype=torch.long, device=device)

    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[train_tensor], data.edge_label[train_tensor])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(data)).detach().cpu().numpy()

    return GNNTrainingResult(
        model=model.cpu(),
        probabilities=np.asarray(probabilities, dtype=float),
        losses=losses,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        device=str(device),
    )


# ---------------------------------------------------------------------------
# Phase 2.A — Heterogeneous (RoadSegment + Intersection) GNN.
# Workstream: Claude
#
# Mirrors the to_hetero() pattern used by the ISARC anchor-based pipeline at
# pipe_network_completion/model.py:98-101 — we lift a homogeneous GNN over the
# anchor-free relation set so the architectural skeleton matches the
# published paper's.
# ---------------------------------------------------------------------------
from torch_geometric.data import HeteroData  # noqa: E402
import torch_geometric.transforms as T  # noqa: E402
from torch_geometric.nn import GATConv, GraphConv, to_hetero  # noqa: E402


def build_hetero_pyg_data(
    graph,
    segment_features,
    intersection_features,
    labels: np.ndarray | None = None,
) -> HeteroData:
    """Lift a HeteroRoadGraph + feature tables into a PyG HeteroData object.

    Mirrors dataset.py's HeteroData construction in the ISARC anchor-based
    pipeline. ``T.ToUndirected()`` adds the reverse relation
    ``("Intersection", "rev_touches", "RoadSegment")`` automatically.
    """
    seg_x = torch.tensor(segment_features.to_numpy(), dtype=torch.float32)
    inter_x = torch.tensor(intersection_features.to_numpy(), dtype=torch.float32)

    crosses = torch.tensor(
        np.asarray(graph.segment_crosses_segment, dtype=np.int64), dtype=torch.long
    )
    touches = torch.tensor(
        np.asarray(graph.segment_touches_intersection, dtype=np.int64), dtype=torch.long
    )

    data = HeteroData()
    data["RoadSegment"].x = seg_x
    data["RoadSegment"].node_id = torch.tensor(
        np.asarray(segment_features.segment_ids, dtype=np.int64), dtype=torch.long
    )
    data["Intersection"].x = inter_x
    data["Intersection"].node_id = torch.tensor(
        np.asarray(intersection_features.intersection_ids, dtype=np.int64),
        dtype=torch.long,
    )

    data["RoadSegment", "crosses", "RoadSegment"].edge_index = crosses
    data["RoadSegment", "touches", "Intersection"].edge_index = touches

    data = T.ToUndirected()(data)

    if labels is not None:
        data["RoadSegment"].y = torch.tensor(np.asarray(labels, dtype=float), dtype=torch.float32)
    return data


def _normalise_hetero_layer_type(layer_type: str) -> str:
    """Return the supported hetero message-passing layer family.

    Workstream: Codex. True PyG ``GCNConv`` is deliberately not used here
    because the anchor-free graph is heterogeneous and includes bipartite
    RoadSegment-Intersection relations. ``GraphConv`` is the supported
    GCN-style operator for this ablation because it accepts bipartite
    ``(-1, -1)`` inputs under ``to_hetero``.
    """

    layer = str(layer_type or "sage").strip().lower().replace("-", "_")
    aliases = {
        "sage": "sage",
        "graphsage": "sage",
        "gat": "gat",
        "graph_attention": "gat",
        "graphconv": "graphconv",
        "graph_conv": "graphconv",
        "gcn_style": "graphconv",
        "gcn_like": "graphconv",
    }
    if layer in aliases:
        return aliases[layer]
    if layer in {"gcn", "gcnconv"}:
        raise ValueError(
            "GCNConv is not supported for the current heterogeneous "
            "RoadSegment-Intersection graph because it cannot handle "
            "bipartite relations. Use 'graphconv' for the GCN-style "
            "ablation, or redesign the model as a homogeneous graph."
        )
    raise ValueError(
        f"Unsupported hetero GNN layer type: {layer_type!r}. "
        "Supported: 'sage', 'gat', 'graphconv'."
    )


def _make_hetero_conv(
    layer_type: str,
    *,
    hidden_dim: int,
    dropout: float,
    gat_heads: int,
):
    layer = _normalise_hetero_layer_type(layer_type)
    if layer == "sage":
        return SAGEConv((-1, -1), hidden_dim)
    if layer == "gat":
        return GATConv(
            (-1, -1),
            hidden_dim,
            heads=max(int(gat_heads), 1),
            concat=False,
            dropout=float(dropout),
            add_self_loops=False,
        )
    if layer == "graphconv":
        return GraphConv((-1, -1), hidden_dim)
    raise AssertionError(f"Unhandled hetero GNN layer type: {layer}")


class _HomogeneousBaseGNN(torch.nn.Module):
    """Plain SAGE stack — gets lifted to heterogeneous by to_hetero."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        *,
        layer_type: str = "sage",
        gat_heads: int = 1,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.layer_type = _normalise_hetero_layer_type(layer_type)
        self.convs = torch.nn.ModuleList(
            [
                _make_hetero_conv(
                    self.layer_type,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    gat_heads=gat_heads,
                )
                for _ in range(max(int(num_layers), 1))
            ]
        )

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class HeteroRoadGNN(torch.nn.Module):
    """Heterogeneous GNN over (RoadSegment + Intersection) with per-segment head.

    Architecture mirrors pipe_network_completion/model.py:98 — to_hetero
    wraps a homogeneous base over the relation set, then a small MLP head
    on the RoadSegment node embeddings produces the per-segment logit.
    """

    def __init__(
        self,
        *,
        metadata,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        layer_type: str = "sage",
        gat_heads: int = 1,
    ):
        super().__init__()
        self.dropout = float(dropout)
        self.layer_type = _normalise_hetero_layer_type(layer_type)
        base = _HomogeneousBaseGNN(
            hidden_dim,
            num_layers,
            dropout,
            layer_type=self.layer_type,
            gat_heads=gat_heads,
        )
        self.gnn = to_hetero(base, metadata=metadata, aggr="sum")
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = self.gnn(data.x_dict, data.edge_index_dict)
        h = x_dict["RoadSegment"]
        return self.head(h).reshape(-1)


@dataclass(frozen=True)
class HeteroGNNTrainingResult:
    model: "HeteroRoadGNN"
    probabilities: np.ndarray
    losses: list[float]
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    device: str


def _hetero_positive_weight(
    labels: torch.Tensor, train_index: np.ndarray
) -> torch.Tensor:
    train_labels = labels[torch.tensor(train_index, dtype=torch.long)]
    positives = torch.clamp(train_labels.sum(), min=1.0)
    negatives = torch.clamp(train_labels.numel() - train_labels.sum(), min=1.0)
    return negatives / positives


def train_hetero_road_gnn(
    data: HeteroData,
    *,
    train_index: np.ndarray,
    val_index: np.ndarray | None = None,
    test_index: np.ndarray | None = None,
    seed: int = 42,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
    lr: float = 0.001,
    epochs: int = 100,
    weight_decay: float = 0.0,
    device: str | torch.device | None = "auto",
    layer_type: str = "sage",
    gat_heads: int = 1,
) -> HeteroGNNTrainingResult:
    """Train HeteroRoadGNN with per-RoadSegment BCEWithLogitsLoss."""
    if "y" not in data["RoadSegment"]:
        raise ValueError("HeteroData['RoadSegment'].y is required to train.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_torch_device(device)
    data = data.to(device)
    train_index = np.asarray(train_index, dtype=int)
    val_index = np.asarray(val_index if val_index is not None else [], dtype=int)
    test_index = np.asarray(test_index if test_index is not None else [], dtype=int)

    model = HeteroRoadGNN(
        metadata=data.metadata(),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        layer_type=layer_type,
        gat_heads=gat_heads,
    ).to(device)

    # Lazy init: PyG convs with (-1, -1) materialise weights on the
    # first forward pass, so we run one dummy pass before building the
    # optimizer.
    with torch.no_grad():
        _ = model(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = _hetero_positive_weight(data["RoadSegment"].y, train_index).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_tensor = torch.tensor(train_index, dtype=torch.long, device=device)

    losses: list[float] = []
    for _ in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[train_tensor], data["RoadSegment"].y[train_tensor])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(data)).detach().cpu().numpy()

    return HeteroGNNTrainingResult(
        model=model.cpu(),
        probabilities=np.asarray(probabilities, dtype=float),
        losses=losses,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        device=str(device),
    )
