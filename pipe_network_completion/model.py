from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, Linear, SAGEConv, to_hetero

from pipe_network_completion.location_encoder import (
    TheoryGridCellSpatialRelationEncoder,
)


DEFAULT_LAYERS = ("gat", "gat", "sage", "sage")


class GNN(torch.nn.Module):
    """Stacked encoder used by the saved architecture variants."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float,
        layers: tuple[str, ...] = DEFAULT_LAYERS,
        skip: bool = False,
    ):
        super().__init__()
        self.layers = layers
        self.skip = skip
        for layer_index, layer_type in enumerate(layers, start=1):
            if layer_type == "gat":
                in_channels = (-1, -1) if layer_index == 1 else hidden_channels
                conv = GATConv(
                    in_channels,
                    hidden_channels,
                    dropout=dropout,
                    add_self_loops=False,
                )
            elif layer_type == "sage":
                in_channels = -1 if layer_index == 1 else hidden_channels
                conv = SAGEConv(in_channels, hidden_channels)
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
            setattr(self, f"conv{layer_index}", conv)

        if skip:
            self.lin1 = Linear(-1, hidden_channels)
            self.lin2 = Linear(-1, hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        for layer_index, layer_type in enumerate(self.layers, start=1):
            conv = getattr(self, f"conv{layer_index}")
            if layer_type == "gat":
                next_x = conv(x, edge_index, edge_attr)
            else:
                next_x = conv(x, edge_index)

            if self.skip and layer_index == 1:
                next_x = next_x + self.lin1(x)
            elif self.skip and layer_index == 4:
                next_x = next_x + self.lin2(x)

            if layer_index < len(self.layers):
                next_x = F.relu(next_x)
            x = next_x
        return x


class Classifier(torch.nn.Module):
    """Dot-product MH-MH edge classifier used in the original notebook."""

    def forward(self, x_mh, edge_label_index):
        edge_feat_line1 = x_mh[edge_label_index[0]]
        edge_feat_line2 = x_mh[edge_label_index[1]]
        return (edge_feat_line1 * edge_feat_line2).sum(axis=-1)


class Model(torch.nn.Module):
    """Notebook model refactored into an importable module."""

    def __init__(
        self,
        hidden_channels: int,
        dropout: float,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        layers: tuple[str, ...] = DEFAULT_LAYERS,
        skip: bool = False,
        frequency_num: int = 16,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(6 * frequency_num, hidden_channels)
        self.location_encoder = TheoryGridCellSpatialRelationEncoder(
            spa_embed_dim=128,
            coord_dim=2,
            frequency_num=frequency_num,
        )
        self.gnn = to_hetero(
            GNN(hidden_channels, dropout, layers=layers, skip=skip),
            metadata=metadata,
        )
        self.classifier = Classifier()

    def forward(self, data: HeteroData):
        center = torch.mean(
            torch.cat([data.x_dict["MH"][:, :2], data.x_dict["Road"][:, :2]]),
            dim=0,
        )
        disp = {
            "MH": self.location_encoder(
                (data.x_dict["MH"][:, :2] - center).reshape(-1, 1, 2)
            ).reshape(-1, self.linear.in_features),
            "Road": self.location_encoder(
                (data.x_dict["Road"][:, :2] - center).reshape(-1, 1, 2)
            ).reshape(-1, self.linear.in_features),
        }
        x_dict = {
            "MH": torch.cat(
                [F.relu(self.linear(disp["MH"])), data.x_dict["MH"][:, 2:]],
                dim=1,
            ),
            "Road": torch.cat(
                [F.relu(self.linear(disp["Road"])), data.x_dict["Road"][:, 2:]],
                dim=1,
            ),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)
        return self.classifier(
            x_dict["MH"],
            data["MH", "link", "MH"].edge_label_index,
        )


def load_state_dict(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a PyTorch state dict across supported PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def infer_architecture_from_state_dict(
    state_dict: dict,
) -> tuple[tuple[str, ...], bool]:
    layer_indexes = sorted(
        {
            int(key.split(".")[1].replace("conv", ""))
            for key in state_dict
            if key.startswith("gnn.conv")
        }
    )
    layers = []
    for layer_index in layer_indexes:
        prefix = f"gnn.conv{layer_index}."
        layer_keys = [key for key in state_dict if key.startswith(prefix)]
        layers.append("gat" if any(".att_src" in key for key in layer_keys) else "sage")
    skip = any(key.startswith("gnn.lin1.") for key in state_dict)
    return tuple(layers), skip


def build_model_from_data(
    data: HeteroData,
    hidden_channels: int,
    dropout: float,
    device: str | torch.device = "cpu",
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    skip: bool = False,
) -> Model:
    model = Model(
        hidden_channels=hidden_channels,
        dropout=dropout,
        metadata=data.metadata(),
        layers=layers,
        skip=skip,
    )
    return model.to(device)

