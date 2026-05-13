"""Build per-split HeteroData graphs from the interim preprocessed pickles.

Reads the `*_proc.pkl` artifacts produced by `process.py` and the
`split_mask.pkl` containing connected-component splits, then writes
`train_data.pkl`, `val_data.pkl`, `test_data.pkl` under
`data/processed/graphs/` for `scripts/evaluate_checkpoint.py`.

Mirrors the notebook cell:

    split = pickle.load(open('split_mask.pkl', 'rb'))
    train_set = set.union(*split['train'])
    val_set   = set.union(*split['val'])
    test_set  = set.union(*split['test'])
    train_data, _ = dataset(MH, Line, Road, MH_R_RL, R_R_RL, train_set)
    ...
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.dataset import dataset
from pipe_network_completion.paths import GRAPH_DATA_DIR, INTERIM_DATA_DIR

INTERIM_FILES = {
    "mh": "MH_proc.pkl",
    "line": "Line_proc.pkl",
    "road": "Road_proc.pkl",
    "mh_r_rl": "MH_R_RL_proc.pkl",
    "r_r_rl": "R_R_proc.pkl",
    "split_mask": "split_mask.pkl",
}


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def dump_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def build_split_graphs(
    interim_dir: Path,
    output_dir: Path,
    seed: int,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> None:
    missing = [
        name for name in INTERIM_FILES.values() if not (interim_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing interim artifacts in "
            f"{interim_dir}: {missing}. Run process.py first."
        )

    mh = load_pickle(interim_dir / INTERIM_FILES["mh"])
    line = load_pickle(interim_dir / INTERIM_FILES["line"])
    road = load_pickle(interim_dir / INTERIM_FILES["road"])
    mh_r_rl = load_pickle(interim_dir / INTERIM_FILES["mh_r_rl"])
    r_r_rl = load_pickle(interim_dir / INTERIM_FILES["r_r_rl"])
    split_mask = load_pickle(interim_dir / INTERIM_FILES["split_mask"])

    for split_name in splits:
        if split_name not in split_mask:
            raise KeyError(
                f"split_mask.pkl has no '{split_name}' entry; available: "
                f"{sorted(split_mask)}"
            )
        components = split_mask[split_name]
        # split_mask values are lists of connected-component node-id sets;
        # the notebook flattens them with set.union before passing to dataset().
        node_ids = set().union(*components) if components else set()
        data, _ = dataset(mh, line, road, mh_r_rl, r_r_rl, node_ids, seed=seed)
        out_path = output_dir / f"{split_name}_data.pkl"
        dump_pickle(data, out_path)
        edge_count = data["MH", "link", "MH"].edge_label_index.shape[1]
        print(
            f"{split_name}: MH={data['MH'].num_nodes}, "
            f"Road={data['Road'].num_nodes}, "
            f"labeled_MH_edges={edge_count} -> "
            f"{out_path.relative_to(REPO_ROOT)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble per-split HeteroData graphs from the interim "
            "preprocessed pickles."
        )
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=INTERIM_DATA_DIR,
        help="Directory containing *_proc.pkl and split_mask.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GRAPH_DATA_DIR,
        help="Directory to write {train,val,test}_data.pkl into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for negative edge sampling inside dataset().",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Subset of splits to build.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_split_graphs(
        interim_dir=args.interim_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        splits=tuple(args.splits),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
