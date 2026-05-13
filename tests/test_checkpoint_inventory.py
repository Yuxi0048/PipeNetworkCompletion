"""Cross-check that metrics CSVs and saved checkpoints stay in sync."""

from __future__ import annotations

import csv
import re
from collections import defaultdict

import pytest

from pipe_network_completion import paths

CHECKPOINT_PATTERN = re.compile(r"^model(?P<arch>[a-zA-Z0-9]+)_hiddensize_(?P<h>\d+)_drop_(?P<dd>\d+)\.pt$")

ARCH_TO_METRICS_CSV = {
    "04": "model_metrics04.csv",
    "22": "model_metrics.csv",
    "1111": "model_metrics.csv",
    "1212": "model_metrics1212.csv",
    "skip": "model_metrics1212_skip.csv",
}


def _checkpoint_index() -> dict[str, set[tuple[int, float]]]:
    index: dict[str, set[tuple[int, float]]] = defaultdict(set)
    for ckpt in paths.CHECKPOINT_DIR.glob("*.pt"):
        match = CHECKPOINT_PATTERN.match(ckpt.name)
        if not match:
            continue
        hidden = int(match.group("h"))
        dropout = int(match.group("dd")) / 10.0
        index[match.group("arch")].add((hidden, dropout))
    return index


def test_checkpoint_filenames_follow_schema() -> None:
    unknown = [
        ckpt.name
        for ckpt in paths.CHECKPOINT_DIR.glob("*.pt")
        if not CHECKPOINT_PATTERN.match(ckpt.name) and ckpt.name != "checkpoint.pt"
    ]
    assert unknown == [], (
        "checkpoints not matching model<arch>_hiddensize_<H>_drop_<DD>.pt: "
        f"{unknown}. Either rename or document in models/README.md."
    )


@pytest.mark.parametrize("arch,csv_name", sorted(ARCH_TO_METRICS_CSV.items()))
def test_metrics_csv_rows_have_a_checkpoint(arch: str, csv_name: str) -> None:
    csv_path = paths.METRICS_DIR / csv_name
    if not csv_path.exists():
        pytest.skip(f"metrics file {csv_name} not present")

    index = _checkpoint_index()
    available_for_arch = index.get(arch, set())
    if not available_for_arch:
        pytest.skip(f"no checkpoints found for arch '{arch}'")

    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            # Some metrics CSVs concatenate multiple architecture sections,
            # each starting with a repeated header row; skip those.
            try:
                hidden = int(row["Model Size"])
                dropout = float(row["Dropout"])
            except (TypeError, ValueError):
                continue
            # Some metrics CSVs aggregate multiple architectures; we only
            # require that *some* arch has a matching checkpoint for the row.
            covered = any(
                (hidden, dropout) in index[candidate_arch]
                for candidate_arch in index
            )
            assert covered, (
                f"{csv_name}: row hidden={hidden} dropout={dropout} has no "
                "corresponding checkpoint on disk."
            )
