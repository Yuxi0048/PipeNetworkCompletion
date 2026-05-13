# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-05-13

Formal replication release matching the ISARC 2024 paper:
*Underground Utility Network Completion based on Spatial Contextual
Information of Ground Facilities and Utility Anchor Points using Graph Neural
Networks* (Zhang & Cai, [DOI:10.22260/ISARC2024/0121](https://doi.org/10.22260/ISARC2024/0121)).

### Added
- Installable Python package `pipe_network_completion` via `pyproject.toml`.
- `scripts/build_graphs.py` — assembles per-split `HeteroData` graphs from
  the interim preprocessed pickles, completing the
  `raw → interim → processed → metrics` pipeline.
- `models/README.md` — documents the `model<arch>_hiddensize_<H>_drop_<DD>.pt`
  naming schema and architecture-code table.
- `tests/` — pytest checks for imports, paths, model utilities, and
  checkpoint/metrics inventory consistency.
- `scripts/bundle_release_assets.py` — packages checkpoints, prepared graphs,
  and metrics into a release tarball for upload.
- `scripts/download_assets.py` — fetches release assets via the GitHub CLI
  for users who clone a slim source tree.
- `.gitignore` rules and folder placeholders keep large data/checkpoint
  artifacts out of Git history.
- `RELEASE.md` — maintainer playbook for cutting a tagged release.
- GitHub Actions CI workflow running `ruff` and `pytest` on push/PR.
- Cross-platform Quick Start (PowerShell + bash) in the README.

### Changed
- Reorganized data into the lifecycle layout (`data/raw`, `data/interim`,
  `data/processed/graphs`, `data/experiments`).
- Refactored the notebook into importable modules under
  `pipe_network_completion/`. The original notebook is retained verbatim
  under `legacy/` for traceability.
- `scripts/create_env.ps1` now installs the package in editable mode.

### Documentation
- Added [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md),
  [docs/TRACEABILITY.md](docs/TRACEABILITY.md), and
  [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md).

[Unreleased]: https://github.com/Yuxi0048/PipeNetworkCompletion/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Yuxi0048/PipeNetworkCompletion/releases/tag/v1.0.0
