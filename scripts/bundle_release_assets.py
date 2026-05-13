"""Bundle large artifacts into a release archive.

Writes `release_assets/pipe-network-artifacts-<version>.<ext>` plus a matching
`.sha256` sidecar so downstream `download_assets.py` runs can verify integrity.

Excluded from git via the standard build directory pattern; see
RELEASE.md for the full release-cut procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipe_network_completion.paths import (
    CHECKPOINT_DIR,
    EXPERIMENT_DATA_DIR,
    GRAPH_DATA_DIR,
    INTERIM_DATA_DIR,
    METRICS_DIR,
    RAW_DATA_DIR,
    SPLIT_SHAPEFILE_DIR,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "release_assets"

ASSET_GROUPS = {
    "checkpoints": CHECKPOINT_DIR,
    "experiments": EXPERIMENT_DATA_DIR,
    "graphs": GRAPH_DATA_DIR,
    "interim": INTERIM_DATA_DIR,
    "metrics": METRICS_DIR,
    "raw": RAW_DATA_DIR,
    "split_shapefiles": SPLIT_SHAPEFILE_DIR,
}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def collect_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"WARN: skipping missing path {root}")
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != ".gitkeep":
                files.append(path)
    return files


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tarball(
    files: list[Path],
    output_path: Path,
    arcname_root: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        for file_path in files:
            arcname = Path(arcname_root) / file_path.relative_to(REPO_ROOT)
            tar.add(file_path, arcname=str(arcname))


def build_zip(
    files: list[Path],
    output_path: Path,
    arcname_root: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            arcname = Path(arcname_root) / file_path.relative_to(REPO_ROOT)
            archive.write(file_path, arcname=str(arcname))


def write_sha256_sidecar(archive_path: Path) -> Path:
    sidecar = archive_path.with_suffix(archive_path.suffix + ".sha256")
    digest = sha256_of(archive_path)
    sidecar.write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle release assets into an archive + sha256 sidecar."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release tag, e.g. v1.0.0. Used in the output filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated archive.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=sorted(ASSET_GROUPS),
        default=[
            "checkpoints",
            "experiments",
            "graphs",
            "interim",
            "metrics",
            "split_shapefiles",
        ],
        help="Which artifact groups to include.",
    )
    parser.add_argument(
        "--format",
        choices=["zip", "tar.gz"],
        default="zip",
        help="Archive format for the generated release asset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.version.startswith("v"):
        print(f"WARN: version '{args.version}' does not start with 'v'.")

    roots = [ASSET_GROUPS[name] for name in args.include]
    files = collect_files(roots)
    if not files:
        print("No files matched the requested asset groups; nothing to bundle.")
        return 1

    output_dir = args.output_dir.resolve()
    extension = "zip" if args.format == "zip" else "tar.gz"
    output_name = f"pipe-network-artifacts-{args.version}.{extension}"
    output_path = output_dir / output_name
    arcname_root = f"pipe-network-artifacts-{args.version}"

    print(f"Bundling {len(files)} file(s) -> {_display_path(output_path)}")
    if args.format == "zip":
        build_zip(files, output_path, arcname_root)
    else:
        build_tarball(files, output_path, arcname_root)

    sidecar = write_sha256_sidecar(output_path)
    print(f"Wrote SHA-256 sidecar -> {_display_path(sidecar)}")
    print(f"Total bundle size: {output_path.stat().st_size / (1024 * 1024):.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
