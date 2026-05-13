"""Download and extract release-asset archives for replication.

Use this on a slim clone (or after `git clone` if the maintainer publishes
artifacts via GitHub release assets instead of tracking them in git):

    python scripts/download_assets.py --version v1.0.0

Requires the GitHub CLI (`gh`); if it is not installed, the script prints
the equivalent `curl` command so it remains usable in offline setups.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "Yuxi0048/PipeNetworkCompletion"


def have_gh() -> bool:
    return shutil.which("gh") is not None


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(archive: Path, sidecar: Path) -> None:
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip().lower()
    actual = sha256_of(archive).lower()
    if expected != actual:
        raise SystemExit(
            f"SHA-256 mismatch for {archive.name}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
    print(f"SHA-256 verified: {archive.name}")


def _strip_archive_root(name: str) -> str | None:
    parts = Path(name).parts
    if not parts:
        return None
    if parts[0].startswith("pipe-network-artifacts-"):
        if len(parts) == 1:
            return None
        return str(Path(*parts[1:]))
    return name


def _safe_target(target_root: Path, relative_name: str) -> Path:
    target = (target_root / relative_name).resolve()
    root = target_root.resolve()
    if root != target and root not in target.parents:
        raise SystemExit(f"Refusing to extract outside target root: {relative_name}")
    return target


def extract_tarball(tarball: Path, target_root: Path) -> None:
    with tarfile.open(tarball, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            relative_name = _strip_archive_root(member.name)
            if relative_name is None:
                continue
            _safe_target(target_root, relative_name)
            member.name = relative_name
            tar.extract(member, path=target_root)


def extract_zip(archive_path: Path, target_root: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member_name in archive.namelist():
            relative_name = _strip_archive_root(member_name)
            if relative_name is None or member_name.endswith("/"):
                continue
            target = _safe_target(target_root, relative_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member_name) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)


def extract(archive_path: Path, target_root: Path) -> None:
    print(f"Extracting {archive_path.name} -> {target_root}")
    if archive_path.suffix == ".zip":
        extract_zip(archive_path, target_root)
    elif archive_path.name.endswith(".tar.gz"):
        extract_tarball(archive_path, target_root)
    else:
        raise SystemExit(f"Unsupported archive format: {archive_path.name}")


def gh_download(version: str, repo: str, target_dir: Path, pattern: str) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gh",
        "release",
        "download",
        version,
        "--repo",
        repo,
        "--pattern",
        pattern,
        "--dir",
        str(target_dir),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the release-asset archive for a tagged version."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release tag to fetch, e.g. v1.0.0.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub repository slug.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=REPO_ROOT / "release_assets",
        help="Directory for downloaded archive + sidecar.",
    )
    parser.add_argument(
        "--extract-to",
        type=Path,
        default=REPO_ROOT,
        help="Directory the archive contents land in (relative paths preserved).",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download only; do not extract.",
    )
    parser.add_argument(
        "--format",
        choices=["zip", "tar.gz"],
        default="zip",
        help="Release asset format to download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extension = "zip" if args.format == "zip" else "tar.gz"
    archive_name = f"pipe-network-artifacts-{args.version}.{extension}"
    archive_path = args.download_dir / archive_name
    sidecar_path = archive_path.with_suffix(archive_path.suffix + ".sha256")

    if not have_gh():
        url = (
            f"https://github.com/{args.repo}/releases/download/"
            f"{args.version}/{archive_name}"
        )
        print(
            "gh CLI not found. Install from https://cli.github.com/, "
            "or fetch manually:\n"
            f"  curl -L -o {archive_path} {url}\n"
            f"  curl -L -o {sidecar_path} {url}.sha256"
        )
        return 1

    gh_download(args.version, args.repo, args.download_dir, archive_name)
    gh_download(args.version, args.repo, args.download_dir, archive_name + ".sha256")

    verify_checksum(archive_path, sidecar_path)

    if not args.skip_extract:
        extract(archive_path, args.extract_to)
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
