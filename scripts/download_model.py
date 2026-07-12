#!/usr/bin/env python3
"""Stream a Hugging Face model download into the ignored models directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import BinaryIO, Callable


class DownloadError(RuntimeError):
    """A model could not be downloaded or validated."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(repository: str, filename: str) -> str:
    if not repository or repository.startswith(("/", ".")) or repository.count("/") != 1:
        raise DownloadError("repository must have the form owner/name")
    if not filename or Path(filename).name != filename:
        raise DownloadError("filename must be a single file name")
    return f"https://huggingface.co/{repository}/resolve/main/{urllib.parse.quote(filename)}?download=true"


def metadata_for(path: Path, repository: str, url: str, filename: str) -> dict[str, object]:
    size = path.stat().st_size
    if size <= 0:
        raise DownloadError(f"rejecting empty model file: {path}")
    return {"source_repository": repository, "source_url": url, "filename": filename,
            "byte_size": size, "sha256": sha256_file(path)}


def _write_metadata(path: Path, metadata: dict[str, object]) -> Path:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def download_model(repository: str, filename: str, models_dir: Path = Path("models"),
                   opener: Callable[..., BinaryIO] = urllib.request.urlopen) -> tuple[Path, dict[str, object]]:
    url = source_url(repository, filename)
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / filename
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists():
        metadata = metadata_for(destination, repository, url, filename)
        if metadata_path.exists():
            try:
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DownloadError(f"cannot read existing metadata: {exc}") from exc
            if previous.get("byte_size") != metadata["byte_size"] or previous.get("sha256") != metadata["sha256"]:
                raise DownloadError("existing model does not match its metadata; remove it and retry")
        _write_metadata(destination, metadata)
        return destination, metadata

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    try:
        with opener(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            resumed = offset > 0 and status == 206
            if offset and not resumed:
                offset = 0
            content_length = response.headers.get("Content-Length")
            expected = offset + int(content_length) if content_length is not None else None
            mode = "ab" if resumed else "wb"
            with partial.open(mode) as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        raise DownloadError(f"download failed for {url}: {exc}") from exc
    actual = partial.stat().st_size if partial.exists() else 0
    if actual == 0:
        raise DownloadError("download returned an empty file")
    if expected is not None and actual != expected:
        raise DownloadError(f"incomplete download: expected {expected} bytes, received {actual}")
    partial.replace(destination)
    metadata = metadata_for(destination, repository, url, filename)
    _write_metadata(destination, metadata)
    return destination, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", help="Hugging Face repository, for example owner/name")
    parser.add_argument("filename", help="exact GGUF filename")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    args = parser.parse_args()
    try:
        path, metadata = download_model(args.repository, args.filename, args.models_dir)
    except DownloadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"model: {path}")
    print(f"bytes: {metadata['byte_size']}")
    print(f"sha256: {metadata['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
