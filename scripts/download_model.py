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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

import yaml


class DownloadError(RuntimeError):
    """A model could not be downloaded or validated."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(repository: str, filename: str, revision: str = "main") -> str:
    if not repository or repository.startswith(("/", ".")) or repository.count("/") != 1:
        raise DownloadError("repository must have the form owner/name")
    if not filename or Path(filename).name != filename:
        raise DownloadError("filename must be a single file name")
    if not revision or "/" in revision or revision.startswith("."):
        raise DownloadError("revision must be a branch, tag, or commit without slashes")
    return f"https://huggingface.co/{repository}/resolve/{urllib.parse.quote(revision)}/{urllib.parse.quote(filename)}?download=true"


def metadata_for(path: Path, repository: str, url: str, filename: str, *, logical_id: str | None = None,
                 quantization: str | None = None, base_model: str | None = None,
                 parameter_count: int | None = None, revision: str = "main") -> dict[str, object]:
    size = path.stat().st_size
    if size <= 0:
        raise DownloadError(f"rejecting empty model file: {path}")
    return {"logical_model_id": logical_id or Path(filename).stem,
            "quantization": quantization, "source_repository": repository,
            "source_filename": filename, "local_filename": path.name, "source_url": url,
            "source_revision": revision, "base_model_name": base_model,
            "parameter_count": parameter_count, "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": filename, "byte_size": size, "sha256": sha256_file(path)}


def _write_metadata(path: Path, metadata: dict[str, object]) -> Path:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def download_model(repository: str, filename: str, models_dir: Path = Path("models"),
                   opener: Callable[..., BinaryIO] = urllib.request.urlopen, *, logical_id: str | None = None,
                   quantization: str | None = None, base_model: str | None = None,
                   parameter_count: int | None = None, revision: str = "main",
                   expected_sha256: str | None = None, expected_byte_size: int | None = None) -> tuple[Path, dict[str, object]]:
    url = source_url(repository, filename, revision)
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / filename
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists():
        metadata = metadata_for(destination, repository, url, filename, logical_id=logical_id,
                                quantization=quantization, base_model=base_model,
                                parameter_count=parameter_count, revision=revision)
        if metadata_path.exists():
            try:
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DownloadError(f"cannot read existing metadata: {exc}") from exc
            identity = ("logical_model_id", "quantization", "source_repository", "source_filename",
                        "source_revision", "base_model_name", "parameter_count", "byte_size", "sha256")
            if any(key in previous and previous.get(key) != metadata.get(key) for key in identity):
                raise DownloadError("existing model does not match its metadata; remove it and retry")
        if expected_sha256 and metadata["sha256"] != expected_sha256.lower():
            raise DownloadError(f"checksum mismatch for {filename}")
        if expected_byte_size and metadata["byte_size"] != expected_byte_size:
            raise DownloadError(f"byte-size mismatch for {filename}")
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
    metadata = metadata_for(destination, repository, url, filename, logical_id=logical_id,
                            quantization=quantization, base_model=base_model,
                            parameter_count=parameter_count, revision=revision)
    if expected_sha256 and metadata["sha256"] != expected_sha256.lower():
        destination.unlink()
        raise DownloadError(f"checksum mismatch for {filename}")
    if expected_byte_size and metadata["byte_size"] != expected_byte_size:
        destination.unlink()
        raise DownloadError(f"byte-size mismatch for {filename}")
    _write_metadata(destination, metadata)
    return destination, metadata


def load_model_config(path: Path) -> dict[str, dict[str, object]]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DownloadError(f"cannot read model configuration: {exc}") from exc
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        raise DownloadError("model configuration requires a models mapping")
    required_quantizations = {"f16": "F16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M"}
    if set(models) != set(required_quantizations):
        raise DownloadError("models must contain exactly f16, q8_0, and q4_k_m")
    ids, compatibility = set(), set()
    for key, item in models.items():
        if not isinstance(item, dict):
            raise DownloadError(f"models.{key} must be a mapping")
        required = {"logical_id", "quantization", "source_repository", "source_filename",
                    "local_filename", "base_model_name", "parameter_count", "source_revision"}
        missing = required - set(item)
        if missing:
            raise DownloadError(f"models.{key} missing: {', '.join(sorted(missing))}")
        if item["logical_id"] in ids:
            raise DownloadError(f"duplicate model ID: {item['logical_id']}")
        ids.add(item["logical_id"])
        if str(item["quantization"]).upper() != required_quantizations[key]:
            raise DownloadError(f"models.{key} has wrong quantization")
        compatibility.add((item["source_repository"], item["base_model_name"], item["parameter_count"], item["source_revision"]))
    if len(compatibility) != 1:
        raise DownloadError("models are not compatible: repository, base model, parameter count, or revision differs")
    return models


def download_config(path: Path, models_dir: Path, opener: Callable[..., BinaryIO] = urllib.request.urlopen) -> list[tuple[Path, dict[str, object]]]:
    results = []
    for item in load_model_config(path).values():
        results.append(download_model(str(item["source_repository"]), str(item["source_filename"]), models_dir, opener,
            logical_id=str(item["logical_id"]), quantization=str(item["quantization"]),
            base_model=str(item["base_model_name"]), parameter_count=int(item["parameter_count"]),
            revision=str(item["source_revision"]), expected_sha256=item.get("sha256"),
            expected_byte_size=item.get("byte_size")))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", help="Hugging Face repository, for example owner/name")
    parser.add_argument("filename", nargs="?", help="exact GGUF filename")
    parser.add_argument("--config", type=Path, help="download all artifacts in a quantization model configuration")
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    args = parser.parse_args()
    try:
        if args.config:
            if args.repository or args.filename:
                raise DownloadError("do not combine positional arguments with --config")
            results = download_config(args.config, args.models_dir)
            for path, metadata in results:
                print(f"model: {path}\nbytes: {metadata['byte_size']}\nsha256: {metadata['sha256']}")
            return 0
        if not args.repository or not args.filename:
            raise DownloadError("repository and filename are required without --config")
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
