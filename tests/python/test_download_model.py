from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from scripts import download_model as downloader


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200) -> None:
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}


def test_checksum_calculation(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"model bytes")
    assert downloader.sha256_file(path) == hashlib.sha256(b"model bytes").hexdigest()


def test_existing_file_handling_does_not_open_http(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"existing")

    def fail(*args: object, **kwargs: object) -> Response:
        raise AssertionError("HTTP must not be used for an existing model")

    result, metadata = downloader.download_model("owner/repo", path.name, tmp_path, fail)
    assert result == path
    assert metadata["byte_size"] == 8


def test_empty_existing_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "model.gguf").touch()
    with pytest.raises(downloader.DownloadError, match="empty"):
        downloader.download_model("owner/repo", "model.gguf", tmp_path)


def test_metadata_generation(tmp_path: Path) -> None:
    path, expected = downloader.download_model(
        "owner/repo", "model.gguf", tmp_path, lambda *args, **kwargs: Response(b"GGUF data")
    )
    metadata = json.loads(path.with_suffix(".gguf.metadata.json").read_text(encoding="utf-8"))
    assert metadata == expected
    assert metadata["source_repository"] == "owner/repo"
    assert metadata["source_url"].endswith("/owner/repo/resolve/main/model.gguf?download=true")


def test_http_failure_has_clear_error(tmp_path: Path) -> None:
    def fail(*args: object, **kwargs: object) -> Response:
        raise urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {}, None)

    with pytest.raises(downloader.DownloadError, match="download failed.*404"):
        downloader.download_model("owner/repo", "model.gguf", tmp_path, fail)
