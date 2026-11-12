"""Download and unpack the release weights, so the demo needs no manual step."""

import hashlib
import tarfile
import urllib.request
from pathlib import Path


def fetch_weights(url: str, sha256: str, dest: Path) -> None:
    """Download the weight archive from ``url``, check its SHA-256, and unpack it into ``dest``.

    The archive holds one folder per model (``config.json`` + ``model.pt``), as written by
    ``scripts/pack_weights.py``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "weights.tar"
    urllib.request.urlretrieve(url, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != sha256:
        archive.unlink()
        raise ValueError(f"weight archive sha256 {digest} does not match the expected {sha256}")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    archive.unlink()
