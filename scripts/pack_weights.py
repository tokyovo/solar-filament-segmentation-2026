"""Pack ``release/weights/`` into one archive for the release asset, and print its SHA-256.

Put the printed SHA-256 into ``notebooks/demo.ipynb`` (``WEIGHTS_SHA256``)::

    python -m scripts.pack_weights
"""

import hashlib
import tarfile
from pathlib import Path

WEIGHTS = Path("release/weights")
OUT = Path("release/filament-seg-weights.tar")

with tarfile.open(OUT, "w") as tar:
    for folder in sorted(p for p in WEIGHTS.iterdir() if p.is_dir()):
        for name in ("config.json", "model.pt", "model.txt"):
            if (folder / name).exists():
                tar.add(folder / name, arcname=f"{folder.name}/{name}")
print(OUT, f"{OUT.stat().st_size / 2**20:.0f} MiB")
print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())
