"""Write the provenance sidecar ``<submission>.json`` before an upload.

Records the git commit, the dirty flag, the model config, the post-processing settings, the
sha256 of the CSV and ``lb_public: null`` (fill it in after Kaggle scores the file)::

    python -m scripts.write_sidecar submission/x.csv --model data/models/unet_r34_f0 \\
        --postprocess '{"threshold": 0.4, "min_area": 600}' --cv '{"fold0_pq": 0.4066}'
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=Path)
parser.add_argument("--model", type=Path, nargs="+", required=True)
parser.add_argument("--postprocess", type=json.loads, required=True)
parser.add_argument("--cv", type=json.loads, required=True)
parser.add_argument("--note", default="")
args = parser.parse_args()

git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
frame = pd.read_csv(args.csv)
sidecar = {
    "file": args.csv.name,
    "sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    "models": {str(m): json.loads((m / "config.json").read_text()) for m in args.model},
    "postprocess": args.postprocess,
    "cv": args.cv,
    "rows": len(frame),
    "images": int(frame["filament_id"].str.rsplit("_", n=1).str[0].nunique()),
    "note": args.note,
    "lb_public": None,
}
out = args.csv.with_suffix(".json")
out.write_text(json.dumps(sidecar, indent=2) + "\n")
print(out)
