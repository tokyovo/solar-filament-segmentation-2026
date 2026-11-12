"""Write the release weights: convolution weights in float16, everything else in float32.

On the GPU, inference runs under float16 autocast, which converts every convolution weight
to float16 anyway. Storing those weights in float16 therefore gives the same masks on the
GPU and about halves the size. BatchNorm parameters and statistics stay in float32, because
autocast keeps BatchNorm in float32.

Each model folder keeps its layout (``config.json`` + ``model.pt``), so ``src.predict``
loads the release folders like the training folders::

    python -m scripts.export_weights --out release/weights
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from src import refiner, train

MODELS = Path("data/models")
FIRST_STAGE = [f"unet_r34_f{f}" for f in range(5)] + [f"unet_r34_r2048_f{f}" for f in range(5)]
REFINERS = [f"refiner_v2_f{f}" for f in range(5)]
FILTER = "filter_v1"  # LightGBM text model: copied unchanged


def half_conv_weights(model: torch.nn.Module) -> dict:
    """State dict with the weights and biases of Conv2d layers in float16."""
    state = model.state_dict()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            for p in ("weight", "bias"):
                key = f"{name}.{p}" if name else p
                if key in state and state[key] is not None:
                    state[key] = state[key].half()
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("release/weights"))
    args = parser.parse_args()
    total = 0
    for name in FIRST_STAGE + REFINERS:
        src = MODELS / name
        if name.startswith("refiner"):
            model = refiner.build_model(weights=None)
        else:
            model = train.build_model(json.loads((src / "config.json").read_text())["encoder"], weights=None)
        model.load_state_dict(torch.load(src / "model.pt", map_location="cpu"))
        dst = args.out / name
        dst.mkdir(parents=True, exist_ok=True)
        torch.save(half_conv_weights(model), dst / "model.pt")
        shutil.copy(src / "config.json", dst / "config.json")
        size = (dst / "model.pt").stat().st_size
        total += size
        print(f"{name}: {size / 2**20:.1f} MiB")
    shutil.copytree(MODELS / FILTER, args.out / FILTER, dirs_exist_ok=True)
    total += sum(f.stat().st_size for f in (args.out / FILTER).iterdir())
    print(f"{FILTER}: copied")
    print(f"total {total / 2**20:.0f} MiB in {args.out}")


if __name__ == "__main__":
    main()
