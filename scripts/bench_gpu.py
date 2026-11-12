"""Measure the peak GPU memory and speed of training steps for several crop and batch sizes.

Runs 5 AMP training steps on random input per configuration::

    python -m scripts.bench_gpu --configs 512x8 768x4 768x6 1024x2 1024x4
"""

import argparse
import time

import torch

from src.train import build_model, loss_fn

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", default="resnet34")
parser.add_argument("--configs", nargs="+", default=["512x8", "768x4", "1024x2"])
args = parser.parse_args()

for config in args.configs:
    crop, batch = map(int, config.split("x"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(args.encoder, weights=None).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler()
    x = torch.rand(batch, 1, crop, crop, device="cuda")
    y = (torch.rand(batch, 1, crop, crop, device="cuda") > 0.99).float()
    try:
        for step in range(5):
            if step == 1:
                torch.cuda.synchronize()
                start = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        pixels_per_s = 4 * batch * crop * crop / (time.time() - start)
        print(f"{args.encoder} crop {crop} batch {batch}: peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
              f"{pixels_per_s / 1e6:.1f} Mpx/s", flush=True)
    except torch.OutOfMemoryError:
        print(f"{args.encoder} crop {crop} batch {batch}: OOM", flush=True)
    del model, optimizer, x, y
