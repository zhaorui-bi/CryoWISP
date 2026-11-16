#!/usr/bin/env python
"""Evaluate a plain FP32 ResNet-18 state_dict (2-class) on crop role folders or engine npz splits.

Usage:
  eval_model.py --model model.pt --folders dir1 dir2 ...
  eval_model.py --model model.pt --npz final.npz --condition new|old|both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(path: Path) -> torch.nn.Module:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet18(num_classes=2)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def forward_labels(model: torch.nn.Module, x: np.ndarray, batch: int = 256) -> np.ndarray:
    preds = []
    for i in range(0, len(x), batch):
        t = torch.from_numpy(x[i : i + batch].astype(np.float32) / 255.0)
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        preds.append(model(t).argmax(1).numpy())
    return np.concatenate(preds)


def score(model: torch.nn.Module, x: np.ndarray, y: np.ndarray, tag: str) -> dict:
    pred = forward_labels(model, x)
    acc = float((pred == y).mean())
    for cls, name in ((0, "background"), (1, "particle")):
        m = y == cls
        if m.any():
            print(f"  {tag} {name}: {float((pred[m] == cls).mean()):.4f} (n={int(m.sum())})")
    print(f"  {tag} accuracy: {acc:.4f} (n={len(y)})")
    return {"acc": acc, "n": int(len(y))}


def load_npz_arrays(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in ("x_new", "y_new", "x_old", "y_old")}


def images_from_folder(root: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for cls, name in ((0, "background"), (1, "particle")):
        for f in sorted((root / name).glob("*.png")):
            xs.append(np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(f).convert("RGB")))
            ys.append(cls)
    return np.stack(xs).transpose(0, 3, 1, 2), np.array(ys, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--folders", type=Path, nargs="*", default=[])
    ap.add_argument("--npz", type=Path, nargs="*", default=[])
    args = ap.parse_args()
    model = load_model(args.model)
    results = {}
    for folder in args.folders:
        x, y = images_from_folder(folder)
        print(folder.name, flush=True)
        results[folder.name] = score(model, x, y, folder.name)
    for path in args.npz:
        arrays = load_npz_arrays(path)
        print(path.name, flush=True)
        results[path.name] = {}
        results[path.name]["new"] = score(model, arrays["x_new"], arrays["y_new"], "new")
        results[path.name]["old"] = score(model, arrays["x_old"], arrays["y_old"], "old")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
