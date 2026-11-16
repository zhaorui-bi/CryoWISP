#!/usr/bin/env python
"""Dose-response of a kept weight program: apply alpha-scaled deltas to the base model and
score each alpha on the engine final split and the external new_test crops.

A smooth monotone accuracy curve in alpha (new condition) with a flat old curve is evidence
that the weld is a well-conditioned, dose-controllable edit rather than a lucky jump.
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


@torch.no_grad()
def acc(model, x: np.ndarray, y: np.ndarray) -> float:
    correct = total = 0
    for i in range(0, len(x), 256):
        t = torch.from_numpy(x[i:i + 256].astype(np.float32) / 255.0)
        pred = model((t - IMAGENET_MEAN) / IMAGENET_STD).argmax(1).numpy()
        correct += int((pred == y[i:i + 256]).sum())
        total += len(pred)
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--attempt-dir", type=Path, required=True,
                    help="engine attempt dir holding child.npz (the kept program's edit)")
    ap.add_argument("--engine-model-dir", type=Path, required=True,
                    help="task model dir (params.npz) the program edited")
    ap.add_argument("--final-npz", type=Path, required=True)
    ap.add_argument("--new-test", type=Path, required=True, help="external new_test crops folder")
    ap.add_argument("--old-test", type=Path, required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base_sd = {k: v.clone() for k, v in torch.load(args.base_model, map_location="cpu",
                                                   weights_only=True).items()}
    child = np.load(args.attempt_dir / "child.npz")
    params = np.load(args.engine_model_dir / "params.npz")
    deltas = {k: child[k].astype(np.float32) - params[k].astype(np.float32)
              for k in ("layer4.1.bn2.weight", "layer4.1.bn2.bias", "fc.bias")}

    z = np.load(args.final_npz)
    from PIL import Image
    def folder_xy(root: Path):
        xs, ys = [], []
        for cls, name in ((0, "background"), (1, "particle")):
            for f in sorted((root / name).glob("*.png")):
                xs.append(np.asarray(Image.open(f).convert("RGB")).transpose(2, 0, 1))
                ys.append(cls)
        return np.stack(xs), np.array(ys)

    xt, yt = folder_xy(args.new_test)
    xo, yo = folder_xy(args.old_test)

    rows = []
    for a in args.alphas:
        model = resnet18(num_classes=2)
        sd = {k: v.clone() for k, v in base_sd.items()}
        for k, d in deltas.items():
            sd[k] = sd[k].float() + torch.from_numpy(d) * a
        model.load_state_dict(sd)
        model.eval()
        row = {
            "alpha": a,
            "final_new": acc(model, z["x_new"], z["y_new"]),
            "final_old": acc(model, z["x_old"], z["y_old"]),
            "external_new": acc(model, xt, yt),
            "external_old": acc(model, xo, yo),
        }
        rows.append(row)
        print(f"alpha {a:4.2f}: final new {row['final_new']:.4f} old {row['final_old']:.4f} | "
              f"ext new {row['external_new']:.4f} old {row['external_old']:.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
