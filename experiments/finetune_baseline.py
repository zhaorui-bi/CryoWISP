#!/usr/bin/env python
"""Few-shot fine-tuning baseline on the same experience data a weight program sees.

Fine-tunes the base ResNet-18 on experience.npz (x_new + x_old), evaluates on selection.npz and final.npz
with the same scoring as the engine (accuracy per condition), so the numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def arrays(npz: Path) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(npz)
    x = np.concatenate([z["x_new"], z["x_old"]]).astype(np.float32) / 255.0
    y = np.concatenate([z["y_new"], z["y_old"]]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def cond_accuracy(model: nn.Module, npz: Path) -> dict:
    z = np.load(npz)
    model.eval()
    out = {}
    for role in ("new", "old"):
        x = torch.from_numpy(z[f"x_{role}"].astype(np.float32) / 255.0)
        y = torch.from_numpy(z[f"y_{role}"].astype(np.int64))
        correct = total = 0
        for i in range(0, len(x), 256):
            pred = model(normalize(x[i:i + 256])).argmax(1)
            correct += int((pred == y[i:i + 256]).sum())
            total += len(pred)
        out[role] = {"accuracy": correct / total, "n": total}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--experience", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model = resnet18(num_classes=2)
    model.load_state_dict(torch.load(args.base_model, map_location="cpu", weights_only=True))

    x, y = arrays(args.experience)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_sel = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x))
        losses = []
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            idx = perm[i:i + args.batch_size]
            logits = model(normalize(x[idx]))
            loss = nn.functional.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        sel = cond_accuracy(model, args.selection)
        net = sel["new"]["accuracy"] + sel["old"]["accuracy"]
        history.append({"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)),
                        "selection": sel})
        print(f"epoch {epoch}: loss {sum(losses) / max(1, len(losses)):.4f} "
              f"sel new {sel['new']['accuracy']:.4f} old {sel['old']['accuracy']:.4f}", flush=True)
        if net >= best_sel:
            best_sel = net
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    result = {"epochs": args.epochs, "lr": args.lr, "seed": args.seed,
              "best_selection_net": best_sel, "history": history,
              "selection": cond_accuracy(model, args.selection),
              "final": cond_accuracy(model, args.final)}
    print("selection", json.dumps(result["selection"]))
    print("final", json.dumps(result["final"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.float() for k, v in model.state_dict().items()}, args.out.with_suffix(".pt"))
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
