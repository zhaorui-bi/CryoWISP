#!/usr/bin/env python
"""Paper baselines for CryoWISP, all on the identical protocol: train on experience.npz only,
select on selection.npz per epoch, score final.npz exactly once.

Modes:
  full          gradient fine-tune of every parameter
  linear_probe  train fc.weight/fc.bias only (1026 params, same count as the weight program)
  knob          train layer4.1.bn2.{weight,bias} + fc.bias only (the program's exact 1026 knobs)
  copied_delta  no training: add seed1's weight-program deltas to this seed's weights
  random_knob   no training: random deltas on the same three tensors, per-tensor Frobenius
                norm matched to the program's deltas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
KNOBS = ("layer4.1.bn2.weight", "layer4.1.bn2.bias", "fc.bias")
FC = ("fc.weight", "fc.bias")


def load_npz_xy(npz: Path) -> tuple[torch.Tensor, torch.Tensor]:
    z = np.load(npz)
    x = np.concatenate([z["x_new"], z["x_old"]]).astype(np.float32) / 255.0
    y = np.concatenate([z["y_new"], z["y_old"]]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


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
            pred = model((x[i : i + 256] - IMAGENET_MEAN) / IMAGENET_STD).argmax(1)
            correct += int((pred == y[i : i + 256]).sum())
            total += len(pred)
        out[role] = {"accuracy": correct / total, "n": total}
    return out


def params_from_engine_model(model_dir: Path) -> dict[str, np.ndarray]:
    z = np.load(model_dir / "params.npz")
    return {k: z[k] for k in z.files}


def program_deltas(attempt_dir: Path, engine_model_dir: Path) -> dict[str, np.ndarray]:
    child = np.load(attempt_dir / "child.npz")
    base = params_from_engine_model(engine_model_dir)
    return {
        k: child[k].astype(np.float64) - base[k].astype(np.float64)
        for k in ("layer4.1.bn2.weight", "layer4.1.bn2.bias", "fc.bias")
    }


def apply_deltas(model: nn.Module, deltas: dict[str, np.ndarray]) -> None:
    sd = model.state_dict()
    for k, d in deltas.items():
        sd[k] = sd[k].float() + torch.from_numpy(d.astype(np.float32))
    model.load_state_dict(sd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--mode", required=True, choices=("full", "linear_probe", "knob", "copied_delta", "random_knob")
    )
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--experience", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--class-weight-new",
        type=float,
        default=1.0,
        help="loss weight of new-condition samples (the program used ~10x)",
    )
    ap.add_argument(
        "--deltas-from",
        type=Path,
        default=None,
        help="copied_delta/random_knob: attempt dir holding child.npz (default: seed1 synth attempt)",
    )
    ap.add_argument(
        "--engine-model-dir",
        type=Path,
        default=Path("/g/data/jz31/zhaorui/weight_program/cryoexp/engine/seed1/task/model"),
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model = resnet18(num_classes=2)
    model.load_state_dict(torch.load(args.base_model, map_location="cpu", weights_only=True))

    if args.mode in ("copied_delta", "random_knob"):
        deltas = program_deltas(
            args.deltas_from
            or Path("/g/data/jz31/zhaorui/weight_program/cryoexp/engine/seed1/synth/attempts/0004"),
            args.engine_model_dir,
        )
        if args.mode == "random_knob":
            rng = np.random.default_rng(args.seed)

            def norm_matched(d):
                g = rng.normal(0, 1, d.shape)
                return g / np.linalg.norm(g) * np.linalg.norm(d)

            deltas = {k: norm_matched(d) for k, d in deltas.items()}
        apply_deltas(model, deltas)
        result = {
            "mode": args.mode,
            "seed": args.seed,
            "selection": cond_accuracy(model, args.selection),
            "final": cond_accuracy(model, args.final),
        }
        print(json.dumps(result["final"]))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        return

    train_pattern = None if args.mode == "full" else (FC if args.mode == "linear_probe" else KNOBS)
    for name, p in model.named_parameters():
        p.requires_grad_(train_pattern is None or name in train_pattern)

    x, y = load_npz_xy(args.experience)
    n_new = int(np.load(args.experience)["y_new"].shape[0])
    weights = torch.ones(len(y))
    weights[:n_new] = args.class_weight_new
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_net, best_state = -1.0, None
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x))
        losses = []
        for i in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            idx = perm[i : i + args.batch_size]
            per = nn.functional.cross_entropy(
                model((x[idx] - IMAGENET_MEAN) / IMAGENET_STD), y[idx], reduction="none"
            )
            loss = (per * weights[idx]).sum() / weights[idx].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        sched.step()
        sel = cond_accuracy(model, args.selection)
        net = sel["new"]["accuracy"] + sel["old"]["accuracy"]
        history.append({"epoch": epoch, "loss": sum(losses) / max(1, len(losses)), "selection": sel})
        print(
            f"epoch {epoch}: loss {sum(losses) / max(1, len(losses)):.4f} "
            f"sel new {sel['new']['accuracy']:.4f} old {sel['old']['accuracy']:.4f}",
            flush=True,
        )
        if net >= best_net:
            best_net = net
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    result = {
        "mode": args.mode,
        "seed": args.seed,
        "lr": args.lr,
        "epochs": args.epochs,
        "best_selection_net": best_net,
        "history": history,
        "selection": cond_accuracy(model, args.selection),
        "final": cond_accuracy(model, args.final),
    }
    print("final", json.dumps(result["final"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
