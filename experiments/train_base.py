#!/usr/bin/env python
"""Train a binary particle/background ResNet-18 on CryoPPP crops (CryoWSP base network).

Saves a plain FP32 torchvision state_dict (num_classes=2) plus labels.json and a metrics
json. Class order is ImageFolder's alphabetical order: background=0, particle=1.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropDataset(Dataset):
    def __init__(self, root: Path, augment: bool, seed: int = 0, keep: set[Path] | None = None):
        self.files = sorted(root.glob("*/*.png")) + sorted(root.glob("*/*.jpg"))
        if keep is not None:
            self.files = [f for f in self.files if f in keep]
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = Image.open(self.files[i]).convert("RGB")
        if self.augment:
            if self.rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rng.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        x = (x - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        y = 0 if self.files[i].parent.name == "background" else 1
        return x, y


@torch.no_grad()
def evaluate(model: nn.Module, root: Path, batch: int = 256) -> dict:
    ds = CropDataset(root, augment=False)
    if len(ds) == 0:
        return {}
    dl = DataLoader(ds, batch_size=batch, num_workers=4)
    model.eval()
    correct = total = 0
    per_class = {0: [0, 0], 1: [0, 0]}
    per_ds: dict[str, list[int]] = {}
    for x, y, paths in _iterate_with_paths(dl, ds):
        pred = model(x).argmax(1)
        for p, t, path in zip(pred.tolist(), y.tolist(), paths):
            per_class[t][1] += 1
            per_class[t][0] += int(p == t)
            key = Path(path).name.split("_")[0]
            per_ds.setdefault(key, [0, 0])
            per_ds[key][1] += 1
            per_ds[key][0] += int(p == t)
        correct += int((pred == y).sum())
        total += len(y)
    return {
        "acc": correct / total,
        "n": total,
        "background_acc": per_class[0][0] / max(1, per_class[0][1]),
        "particle_acc": per_class[1][0] / max(1, per_class[1][1]),
        "per_dataset": {k: {"acc": c / n, "n": n} for k, (c, n) in sorted(per_ds.items())},
    }


def _iterate_with_paths(dl: DataLoader, ds: CropDataset):
    indices = []
    idx = 0
    for x, y in dl:
        paths = [ds.files[idx + j] for j in range(len(y))]
        idx += len(y)
        yield x, y, paths


def as_state_dict(model: nn.Module) -> dict:
    return {
        k: (v.float() if v.is_floating_point() else v.detach().clone()) for k, v in model.state_dict().items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crops-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--imagenet-init",
        type=Path,
        default=Path("/g/data/jz31/zhaorui/cryoppp_data/torch_cache/resnet18_imagenet.pt"),
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(int(torch.get_num_threads()))

    files = sorted((args.crops_dir / "base_train").glob("*/*.png"))
    rng = random.Random(args.seed * 1000 + 5)
    rng.shuffle(files)
    n_val = max(64, len(files) // 25)
    val_set = {f for f in files[:n_val]}
    train_set = {f for f in files[n_val:]}

    train_ds = CropDataset(args.crops_dir / "base_train", True, args.seed, keep=train_set)
    val_ds = CropDataset(args.crops_dir / "base_train", False, keep=val_set)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    model = resnet18(num_classes=2)
    if args.imagenet_init and args.imagenet_init.exists():
        sd = torch.load(args.imagenet_init, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        missing = model.load_state_dict(sd, strict=False)
        assert missing.missing_keys == ["fc.weight", "fc.bias"] and not missing.unexpected_keys, missing
        print(f"loaded ImageNet init ({len(sd)} tensors), fc reinitialized", flush=True)
    nn.init.normal_(model.fc.weight, std=0.01)
    nn.init.zeros_(model.fc.bias)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)
    crit = nn.CrossEntropyLoss()

    history = []
    best_val = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        running = count = 0
        for x, y in train_dl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            count += 1
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                correct += int((model(x).argmax(1) == y).sum())
                total += len(y)
        val_acc = correct / total
        history.append({"epoch": epoch, "train_loss": running / count, "val_acc": val_acc})
        print(f"epoch {epoch}: loss {running / count:.4f} val_acc {val_acc:.4f}", flush=True)
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(as_state_dict(model), args.out_dir / f"model_seed{args.seed}.pt")
    (args.out_dir / "labels.json").write_text(json.dumps(["background", "particle"]))

    metrics = {"seed": args.seed, "best_val_acc": best_val, "history": history, "eval": {}}
    for role in ("old_test", "new_test"):
        metrics["eval"][role] = evaluate(model, args.crops_dir / role)
        print(role, json.dumps(metrics["eval"][role]), flush=True)
    (args.out_dir / f"metrics_seed{args.seed}.json").write_text(json.dumps(metrics, indent=2))
    print("saved", args.out_dir / f"model_seed{args.seed}.pt", flush=True)


if __name__ == "__main__":
    main()
