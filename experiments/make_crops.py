#!/usr/bin/env python
"""Extract particle/background crops from CryoPPP_Lite micrographs for WeightCode.

Reads <lite_root>/<empiar_id>/micrographs/*.jpg and ground_truth/particle_coordinates/*.csv
(false_positives/*.csv used only to keep negative crops away from expert-marked spots),
writes 224x224 RGB PNG crops into role folders under --out. All splits are at the
micrograph level, so no micrograph contributes crops to two roles.

Old datasets (default 10005 10291 10081): base_train 70% of micrographs; the held-out
30% is quartered into old_exp / old_sel / old_fin (engine's old condition) and old_test
(our own evaluation of the base model). Crops of all old datasets are mixed per role.

New dataset (default 10947): micrograph blocks for new_exp / new_sel / new_fin (engine)
and new_test (our evaluation).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

OLD_ROLES = ("old_exp", "old_sel", "old_fin", "old_test")


def read_coords(csv_path: Path) -> list[tuple[float, float, float]]:
    rows = []
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append((float(row["X-Coordinate"]), float(row["Y-Coordinate"]),
                         float(row.get("Diameter") or 0)))
    return rows


def clamp_center(x: float, y: float, w: int, h: int, box: int) -> tuple[int, int]:
    cx = int(min(max(x, box / 2), w - box / 2))
    cy = int(min(max(y, box / 2), h - box / 2))
    return cx, cy


def save_crop(im: np.ndarray, cx: int, cy: int, box: int, out: Path) -> None:
    half = box // 2
    patch = im[cy - half:cy - half + box, cx - half:cx - half + box]
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patch).resize((224, 224), Image.BILINEAR).convert("RGB").save(out)


def process_micrograph(ds_dir: Path, mgr: Path, coords_csv: Path, diameter: float,
                       out_dir: Path, tag: str, n_pos: int, n_neg: int,
                       rng: random.Random) -> dict[str, int]:
    im = np.asarray(Image.open(mgr))
    h, w = im.shape[:2]
    box = max(160, min(int(np.ceil(1.5 * diameter)), min(h, w)))
    fp_csv = ds_dir / "ground_truth" / "false_positives" / f"{mgr.stem}_false_positives.csv"
    pts = read_coords(coords_csv)
    avoid = np.array([(p[0], p[1]) for p in pts], dtype=float)
    if fp_csv.exists():
        avoid = np.vstack([avoid, np.array([(p[0], p[1]) for p in read_coords(fp_csv)])])
    keep = box / 2 + diameter / 2

    rng.shuffle(pts)
    saved = 0
    for x, y, _ in pts[: 4 * n_pos]:
        if saved >= n_pos:
            break
        cx, cy = clamp_center(x, y, w, h, box)
        save_crop(im, cx, cy, box, out_dir / "particle" / f"{tag}_p{saved:05d}.png")
        saved += 1

    neg = 0
    tries = 0
    while neg < n_neg and tries < 300 * n_neg:
        tries += 1
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        if len(avoid) and (((avoid[:, 0] - x) ** 2 + (avoid[:, 1] - y) ** 2 < keep * keep).any()):
            continue
        cx, cy = clamp_center(x, y, w, h, box)
        save_crop(im, cx, cy, box, out_dir / "background" / f"{tag}_n{neg:05d}.png")
        neg += 1
    return {"pos": saved, "neg": neg}


def pair_files(ds_dir: Path) -> list[tuple[Path, Path]]:
    mgrs = sorted((ds_dir / "micrographs").glob("*.jpg"))
    return [(m, ds_dir / "ground_truth" / "particle_coordinates" / f"{m.stem}.csv")
            for m in mgrs
            if (ds_dir / "ground_truth" / "particle_coordinates" / f"{m.stem}.csv").exists()]


def dataset_diameter(pairs: list[tuple[Path, Path]]) -> float:
    if not pairs:
        raise SystemExit("no micrograph/csv pairs found for a requested dataset")
    diams = [d for _, _, d in read_coords(pairs[0][1]) if d > 0]
    for _, c in pairs[1:3]:
        diams += [d for _, _, d in read_coords(c) if d > 0]
    return float(np.median(diams)) if diams else 120.0


def crops_for_role(ds_dir: Path, plist: list[tuple[Path, Path]], diameter: float, role: str,
                   per_class: int, out: Path, rng: random.Random, summary: dict) -> None:
    per = int(np.ceil(per_class / max(1, len(plist))))
    for m, c in plist:
        r = process_micrograph(ds_dir, m, c, diameter, out / role, f"{ds_dir.name}_{m.stem}",
                               per, per, rng)
        key = f"{ds_dir.name}:{role}"
        a, b = summary.get(key, (0, 0))
        summary[key] = (a + r["pos"], b + r["neg"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lite-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old-ids", nargs="+", default=["10005", "10291", "10081"])
    ap.add_argument("--new-id", default="10947")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train-per-ds", type=int, default=2000, help="crops per class per old dataset in base_train")
    ap.add_argument("--oldxsf-per-ds", type=int, default=200, help="crops per class per old dataset in old_exp/old_sel/old_fin")
    ap.add_argument("--oldtest-per-ds", type=int, default=300, help="crops per class per old dataset in old_test")
    ap.add_argument("--new-per-class", type=int, nargs=4, default=[32, 150, 200, 1000],
                    metavar=("EXP", "SEL", "FIN", "TEST"), help="crops per class for new_exp/new_sel/new_fin/new_test")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = args.out
    summary: dict = {}

    for ds in args.old_ids:
        ds_dir = args.lite_root / ds
        pairs = pair_files(ds_dir)
        rng.shuffle(pairs)
        diam = dataset_diameter(pairs)
        n_train = int(len(pairs) * 0.70)
        train, held = pairs[:n_train], pairs[n_train:]
        q = max(1, len(held) // 4)
        quarters = [held[i * q:(i + 1) * q] if i < 3 else held[3 * q:] for i in range(4)]

        crops_for_role(ds_dir, train, diam, "base_train", args.train_per_ds, out, rng, summary)
        for role, plist in zip(OLD_ROLES, quarters):
            crops_for_role(ds_dir, plist, diam, role,
                           args.oldxsf_per_ds if role != "old_test" else args.oldtest_per_ds,
                           out, rng, summary)
        print(ds, {"diam": diam, "micro": len(pairs), "train": n_train, "held": len(held)}, flush=True)

    ds_dir = args.lite_root / args.new_id
    pairs = pair_files(ds_dir)
    rng.shuffle(pairs)
    diam = dataset_diameter(pairs)
    n_micro = {"new_exp": 4, "new_sel": 8, "new_fin": 10, "new_test": 30}
    per_class = dict(zip(("new_exp", "new_sel", "new_fin", "new_test"), args.new_per_class))
    used = 0
    for role in ("new_exp", "new_sel", "new_fin", "new_test"):
        plist = pairs[used:used + n_micro[role]]
        used += n_micro[role]
        crops_for_role(ds_dir, plist, diam, role, per_class[role], out, rng, summary)
    print(args.new_id, {"diam": diam, "micro": len(pairs)}, flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        {k: list(v) if isinstance(v, tuple) else v for k, v in summary.items()}, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
