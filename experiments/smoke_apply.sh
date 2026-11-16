#!/usr/bin/env bash
# Engine end-to-end smoke test on a synthetic task: build a random-init ResNet-18 and random
# crops, then run import-model -> prepare-data x3 -> prepare-task -> pack-program -> apply.
# Must finish with a report.json under experiments/runs/smoke/reuse/final.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
OUT=$REPO/experiments/runs/smoke
rm -rf "$OUT"
mkdir -p "$OUT"

.venv/bin/python -I - <<'PY'
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18

out = Path("experiments/runs/smoke")
m = resnet18(num_classes=2)
sd = {k: (v.float() if v.is_floating_point() else v.detach().clone()) for k, v in m.state_dict().items()}
torch.save(sd, out / "model.pt")
(out / "labels.json").write_text(json.dumps(["background", "particle"]))
rng = np.random.default_rng(0)
for role in ("exp", "sel", "fin"):
    for cond in ("new", "old"):
        for cls in ("background", "particle"):
            d = out / "crops" / role / cond / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(8):
                Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(d / f"{i:03d}.png")
print("synthetic assets written")
PY

.venv/bin/cryowisp import-model --checkpoint $OUT/model.pt --labels $OUT/labels.json --out $OUT/model
for r in exp sel fin; do
  .venv/bin/cryowisp prepare-data --model $OUT/model --new $OUT/crops/$r/new --old $OUT/crops/$r/old --out $OUT/$r.npz > /dev/null
done
.venv/bin/cryowisp prepare-task --model $OUT/model --experience $OUT/exp.npz --selection $OUT/sel.npz \
  --final $OUT/fin.npz --goal 'synthetic smoke test' --retention-drop 0.01 --min-new-gain 0.01 --out $OUT/task
.venv/bin/cryowisp pack-program --program experiments/engine/programs/resnet18-ridge-head/program.py \
  --model $OUT/model --out $OUT/program
SHA=$(sha256sum experiments/engine/programs/resnet18-ridge-head/program.py | cut -d' ' -f1)
.venv/bin/cryowisp apply --task $OUT/task --program $OUT/program --program-sha256 $SHA \
  --wall-s 300 --out $OUT/reuse > /dev/null
.venv/bin/python -I -c "
import json
from pathlib import Path
r = json.loads(Path('$OUT/reuse/final/report.json').read_text())
print('smoke report status:', r['status'], '| isolation:', r['program_isolation'])
assert r['status'] in ('accepted', 'base-kept', 'final-rejected')
print('SMOKE OK')
"
