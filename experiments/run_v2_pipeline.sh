#!/usr/bin/env bash
# CryoWISP v2 optimization round: richer new-condition experience (192 crops), leaner check budget.
# Waits for the crops regeneration, builds engine tasks (suffix _v2) for both seeds, runs the
# editor session on seed 1, and finalizes if the editor hit the wall without calling finish().
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd $REPO/CryoWISP
export PATH="$HOME/.local/bin:$PATH"

# 1. wait for the crops regeneration (make_crops prints "done" at the end)
until grep -q "^done" /tmp/crops_v2.log 2>/dev/null; do sleep 30; done
echo "=== crops v2 ready $(date) ==="

# 2. engine tasks for both seeds on the v2 experience
bash $REPO/experiments/run_engine_task.sh 1 '' _v2
bash $REPO/experiments/run_engine_task.sh 2 '' _v2
echo "=== engine tasks v2 ready $(date) ==="

# 3. editor session on seed 1 (v2): fewer checks, longer wall, roomier program budget
OUT=$REPO/experiments/engine/seed1_v2/synth
rm -rf "$OUT"
.venv/bin/cryowisp --threads 2 synthesize \
  --task $REPO/experiments/engine/seed1_v2/task --out "$OUT" \
  --sandbox-python .venv-sandbox/bin/python \
  --editor-isolation local \
  --editor-model 'glm-5.3[1m]' \
  --base-url-file "$HOME/.config/cryowisp/glm_base_url" \
  --auth-token-file "$HOME/.config/cryowisp/glm_auth_token" \
  --checks 6 --cells 120 --wall-s 7200 --program-wall-s 420 || true

# 4. finalize if the editor was wall-killed (freeze already written, final not scored)
if [ ! -f "$OUT/final/report.json" ] && [ -f "$OUT/freeze.json" ]; then
  echo "=== finalizing wall-killed session $(date) ==="
  .venv/bin/python -I - <<'PY'
from pathlib import Path
import json
from cryowisp.application.runtime import Application
app = Application(Path("$REPO/experiments/engine/seed1_v2/synth"))
report = app.finalize()
print(json.dumps({"status": report["status"],
                  "final_base": report["final_base"], "final_candidate": report["final_candidate"]}, indent=2))
app.close()
PY
fi
echo "=== V2 PIPELINE DONE $(date) ==="
