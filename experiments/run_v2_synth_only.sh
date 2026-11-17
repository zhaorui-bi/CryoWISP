#!/usr/bin/env bash
# Restart the CryoWSP v2 editor session only (engine tasks already built), with the raised
# local-kernel cell cap, then finalize if wall-killed.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd $REPO/CryoWSP
export PATH="$HOME/.local/bin:$PATH"
OUT=$REPO/experiments/runs/seed1_v2/synth
rm -rf "$OUT"
.venv/bin/cryowsp --threads 2 synthesize \
  --task $REPO/experiments/runs/seed1_v2/task --out "$OUT" \
  --sandbox-python .venv-sandbox/bin/python \
  --editor-isolation local \
  --editor-model 'glm-5.3[1m]' \
  --base-url-file "$HOME/.config/cryowsp/glm_base_url" \
  --auth-token-file "$HOME/.config/cryowsp/glm_auth_token" \
  --checks 6 --cells 120 --wall-s 7200 --program-wall-s 420 || true

if [ ! -f "$OUT/final/report.json" ] && [ -f "$OUT/freeze.json" ]; then
  echo "=== finalizing wall-killed session $(date) ==="
  .venv/bin/python -I - <<'PY'
from pathlib import Path
import json
from cryowsp.application.runtime import Application
app = Application(Path("$REPO/experiments/runs/seed1_v2/synth"))
report = app.finalize()
print(json.dumps({"status": report["status"],
                  "final_base": report["final_base"], "final_candidate": report["final_candidate"]}, indent=2))
app.close()
PY
fi
echo "=== V2 SYNTH RESTART DONE $(date) ==="
