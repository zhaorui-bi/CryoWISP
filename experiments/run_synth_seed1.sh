#!/usr/bin/env bash
# Real editor session on seed 1's cryoEM task (Gadi login node: needs the GLM endpoint).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
SEED=${1:-1}
OUT=$REPO/experiments/runs/seed$SEED/synth
rm -rf "$OUT"
.venv/bin/cryowsp --threads 2 synthesize \
  --task $REPO/experiments/runs/seed$SEED/task --out "$OUT" \
  --sandbox-python .venv-sandbox/bin/python \
  --editor-isolation local \
  --editor-model 'glm-5.3[1m]' \
  --base-url-file "$HOME/.config/cryowsp/glm_base_url" \
  --auth-token-file "$HOME/.config/cryowsp/glm_auth_token" \
  --checks 8 --cells 100 --wall-s 5400 --program-wall-s 300
