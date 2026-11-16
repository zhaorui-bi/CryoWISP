#!/usr/bin/env bash
# Real editor session on seed 1's cryoEM task (Gadi login node: needs the GLM endpoint).
set -euo pipefail
cd /g/data/jz31/zhaorui/weight_program/WeightCode
export PATH="$HOME/.local/bin:$PATH"
SEED=${1:-1}
OUT=/g/data/jz31/zhaorui/weight_program/cryoexp/engine/seed$SEED/synth
rm -rf "$OUT"
.venv/bin/weightcode --threads 2 synthesize \
  --task /g/data/jz31/zhaorui/weight_program/cryoexp/engine/seed$SEED/task --out "$OUT" \
  --sandbox-python .venv-sandbox/bin/python \
  --editor-isolation local \
  --editor-model 'glm-5.3[1m]' \
  --base-url-file "$HOME/.config/weightcode/glm_base_url" \
  --auth-token-file "$HOME/.config/weightcode/glm_auth_token" \
  --checks 8 --cells 100 --wall-s 5400 --program-wall-s 300
