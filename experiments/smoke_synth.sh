#!/usr/bin/env bash
# Mini editor-session smoke test on the synthetic task (validates LocalKernel + GLM endpoint + MCP loop).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
rm -rf /tmp/synth_work/synth
.venv/bin/cryowsp --threads 2 synthesize \
  --task /tmp/synth_work/task --out /tmp/synth_work/synth \
  --sandbox-python .venv-sandbox/bin/python \
  --editor-isolation local \
  --editor-model 'glm-5.3[1m]' \
  --base-url-file "$HOME/.config/cryowsp/glm_base_url" \
  --auth-token-file "$HOME/.config/cryowsp/glm_auth_token" \
  --checks 2 --cells 30 --wall-s 900
