#!/usr/bin/env bash
# Build the WeightCode engine task for one seed: import model, pack the three data roles, bind the task.
# Usage: bash run_engine_task.sh <seed> [crops_dir] [suffix]
set -euo pipefail
SEED=${1:?seed}
CROPS=${2:-/g/data/jz31/zhaorui/weight_program/cryoexp/data/crops}
SUFFIX=${3:-}
REPO=/g/data/jz31/zhaorui/weight_program
ENG=$REPO/cryoexp/engine
CROPDIR=$CROPS

cd $REPO/WeightCode
rm -rf $ENG/seed$SEED$SUFFIX
mkdir -p $ENG/seed$SEED$SUFFIX
W=$ENG/seed$SEED$SUFFIX

.venv/bin/weightcode import-model \
  --checkpoint $REPO/cryoexp/models/model_seed$SEED.pt \
  --labels $REPO/cryoexp/models/labels.json --out $W/model

.venv/bin/weightcode prepare-data --model $W/model \
  --new $CROPDIR/new_exp --old $CROPDIR/old_exp --out $W/experience.npz
.venv/bin/weightcode prepare-data --model $W/model \
  --new $CROPDIR/new_sel --old $CROPDIR/old_sel --out $W/selection.npz
.venv/bin/weightcode prepare-data --model $W/model \
  --new $CROPDIR/new_fin --old $CROPDIR/old_fin --out $W/final.npz

.venv/bin/weightcode prepare-task --model $W/model \
  --experience $W/experience.npz --selection $W/selection.npz --final $W/final.npz \
  --goal 'Recognize particles of the new dataset 10947 (240 px viral particles) from a few labeled crops while keeping particle/background recognition on the old datasets 10005/10291/10081' \
  --retention-drop 0.01 --min-new-gain 0.01 --out $W/task

echo "engine task ready: $W/task"
