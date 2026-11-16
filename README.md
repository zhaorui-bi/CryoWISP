# CryoWISP

**Cryo-EM particle picking via Weight-space Installation of Skills by Program synthesis**

CryoWISP installs a *new* particle-recognition capability into a **frozen** cryo-EM picking
network from a few hundred expert-labeled crops — no gradient training, no architecture
change. A frozen LLM editor reads the network's weights and forward code, then writes an
executable **weight program** that computes a tiny additive edit (v1: 1,026 of 11.69M
parameters, 0.009%, zero 2-D convolutions touched). The kept program re-runs on any
independently trained compatible network and recomputes its edit — one program, every network.

Built on the [WeightCode](https://github.com/Gen-Verse/WeightCode) weight-space programming
framework, evaluated on the [CryoPPP](https://www.nature.com/articles/s41597-023-02280-2)
expert-curated cryo-EM particle-picking dataset.

## Headline results (CryoPPP: 10005/10291/10081 → new dataset 10947)

Frozen base pickers recognize 240-px viral particles of the new dataset at **0% recall**
(~50% balanced accuracy). After one CryoWISP session (64 + 600 labeled crops):

| method (final split, new / old acc) | seed 1 | seed 2 | weights changed |
|---|---|---|---|
| Zero-shot | 49.75 / 85.94 | 49.00 / 83.13 | 0 |
| **CryoWISP weight program** | **72.25 / 89.69** | **75.50 / 89.38** (re-welded) | **1,026, no gradients** |
| Full fine-tune (20 ep, same crops) | 74.50 / 93.44 | 71.75 / 93.13 | 11.69M (100%) |
| Linear probe (fc, 1,026 params) | 49.00 / 93.59 | 48.00 / 92.34 | 1,026 |
| Knob gradient fit (the program's exact 1,026 knobs; 60 ep, 10x class weight) | 49.00 / 93.44 | 48.50 / 92.19 | 1,026 |
| Copied delta → other network | — | 48.25 / 83.28 | 1,026 (frozen) |
| Random knob perturbation | 49.50 / 83.91 | 49.00 / 82.19 | 1,026 |

Key evidence: with the *same* 1,026-parameter subset, gradient descent fails while the
program's structured (convex, old-anchored) fit succeeds; copying the fitted deltas to an
independently trained network fails while re-running the program succeeds; random
perturbations of the same knobs do nothing. External held-out test sets (2,040 new crops,
never visible to the editor): +12–13 pt on both seeds with ≤0.6 pt old-condition change.
Full logs and protocol disclosures: `experiments/RESULTS.md`, design: `experiments/DESIGN.md`.

## How it works — Freeze · Show · Weld · Prove · Re-weld

1. **Freeze** a particle/background ResNet-18 on old EMPIAR datasets (micrograph-level splits).
2. **Show** ~10² expert crops from the new dataset plus old-dataset anchors (experience split).
3. **Weld**: an LLM editor (GLM / Claude via an Anthropic-compatible endpoint) writes a weight
   program `update(student, experience) -> {tensor: delta}` inside fixed edit budgets. In v1 it
   discovered that BatchNorm 1-D affine parameters are the legal high-leverage knobs when the
   2-D relative-Frobenius budget cannot rotate a frozen head, and fit them by L2-regularized
   logistic regression with 10x new-condition weighting.
4. **Prove**: fixed selection rule on a held-out selection split; the frozen best program is
   scored once on the final split; below threshold → base model kept, nothing ships.
5. **Re-weld**: the program is a file; `apply` recomputes the edit for any compatible checkpoint.

## Repository layout

```
engine/        WeightCode framework (upstream MIT) + our adaptations:
                 --editor-isolation local   LocalKernel for hosts without user namespaces (HPC):
                                             NumPy-only -I kernel, memory rlimits, cell budgets,
                                             path guard; runs record program_isolation: none
                 --base-url-file /           Anthropic-compatible endpoints (GLM/Kimi/DeepSeek)
                 --auth-token-file           credential files for the editor CLI
               in scripts/run_editor_session.py, weightcode/{application,sandbox,harness}
experiments/   the CryoWISP pipeline on CryoPPP_Lite:
                 make_crops.py              micrograph-level splits -> 224x224 crops
                 train_base.py              frozen base pickers (2 seeds)
                 run_engine_task.sh         import-model + prepare-data x3 + prepare-task
                 run_synth_seed1.sh         editor session (login node, GLM endpoint)
                 baselines.py               linear probe / knob / full FT / copied delta / random
                 run_baselines.pbs          baseline suite (PBS)
                 DESIGN.md RESULTS.md METHOD.md
```

## Quickstart

```bash
# data: CryoPPP_Lite per-dataset tarballs (JPG micrographs + expert coordinates)
aria2c -x16 https://calla.rnet.missouri.edu/cryoppp_lite/10947.tar.gz
# crops -> base models -> engine task -> editor session -> re-weld
python experiments/make_crops.py --lite-root <lite> --out crops --old-ids 10005 10291 10081 --new-id 10947
python experiments/train_base.py --crops-dir crops --out-dir models --seed 1
bash  experiments/run_engine_task.sh 1
bash  experiments/run_synth_seed1.sh 1
engine/.venv/bin/weightcode apply --task engine/seed2/task --program engine/seed1/synth/final/program \
       --program-sha256 <sha256> --wall-s 900 --program-wall-s 480 --out engine/seed2/reuse
```

## Attribution & license

Engine code is a fork of [Gen-Verse/WeightCode](https://github.com/Gen-Verse/WeightCode)
(MIT, see `engine/LICENSE`); CryoWISP additions under the same MIT license. Data:
CryoPPP (CC BY 4.0, Dhakal et al., Sci Data 2023). This repository is research code for a
submission in preparation; numbers in `experiments/RESULTS.md` are the authoritative record.
