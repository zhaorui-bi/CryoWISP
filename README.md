# CryoWSP

**Cryo-EM Weight-Space Programming** — new picking skills welded into frozen networks, one program at a time.

## What is CryoWSP?

CryoWSP is an open-source framework for adapting frozen cryo-EM particle pickers to new
datasets without gradient training. It is built on top of components from other open-source
projects, including:

- [WeightCode](https://github.com/Gen-Verse/WeightCode): the weight-space programming engine
  (edit budgets, checked weight programs, program reuse)
- [CryoPPP](https://www.nature.com/articles/s41597-023-02280-2): expert-curated cryo-EM
  micrographs and particle coordinates
- Claude Code CLI: the editor runtime, pointed at any Anthropic-compatible endpoint
  (GLM, Kimi, DeepSeek) via credential files

It extends these projects with additional features, including:

- the Freeze / Show / Weld / Prove / Re-weld pipeline for cryo-EM particle picking
  (micrograph-level splits, patch extraction, few-shot capability installation)
- `LocalKernel` program runtime for HPC hosts that disable unprivileged user namespaces
  (no bubblewrap needed; runs record `program_isolation: none`)
- endpoint-agnostic editor sessions (`--base-url-file` / `--auth-token-file`)
- a matched-capacity baseline suite (linear probe, same-knob gradient fits, copied-delta
  and random-knob controls, full fine-tuning) under one evaluation protocol

## Results

Two transfer regimes on CryoPPP (frozen pickers trained on 10005/10291/10081; balanced
accuracy new / old on the engine's final split, scored once; **bold** = one weight program,
no gradients, 1,026 / 11.69M weights changed, zero 2-D convolutions touched):

| method | 10947 far shift, seed 1 | seed 2 (re-welded) | 10760 near shift, seed 1 | seed 2 (re-welded) |
|---|---|---|---|---|
| Zero-shot | 49.8 / 85.9 | 49.0 / 83.1 | 52.8 / 85.9 | 51.8 / 83.1 |
| Linear probe (1,026 params) | 49.0 / 93.6 | 48.0 / 92.3 | 58.0 / 92.0 | 59.5 / 92.8 |
| Knob gradient fit (the program's exact 1,026 params; 60 ep, 10x class weight) | 49.0 / 93.4 | 48.5 / 92.2 | 56.8 / 93.4 | 55.8 / 92.0 |
| Full fine-tune (20 ep, same crops) | 74.5 / 93.4 | 71.8 / 93.1 | 69.8 / 91.3 | 67.3 / 92.7 |
| **CryoWSP weight program** | **72.3 / 89.7** | **75.5 / 89.4** | **60.8 / 89.2** | **59.8 / 90.8** |

Four findings (all under one protocol: same experience crops, selection-based model choice,
final scored once):

1. **Far shift is where programs win.** 10947 (240-px viral particles, 0% base recall): the
   program matches or beats full fine-tuning on new-condition accuracy (+22.5 / +26.5 pt on
   two independently trained networks) while gradient fitting of the *same* 1,026 parameters
   fails entirely (~49%) — the program's convex, old-anchored fit finds a solution gradient
   descent cannot. Near shift (10760, base already 52.8%): full fine-tuning leads; the
   program still adds +8.0 pt on both networks with the identical gain.
2. **Programs transfer, deltas do not.** Re-welding the kept program on an independently
   trained network reproduces the gain (10947: +26.5; 10760: +8.00 = seed 1's +8.00);
   copying the fitted deltas across fails (48.3, i.e. zero-shot level). Random perturbations
   of the same knobs also do nothing.
3. **Dose-response dichotomy.** Scaling the kept edit by alpha: old-condition accuracy rises
   smoothly (85.9 -> 89.7 by alpha=0.75) while new-condition accuracy is exactly zero until
   alpha=1 (49.0 -> 72.3 in one step), then decays on overshoot. Retention is a convex
   re-gating effect; acquisition is a single boundary crossing at the program's own dose.
4. **Experience composition matters more than amount.** 64 crops from 4 micrographs (v1)
   generalize across micrographs (+13.4 external); 177 crops from the same 4 (v2) or 64 from
   8 with a different draw (v3) reach +17/+16 on their final splits but collapse externally
   (+0 / +3). Session-level variance is large; the external-validating protocol is what
   keeps the headline honest.

CryoWSP is research code for a submission in preparation, developed by a single person on
an HPC login node. Expect rough edges. It can serve as a good starting point for research on
weight-space adaptation of cryo-EM pipelines: the authoritative record of every session,
program and baseline lives under `experiments/`; dig around there.

## Install

`make setup-env` installs the engine and training dependencies and builds the NumPy-only
program interpreter. On hosts where bubblewrap cannot run (e.g. user namespaces disabled,
`sysctl user.max_user_namespaces` = 0), skip the doctor check and pass
`--editor-isolation local` to `synthesize`.

## Quick Start

After downloading a CryoPPP_Lite tarball (JPG micrographs + expert coordinates):

```bash
make smoke                                   # engine end-to-end on a synthetic task
python experiments/make_crops.py --lite-root <lite-dir> --out experiments/data/crops \
       --old-ids 10005 10291 10081 --new-id 10947
python experiments/train_base.py --crops-dir experiments/data/crops --out-dir experiments/models --seed 1
bash experiments/run_engine_task.sh 1        # import model, bind experience/selection/final
bash experiments/run_synth_seed1.sh 1        # editor session (needs an endpoint credential)
```

## How Do Weight-Space Welds Work?

A *weight program* is an executable `update(student, experience) -> {tensor: delta}` written
by a frozen LLM inside fixed edit budgets. The budgets and their validation live in
`cryowsp/patch/`; the budgets explain where the programs actually edit:

- `dense_delta` ops: at most 5% of entries, relative Frobenius change <= 0.5 for rank >= 2
  tensors (`cryowsp/patch/limits.py`)
- the 1-D escape hatch: per-channel BatchNorm affine parameters (gamma/beta) are 1-D, hence
  outside the 2-D relative-Frobenius cap. Every kept program so far welds through
  `layer4.1.bn2.{weight,bias}` plus `fc.bias` — 1,026 parameters that re-gate the pooled
  features in front of the frozen head
- selection and release: each candidate is scored on a held-out selection split; the frozen
  best is scored once on the final split (`cryowsp/application/runtime.py`); below
  threshold, the base model ships unchanged
- re-welding: the kept program is a file; `apply` recomputes the edit for any compatible
  checkpoint. Copying the fitted deltas to another network fails (48.25% vs 75.50% when
  re-welding) — see `experiments/baselines.py --mode copied_delta`

## E2E Testing

`make test` runs the patch/budget/application test suite (tests that need a bubblewrap
sandbox fail on hosts without user namespaces; everything else must pass).
`make smoke` runs the full engine chain — import-model, prepare-data, prepare-task,
pack-program, apply — on a synthetic task and must end with a written report.

### Code Style

Install `ruff`, then run `make lint` to enumerate and `make format` to apply formatting
and import-order fixes across `cryowsp/` and `experiments/`.

## Contributors

CryoWSP is developed by zhaorui-bi. The engine derives from WeightCode
(Gen-Verse); the WeightCode kernel/patch/application architecture is preserved and credited
in `LICENSE` and `experiments/engine/ENGINE.md`.

## Cite

If you find this project useful, consider citing the WeightCode framework and the CryoPPP
dataset (references in the links above). A CryoWSP preprint is in preparation.
