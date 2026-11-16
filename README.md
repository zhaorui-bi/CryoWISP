# CryoWISP

Weight-space installation of new particle-picking capabilities into frozen cryo-EM networks.

## What is CryoWISP?

CryoWISP is an open-source framework for adapting frozen cryo-EM particle pickers to new
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

Frozen pickers trained with CryoWISP on three CryoPPP datasets (10005/10291/10081) gain
recognition of a held-out particle type (10947, 240-px viral particles, 0% base recall):
+22.5 / +26.5 pt balanced accuracy on two independently trained networks from one program,
matching full fine-tuning while changing 1,026 of 11.69M weights (0.009%) with zero gradient
steps. Full numbers and protocol disclosures: `experiments/RESULTS.md`.

CryoWISP is research code for a submission in preparation, developed by a single person on
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
`cryowisp/patch/`; the budgets explain where the programs actually edit:

- `dense_delta` ops: at most 5% of entries, relative Frobenius change <= 0.5 for rank >= 2
  tensors (`cryowisp/patch/limits.py`)
- the 1-D escape hatch: per-channel BatchNorm affine parameters (gamma/beta) are 1-D, hence
  outside the 2-D relative-Frobenius cap. Every kept program so far welds through
  `layer4.1.bn2.{weight,bias}` plus `fc.bias` — 1,026 parameters that re-gate the pooled
  features in front of the frozen head
- selection and release: each candidate is scored on a held-out selection split; the frozen
  best is scored once on the final split (`cryowisp/application/runtime.py`); below
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
and import-order fixes across `cryowisp/` and `experiments/`.

## Contributors

CryoWISP is developed by zhaorui-bi. The engine derives from WeightCode
(Gen-Verse); the WeightCode kernel/patch/application architecture is preserved and credited
in `LICENSE` and `experiments/engine/ENGINE.md`.

## Cite

If you find this project useful, consider citing the WeightCode framework and the CryoPPP
dataset (references in the links above). A CryoWISP preprint is in preparation.
