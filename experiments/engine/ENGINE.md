# CryoWISP engine

The executable core of CryoWISP: model import and parity checking, task binding,
the LLM editor harness with checked weight programs, patch validation and edit
budgets, program packaging and reuse, and the sandboxed (or local-kernel) program
runtime.

```
cryowisp/            the Python package (CLI entry point: cryowisp)
  application/       import-model / prepare-data / prepare-task / apply / synthesize
  sandbox/           bubblewrap kernel + LocalKernel (hosts without user namespaces)
  harness/           editor sessions, launcher (Anthropic-compatible endpoints)
  patch/             weight-edit budgets, patch ops and validation
scripts/             research drivers (train_base, run_editor_session, ...)
skills/cryowisp/     the operator skill: CLI walkthrough (references/workflow.md)
programs/            reusable example weight program
tests/               unit and integration tests
```

## Install

```bash
uv sync --locked --extra train --extra dev
bash scripts/setup_sandbox_venv.sh      # NumPy-only program interpreter
uv run cryowisp doctor --sandbox-python .venv-sandbox/bin/python
```

On hosts that disable unprivileged user namespaces (e.g. NCI Gadi: `user.max_user_namespaces = 0`),
bubblewrap cannot run; use the explicit local mode instead — the editor kernel runs as a
plain `-I` NumPy process with memory rlimits and a static path guard, and every run records
`program_isolation: none`:

```bash
uv run cryowisp --threads 2 synthesize --task <task> --out <out> \
    --sandbox-python .venv-sandbox/bin/python --editor-isolation local \
    --editor-model <model> --base-url-file <url-file> --auth-token-file <token-file> \
    --checks 6 --cells 120 --wall-s 7200 --program-wall-s 420
```

The engine derives from the [WeightCode](https://github.com/Gen-Verse/WeightCode)
weight-space programming framework (MIT); CryoWISP adapts it to cryo-EM particle picking
and to HPC hosts. See the repository root README for the method and results.
