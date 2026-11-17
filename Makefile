.PHONY: setup-env lint format test smoke

# Set UV_CACHE_DIR to a writable location with quota headroom (an over-quota or deleted
# default cache makes uv sync / uv run stall or fail on HPC machines).
export UV_CACHE_DIR ?= $(HOME)/.cache/uv

setup-env:
	uv sync --locked --extra train --extra dev
	bash experiments/engine/scripts/setup_sandbox_venv.sh

lint:
	uvx ruff check cryowsp experiments

format:
	uvx ruff check --fix cryowsp experiments && uvx ruff format experiments

test:
	.venv/bin/python -m pytest -q experiments/engine/tests/test_patch_ops.py \
		experiments/engine/tests/test_patch_validation.py \
		experiments/engine/tests/test_application.py

smoke:
	bash experiments/smoke_apply.sh
