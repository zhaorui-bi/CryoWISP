.PHONY: setup-env lint format test smoke

setup-env:
	uv sync --locked --extra train --extra dev
	bash experiments/engine/scripts/setup_sandbox_venv.sh

lint:
	uvx ruff check cryowisp experiments

format:
	uvx ruff check --fix cryowisp experiments && uvx ruff format experiments

test:
	uv run python -m pytest -q experiments/engine/tests/test_patch_ops.py \
		experiments/engine/tests/test_patch_validation.py \
		experiments/engine/tests/test_application.py

smoke:
	bash experiments/smoke_apply.sh
