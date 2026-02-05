.PHONY: help prepare-dev test test-disable-gpu doc serve-doc
.DEFAULT: help

help:
	@echo "make prepare-dev"
	@echo "       create and prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests and linting in current env"
	@echo "make test-disable-gpu"
	@echo "       run test with gpu disabled"
	@echo "make pc_check"
	@echo "       check all files using pre-commit tool"
	@echo "make pc_update"
	@echo "       update pre-commit tool"
	@echo "make serve-doc"
	@echo "       run documentation server for development"
	@echo "make doc"
	@echo "       build mkdocs documentation"

prepare-dev:
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh;
	uv venv
	. .venv/bin/activate && uv sync
	. .venv/bin/activate && pre-commit autoupdate --repo https://github.com/pre-commit/pre-commit-hooks
	. .venv/bin/activate && pre-commit install
	. .venv/bin/activate && pre-commit install-hooks

test:
	. .venv/bin/activate && UV_CACHE_DIR="$(uv cache dir)" nohup tox > tox.out &

test-disable-gpu:
	. .venv/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox

pc_check:
	. .venv/bin/activate && pre-commit run --all-files

pc_update:
	. .venv/bin/activate && pre-commit autoupdate

doc:
	. .venv/bin/activate && mkdocs build

serve-doc:
	. .venv/bin/activate && CUDA_VISIBLE_DEVICES=-1 mkdocs serve
