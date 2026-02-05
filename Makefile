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
	uv venv xplique_dev_env
	. xplique_dev_env/bin/activate && uv pip install -r requirements.txt -r requirements_dev.txt
	. xplique_dev_env/bin/activate && pre-commit install
	. xplique_dev_env/bin/activate && pre-commit install-hooks

test:
	. xplique_dev_env/bin/activate && tox

test-disable-gpu:
	. xplique_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 tox

pc_check:
	. xplique_dev_env/bin/activate && pre-commit run --all-files

pc_update:
	. xplique_dev_env/bin/activate && pre-commit autoupdate

doc:
	. xplique_dev_env/bin/activate && mkdocs build

serve-doc:
	. xplique_dev_env/bin/activate && CUDA_VISIBLE_DEVICES=-1 mkdocs serve
