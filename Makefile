.PHONY: help prepare-dev test test-disable-gpu lint doc serve-doc
.DEFAULT: help

PYTHON=venv/bin/python

help:
	@echo "make prepare-dev"
	@echo "       prepare development environment, use only once"
	@echo "make test"
	@echo "       run tests"
	@echo "make test-disable-gpu"
	@echo "       run tests with gpu disabled"
	@echo "make lint"
	@echo "       run pylint"
	@echo "make serve-doc"
	@echo "       run documentation server for development"
	@echo "make doc"
	@echo "       build mkdoc documentation"

prepare-dev:
	python3 -m pip install virtualenv
	python3 -m virtualenv venv
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements_dev.txt

venv:
	. venv/bin/activate

test: venv
	$(PYTHON) -m tox

test-disable-gpu: venv
	CUDA_VISIBLE_DEVICES=-1 $(PYTHON) -m tox

lint: venv
	$(PYTHON) -m pylint xplique

doc: venv
	$(PYTHON) -m mkdocs build
	$(PYTHON) -m mkdocs gh-deploy

serve-doc: venv
	CUDA_VISIBLE_DEVICES=-1 $(PYTHON) -m mkdocs serve