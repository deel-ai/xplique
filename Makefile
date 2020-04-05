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
	sudo apt-get install python3.6 python3-pip && python3.6 -m pip install virtualenv
	virtualenv --no-site-packages venv
	make venv
	pip install -r requirements.txt
	pip install -r requirements_dev.txt


venv:
	. venv/bin/activate


test: venv
	$(PYTHON) -m pytest tests


test-disable-gpu: venv
	CUDA_VISIBLE_DEVICES=-1 $(PYTHON) -m pytest tests


lint: venv
	$(PYTHON) -m pylint --generated-members=cv2 xplique


doc: venv
	$(PYTHON) -m mkdocs build
	$(PYTHON) -m mkdocs gh-deploy