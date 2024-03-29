SHELL=/bin/bash
PYTHON=venv/bin/python
PYTHON_BIN=venv/bin

.PHONY: pre-commit test pip clean

venv:
	python -m venv venv
	venv/bin/pip install -U pip
	venv/bin/pip install -r build.requirements.txt

clean:
	rm -r venv
	rm -r target/

pre-commit: venv
	$(PYTHON_BIN)/isort .
	$(PYTHON_BIN)/black .
	$(PYTHON_BIN)/blackdoc .
	$(PYTHON_BIN)/mypy
	$(PYTHON_BIN)/flake8
	make -C .. fmt_toml
	cargo fmt --all

test: venv
	source venv/bin/activate; \
	$(PYTHON_BIN)/maturin develop; \
	$(PYTHON) -m pytest tests

test-with-cov: venv
	@cd tests && ../$(PYTHON) -m pytest --cov=polars --cov-fail-under=91 --import-mode=importlib

doctest:
	cd tests && ../$(PYTHON) run_doc_examples.py

install-wheel:
	pip install --force-reinstall -U wheels/polars-*.whl

build-no-venv:
	maturin build -o wheels

build-and-test-no-venv:
	maturin build -o wheels
	pip install --force-reinstall -U wheels/polars-*.whl
	pytest tests

install-no-venv: build-no-venv install-wheel

