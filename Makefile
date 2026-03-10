.PHONY: install shell run run-macos check

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

install:
	poetry install

shell:
	poetry shell

run:
	poetry run python main.py

run-macos:
	MPLBACKEND=Agg "$(PROJECT_ROOT)/.venv/bin/python" "$(PROJECT_ROOT)/.venv/bin/mjpython" main.py

check:
	poetry run python -m compileall main.py src
