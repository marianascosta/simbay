.PHONY: install shell run run-macos check docker-build docker-run

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

docker-build:
	docker compose build

docker-run:
	docker compose up --build
	open "$(PROJECT_ROOT)/temp/particle_filter_evolution.png"

bootstrap:
	pip install poetry

install:
	poetry install --no-root

setup-precommit:
	poetry run pre-commit install

format:
	poetry run black -l 120 ./

lint:
	poetry run black --check -l 120 ./