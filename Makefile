.PHONY: install shell run run-warp run-macos run-local-observability check format lint docker-build docker-run docker-simbay-up docker-simbay-down make-smoke-test make-smoke-test-warp docker-simbay-profile

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

install:
	poetry install --no-root

shell:
	poetry shell

run:
	poetry run python main.py

run-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 SIMBAY_METRICS_ENABLED=1 poetry run python main.py

run-local-observability:
	SIMBAY_METRICS_ENABLED=1 SIMBAY_METRICS_PORT=8000 poetry run python main.py

run-macos:
	MPLBACKEND=Agg "$(PROJECT_ROOT)/.venv/bin/python" "$(PROJECT_ROOT)/.venv/bin/mjpython" main.py

check:
	poetry run python -m compileall main.py src

make-smoke-test:
	SIMBAY_HEADLESS=1 SIMBAY_USE_MJX=1 SIMBAY_PARTICLES=1 python main.py

make-smoke-test-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 SIMBAY_PARTICLES=1 python main.py

docker-build:
	docker compose build

docker-run:
	docker compose up --build
	open "$(PROJECT_ROOT)/temp/particle_filter_evolution.png"

docker-simbay-up:
	docker compose up --build -d simbay

docker-simbay-profile:
	SIMBAY_ENABLE_NSIGHT=1 docker compose up --build simbay

docker-simbay-down:
	docker compose stop simbay

bootstrap:
	pip install poetry

setup-precommit:
	poetry run pre-commit install

format:
	poetry run black -l 120 ./

lint:
	poetry run black --check -l 120 ./
