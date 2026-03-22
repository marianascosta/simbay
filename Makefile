.PHONY: install shell run run-macos run-local-observability check docker-build docker-run docker-simbay-up docker-simbay-down make-smoke-test docker-smoke-gpu

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

install:
	poetry install

shell:
	poetry shell

run:
	poetry run python main.py

run-local-observability:
	SIMBAY_METRICS_ENABLED=1 SIMBAY_METRICS_PORT=8000 poetry run python main.py

run-macos:
	MPLBACKEND=Agg "$(PROJECT_ROOT)/.venv/bin/python" "$(PROJECT_ROOT)/.venv/bin/mjpython" main.py

check:
	poetry run python -m compileall main.py src

make-smoke-test:
	SIMBAY_HEADLESS=1 SIMBAY_USE_MJX=1 SIMBAY_PARTICLES=1 python main.py

docker-smoke-gpu:
	docker compose run --build --rm -e SIMBAY_HEADLESS=1 -e SIMBAY_USE_MJX=1 -e SIMBAY_REQUIRE_GPU=1 -e SIMBAY_SMOKE_TEST_ONLY=1 -e SIMBAY_PARTICLES=1 -e SIMBAY_REPLAY_BENCHMARK_CHUNK_SIZES= -e SIMBAY_REPLAY_CHUNK_SIZE=0 simbay

docker-build:
	docker compose build

docker-run:
	docker compose up --build -d
	open "$(PROJECT_ROOT)/temp/particle_filter_evolution.png"

docker-simbay-up:
	docker compose up --build -d simbay

docker-simbay-down:
	docker compose stop simbay

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

down:
	docker compose down

up:
	docker compose up --build -d
