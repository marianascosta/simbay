.PHONY: install shell local-mujoco local-mujoco-warp docker-mujoco docker-mujoco-warp run run-warp simbay-run simbay-run-warp run-macos run-local-observability check format lint docker-build docker-run docker-simbay-up docker-simbay-down make-smoke-test make-smoke-test-warp docker-simbay-profile

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
COMPOSE_MUJOCO := docker compose -f docker-compose.mujoco.yml
COMPOSE_MUJOCO_WARP := docker compose -f docker-compose.mujoco-warp.yml

install:
	poetry install --no-root

shell:
	poetry shell

local-mujoco:
	SIMBAY_HEADLESS=0 poetry run mjpython main.py

local-mujoco-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 poetry run python main.py

docker-mujoco:
	$(COMPOSE_MUJOCO) up --build simbay-mujoco

docker-mujoco-warp:
	$(COMPOSE_MUJOCO_WARP) up --build simbay-mujoco-warp

# Backward-compatible aliases
run: local-mujoco
run-warp: local-mujoco-warp
simbay-run: docker-mujoco
simbay-run-warp: docker-mujoco-warp

run-local-observability:
	poetry run python main.py

run-macos:
	SIMBAY_HEADLESS=0 poetry run mjpython main.py

check:
	poetry run python -m compileall main.py src

make-smoke-test:
	SIMBAY_HEADLESS=1 SIMBAY_PARTICLES=1 python main.py

make-smoke-test-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 SIMBAY_PARTICLES=1 python main.py

docker-build:
	$(COMPOSE_MUJOCO) build simbay-mujoco
	$(COMPOSE_MUJOCO_WARP) build simbay-mujoco-warp

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
