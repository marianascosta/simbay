.PHONY: help install install-dev bootstrap lint format local-mujoco local-mujoco-warp docker-mujoco docker-mujoco-warp

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MUJOCO_CMD := poetry run python main.py

help:
	@printf "Targets:\n"
	@printf "  install                Install production dependencies with Poetry.\n"
	@printf "  install-dev            Install all dependencies including dev extras (default Poetry behavior).\n"
	@printf "  bootstrap              Ensure Poetry is available for the repo tooling.\n"
	@printf "  lint                   Run Black in check mode for formatting validation.\n"
	@printf "  format                 Run Black to format the codebase.\n"
	@printf "  local-mujoco           Run the MuJoCo backend headlessly in the repository environment.\n"
	@printf "  local-mujoco-warp      Run the MuJoCo-Warp backend headlessly in the repository environment.\n"
	@printf "  docker-mujoco          Start the MuJoCo Docker service detached.\n"
	@printf "  docker-mujoco-warp     Start the MuJoCo-Warp Docker service detached with the GPU profile.\n"

install:
	poetry install --no-root

install-dev:
	poetry install

bootstrap:
	pip install poetry

lint:
	poetry run black --check -l 120 .

format:
	poetry run black -l 120 .

local-mujoco:
	SIMBAY_BACKEND=mujoco SIMBAY_HEADLESS=1 $(MUJOCO_CMD)

local-mujoco-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 $(MUJOCO_CMD)

docker-mujoco:
	SIMBAY_BACKEND=mujoco SIMBAY_HEADLESS=1 SIMBAY_PARTICLES=100 docker compose up --build -d simbay

docker-mujoco-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 SIMBAY_ENABLE_GPU=1 SIMBAY_PARTICLES=400 docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu up --build -d simbay
