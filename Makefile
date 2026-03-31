.PHONY: help install shell \
  run run-mujoco run-warp run-mujoco-warp run-local-observability run-macos \
  local-simbay local-simbay-warp \
  mujo-smoke mujo-smoke-warp make-smoke-test make-smoke-test-warp \
  docker-build docker-run \
  docker-simbay docker-simbay-warp docker-simbay-up docker-simbay-down docker-simbay-profile \
  bootstrap setup-precommit

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

MUJOCO_CMD := poetry run python main.py
MUJOCO_WARP_ENV := SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1
SMOKE_ENV := SIMBAY_HEADLESS=1 SIMBAY_PARTICLES=1
SMOKE_WARP_ENV := SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 SIMBAY_PARTICLES=1
SMOKE_CMD := python main.py

help:
	@printf "Usage: make <target>\n\n"
	@printf "Local sims:\n"
	@printf "  local-simbay       Runs the default MuJoCo backend locally.\n"
	@printf "  local-simbay-warp  Runs MuJoCo-warp locally with headless metrics.\n"
	@printf "  run/run-warp       Aliases for the above, respectively.\n"
	@printf "  run-local-observability  Run the default backend without warp (for metrics/observability).\n"
	@printf "  run-macos          Run through the repository mjpython wrapper on macOS.\n"
	@printf "\nSmoke test targets:\n"
	@printf "  make-smoke-test     Quick headless run for the default backend.\n"
	@printf "  make-smoke-test-warp\tQuick headless run with the warp backend.\n"
	@printf "\nDocker helpers:\n"
	@printf "  docker-build       Build the compose services (CPU default).\n"
	@printf "  docker-run         Bring up Prometheus, Grafana, node-exporter, and the app (add SIMBAY_ENABLE_GPU=1 to include the GPU overlay).\n"
	@printf "  docker-simbay      Bring up the simbay service using the default backend.\n"
	@printf "  docker-simbay-warp Bring up the simbay service with the warp backend.\n"
	@printf "  docker-simbay-up/down/profile  Compose helpers for the simbay service.\n"
	@printf "\nOther helpers: install, shell, check, format, lint, bootstrap, setup-precommit.\n"

install:
	poetry install --no-root

shell:
	poetry shell

# MuJoCo entrypoints
local-simbay: run-mujoco

local-simbay-warp: run-mujoco-warp

run: local-simbay

run-mujoco:
	$(MUJOCO_CMD)

run-warp: local-simbay-warp

run-mujoco-warp:
	$(MUJOCO_WARP_ENV) $(MUJOCO_CMD)

run-local-observability:
	$(MUJOCO_CMD)

run-macos:
	"$(PROJECT_ROOT)/.venv/bin/python" "$(PROJECT_ROOT)/.venv/bin/mjpython" main.py

check:
	poetry run python -m compileall main.py src

# Fast headless smoke tests
make-smoke-test: mujo-smoke
make-smoke-test-warp: mujo-smoke-warp

mujo-smoke:
	$(SMOKE_ENV) $(SMOKE_CMD)

mujo-smoke-warp:
	$(SMOKE_WARP_ENV) $(SMOKE_CMD)

docker-build:
	docker compose build

docker-run:
ifneq ($(SIMBAY_ENABLE_GPU),1)
	docker compose up --build
else
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile gpu up --build
endif
	open "$(PROJECT_ROOT)/temp/particle_filter_evolution.png"

docker-simbay:
	docker compose up --build simbay

docker-simbay-warp:
	SIMBAY_BACKEND=mujoco-warp docker compose up --build simbay

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
