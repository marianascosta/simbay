.PHONY: bootstrap install install-dev local-mujoco local-mujoco-warp docker-mujoco docker-mujoco-warp

bootstrap:
	python -m pip install --upgrade pip
	python -m pip install poetry

install:
	poetry install --no-root

install-dev:
	poetry install --no-root --with dev

local-mujoco:
	SIMBAY_BACKEND=cpu SIMBAY_HEADLESS=0 poetry run mjpython main.py

local-mujoco-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 poetry run python main.py

docker-mujoco:
	docker compose -f docker-compose.mujoco.yml up --build simbay-cpu

docker-mujoco-warp:
	docker compose -f docker-compose.mujoco-warp.yml up --build simbay
