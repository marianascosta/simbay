.PHONY: bootstrap install install-dev local-mujoco local-mujoco-warp docker-mujoco docker-mujoco-warp

bootstrap:
	python -m pip install --upgrade pip
	python -m pip install poetry

install:
	poetry install --no-root

install-dev:
	poetry install --no-root --with dev

local-mujoco:
	SIMBAY_ENVIRONMENT=dev SIMBAY_BACKEND=cpu SIMBAY_HEADLESS=0 poetry run mjpython main.py

local-mujoco-warp:
	SIMBAY_ENVIRONMENT=dev SIMBAY_BACKEND=mujoco-warp SIMBAY_HEADLESS=1 poetry run python main.py

docker-mujoco:
	SIMBAY_BACKEND=cpu SIMBAY_DOCKER_TARGET=base SIMBAY_ENABLE_NSIGHT=0 docker compose -f docker-compose.yml up --build -d

docker-mujoco-warp:
	SIMBAY_BACKEND=mujoco-warp SIMBAY_DOCKER_TARGET=cuda docker compose -f docker-compose.yml up --build -d 
