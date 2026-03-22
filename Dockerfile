FROM python:3.11-slim

ARG JAX_VERSION=0.5.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --no-interaction --no-ansi
RUN pip install "jax[cuda12]==${JAX_VERSION}"

COPY assets ./assets
COPY src ./src
COPY main.py ./

CMD ["python", "main.py"]
