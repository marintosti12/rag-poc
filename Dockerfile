FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS deps

ENV POETRY_VERSION=1.8.3
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* ./ 

RUN poetry install --no-interaction --no-ansi --only main

FROM deps AS app

COPY src ./src


EXPOSE 8000

ENV APP_MODULE=src.api.main:app \
    HOST=0.0.0.0 \
    PORT=8000

CMD ["bash", "-lc", "uvicorn ${APP_MODULE} --host ${HOST} --port ${PORT}"]
