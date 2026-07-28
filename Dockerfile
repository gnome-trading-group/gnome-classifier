FROM python:3.13-slim

RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0" awslambdaric

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

COPY classifier/ classifier/
COPY scripts/ scripts/

ENTRYPOINT ["python", "-m", "classifier.workers.runner"]
