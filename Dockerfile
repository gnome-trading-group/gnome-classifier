FROM python:3.13-slim

RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

WORKDIR /var/task

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --with lambda --no-root --no-interaction --no-ansi

COPY classifier/ classifier/
COPY handler.py .

ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["handler.handler"]
