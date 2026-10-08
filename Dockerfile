# Aurora Forecast server for Hugging Face Spaces (Docker SDK, free CPU).
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update     && apt-get install -y --no-install-recommends libgomp1     && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/src

COPY --chown=app pyproject.toml CLAUDE.md ./
COPY --chown=app aurora ./aurora
COPY --chown=app app ./app
COPY --chown=app models ./models
RUN pip install --no-cache-dir -e . \n    && mkdir -p data && chown app:app . data

USER app
ENV PYTHONUNBUFFERED=1     OPENBLAS_NUM_THREADS=1     BRIEFING_DB=/tmp/briefings.sqlite

EXPOSE 7860
# One worker: the in-process scheduler must run exactly once.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
