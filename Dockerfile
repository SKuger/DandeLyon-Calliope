FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# The sample week is part of the image. Without it the first thing anyone
# sees is six empty cards, which is correct and a bad first impression.
COPY samples ./samples

# Speech recognition is deliberately not installed here. The image is
# 200MB instead of 2GB, it builds without downloading model weights, and
# the app falls back to replaying transcripts and says so when asked to
# transcribe something it has never seen. Install requirements-asr.txt to
# make it real.

RUN useradd --create-home learner \
 && mkdir -p /app/data /app/recordings \
 && chown -R learner /app/data /app/recordings
USER learner

EXPOSE 8000

# Seed, then serve. Seeding is idempotent, so this is safe on every start.
CMD ["sh", "-c", "python -m app.demo && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
