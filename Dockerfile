# Production image: Python 3.12, no dev reload, non-root user.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (never contains .env / secrets — see .dockerignore).
COPY app ./app

# Run as an unprivileged user.
RUN useradd --create-home --shell /usr/sbin/nologin appuser
USER appuser

EXPOSE 8000

# No --reload: this image is for production-style runs.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
