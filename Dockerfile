# The builder image, used to build the virtual environment
FROM python:3.11.12-slim AS builder

RUN pip install poetry==1.7.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-dev --no-root && rm -rf $POETRY_CACHE_DIR

# The runtime image, used to just run the code provided its virtual environment
FROM python:3.11.12-slim AS runtime

# Install emoji fonts, fontconfig, and locale support
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-color-emoji \
    fontconfig \
    locales \
    && fc-cache -f -v \
    && rm -rf /var/lib/apt/lists/*

# Set locale for Unicode support
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY chatgpt_streamlit.py ./
COPY .chat-env ./

EXPOSE 8503

CMD ["streamlit", "run", "chatgpt_streamlit.py", "--server.port", "8503"]
