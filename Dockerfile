# Field Console API: the Rust/Axum server plus the Python model workers it runs.
# Used by the "api" service in vercel.json; also runs anywhere Docker does.

FROM python:3.12-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked
# pywaterflood has a Rust core (abi3 wheel); build it from the bundled checkout.
COPY pipelines/pywaterflood /pywaterflood
RUN pip wheel --no-deps -w /wheels /pywaterflood

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/*.whl && rm -rf /wheels

WORKDIR /app
COPY --from=build /src/target/release/field-console /app/field-console
COPY models ./models
COPY pipelines ./pipelines
COPY test-data ./test-data
COPY deploy/start.sh /app/start.sh

ENV FIELD_PYTHON=python \
    FIELD_PIPELINES=/app/pipelines \
    PYTHONUNBUFFERED=1
# Precompute the offline replay inputs (crm.json, pump.json, curves.json) into the image.
RUN FIELD_RUNTIME=/app/runtime-seed python models/prepare.py \
    && FIELD_RUNTIME=/app/runtime-seed python models/curves.py

ENV FIELD_RUNTIME=/tmp/runtime \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    HOME=/tmp \
    PORT=80
EXPOSE 80
RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
