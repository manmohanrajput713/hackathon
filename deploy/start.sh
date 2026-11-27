#!/bin/sh
# Vercel containers can only write to /tmp, and it is wiped when an instance stops.
# Seed the runtime folder from the files baked into the image, then start the API on $PORT.
set -e
mkdir -p "$FIELD_RUNTIME"
cp -Rn /app/runtime-seed/. "$FIELD_RUNTIME"/
export FIELD_API_ADDR="0.0.0.0:${PORT:-80}"
exec /app/field-console
