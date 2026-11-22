from pathlib import Path
import json
import os
import tempfile

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
RUN = APP / "runtime"

def save(name, data):
    RUN.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=RUN, delete=False, encoding="utf8") as stream:
        try:
            json.dump(data, stream, allow_nan=False, separators=(",", ":"))
        except Exception:
            stream.close()
            Path(stream.name).unlink(missing_ok=True)
            raise
    os.replace(stream.name, RUN / name)

