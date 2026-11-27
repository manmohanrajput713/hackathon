from pathlib import Path
import json
import os
import tempfile

APP = Path(__file__).resolve().parents[1]
# Pipeline sources (pinn, gas_pinc, failure_1, failure_2, pywaterflood): FIELD_PIPELINES, else ./pipelines, else siblings.
ROOT = Path(os.environ.get("FIELD_PIPELINES") or (APP / "pipelines" if (APP / "pipelines").is_dir() else APP.parent))
RUN = Path(os.environ.get("FIELD_RUNTIME") or APP / "runtime")

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

