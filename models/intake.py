"""Staged CSV intake for well lists: preview first, then commit.

Columns: well_id, name, role, lift_type, field, status, notes.
"""
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from common import RUN
from schema import WELL_TYPES
import store
from store import UserError

STAGED = RUN / "staged"
WELL_COLS = {"code": ["well_id", "id", "code", "well", "uwi", "well_code"], "name": ["name", "well_name"],
             "role": ["role"], "lift": ["lift_type", "lift", "type", "artificial_lift"], "field": ["field", "pad", "field_pad"],
             "status": ["status"], "notes": ["notes", "comment", "comments"]}


def _norm(text):
    text = re.sub(r"\(.*?\)|\[.*?\]", "", str(text)).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def read_table(path, filename=""):
    import pandas as pd
    suffix = (Path(filename).suffix or Path(path).suffix).lower()
    try:
        if suffix in (".xlsx", ".xlsm", ".xls"):
            sheets = pd.read_excel(path, sheet_name=None, dtype=object)
            frame = sheets.get("Data", next(iter(sheets.values())))
        elif suffix in (".csv", ".txt", ".tsv"):
            frame = pd.read_csv(path, dtype=object, sep=None, engine="python", encoding="utf-8-sig")
        else:
            raise UserError("Upload a CSV (.csv) or Excel (.xlsx) file.")
    except UserError:
        raise
    except Exception as error:
        raise UserError(f"Could not read the file: {error}")
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.empty:
        raise UserError("The file has no data rows.")
    return frame


def _stage(path, filename, kind, field):
    STAGED.mkdir(parents=True, exist_ok=True)
    for old in STAGED.glob("*"):  # drop abandoned previews older than a day
        if time.time() - old.stat().st_mtime > 86400:
            old.unlink(missing_ok=True)
    token = uuid.uuid4().hex
    suffix = Path(filename).suffix.lower() or Path(path).suffix
    shutil.copy(path, STAGED / f"{token}{suffix}")
    (STAGED / f"{token}.json").write_text(json.dumps(dict(kind=kind, filename=filename, suffix=suffix, field=field)))
    return token


def _staged(token):
    if not re.fullmatch(r"[0-9a-f]{32}", str(token)):
        raise UserError("Invalid import token.")
    meta_path = STAGED / f"{token}.json"
    if not meta_path.exists():
        raise UserError("This preview has expired. Upload the file again.", 404)
    meta = json.loads(meta_path.read_text())
    return meta, STAGED / f"{token}{meta['suffix']}"


def _discard(token):
    for f in STAGED.glob(f"{token}*"):
        f.unlink(missing_ok=True)


def _pick(frame, names):
    for col in frame.columns:
        if _norm(col) in names:
            return col
    return None


# ---------- well lists ----------

def _well_type(role, lift):
    role, lift = _norm(role or ""), _norm(lift or "")
    if role.startswith("inj") or lift in ("injector", "water_injector"):
        return "injector"
    if lift in ("esp", "electric_submersible_pump"):
        return "esp"
    if lift in ("gas_lift", "gaslift", "gl"):
        return "gas_lift"
    if lift in ("natural", "flowing", "free_flow", "none", "producer") or (role.startswith("prod") and not lift):
        return "producer"
    return None


def preview_wells(db, path, filename, field):
    frame = read_table(path, filename)
    cols = {k: _pick(frame, set(v)) for k, v in WELL_COLS.items()}
    if not cols["code"] and not cols["name"]:
        raise UserError("The well list needs a well_id or name column.")
    rows = []
    for i, r in enumerate(frame.to_dict("records")):
        get = lambda k: (str(r[cols[k]]).strip() if cols[k] and r.get(cols[k]) is not None and str(r[cols[k]]).strip() not in ("", "nan") else "")
        code, name = get("code") or get("name"), get("name") or get("code")
        wtype = _well_type(get("role"), get("lift"))
        existing = store.find_well(db, code) or store.find_well(db, name)
        status = get("status").lower().replace(" ", "_") or "active"
        problem = None if wtype else "Role/lift type not recognised (use injector, or producer with esp, gas_lift or natural)"
        if existing and wtype and existing["type"] != wtype:
            problem = f"Exists as {WELL_TYPES[existing['type']]['label']}; type cannot change"
        rows.append(dict(line=i + 2, code=code, name=name, type=wtype, field=get("field") or field or "", status=status,
                         notes=get("notes"), action="update" if existing else "create", well_id=existing["id"] if existing else None,
                         problem=problem))
    token = _stage(path, filename, "wells", field)
    return dict(token=token, kind="wells", filename=filename, rows=rows, columns={k: v for k, v in cols.items() if v})


def commit_wells(db, token, body):
    meta, path = _staged(token)
    preview = preview_wells(db, path, meta["filename"], body.get("field", meta.get("field") or ""))
    _discard(preview["token"])
    created, updated, problems = [], [], []
    for r in preview["rows"]:
        if r["problem"]:
            problems.append(f"Row {r['line']} ({r['code']}): {r['problem']}")
            continue
        payload = dict(name=r["name"], code=r["code"], type=r["type"], field=r["field"], status=r["status"], notes=r["notes"])
        try:
            if r["action"] == "update":
                store.update_well(db, r["well_id"], payload)
                updated.append(r["name"])
            else:
                store.create_well(db, payload)
                created.append(r["name"])
        except UserError as error:
            problems.append(f"Row {r['line']} ({r['code']}): {error}")
    _discard(token)
    return dict(created=created, updated=updated, problems=problems)


def preview(db, kind, path, filename, field):
    if kind == "wells":
        return preview_wells(db, path, filename, field)
    raise UserError("Unknown import kind.", 404)


def commit(db, token, body):
    return commit_wells(db, token, body)
