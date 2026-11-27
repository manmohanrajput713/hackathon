"""SQLite registries: wells, datasets (versioned rows), runs and scenarios."""
import json
import math
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from common import RUN
from schema import DATASETS, FACILITY, POLICIES, SETTINGS, STATUSES, WELL_TYPES

DB = RUN / "field.db"
FIELD = "__field__"


class UserError(Exception):
    """A problem with the request that the user can fix. The message is shown in the UI."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def connect():
    RUN.mkdir(exist_ok=True)
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS wells(
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, type TEXT NOT NULL,
            field TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'active', notes TEXT NOT NULL DEFAULT '',
            settings TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rows(
            id INTEGER PRIMARY KEY, well_id TEXT NOT NULL REFERENCES wells(id) ON DELETE CASCADE,
            dataset TEXT NOT NULL, k TEXT NOT NULL, sk REAL NOT NULL, data TEXT NOT NULL,
            UNIQUE(well_id, dataset, k));
        CREATE INDEX IF NOT EXISTS rows_order ON rows(well_id, dataset, sk);
        CREATE TABLE IF NOT EXISTS results(
            well_id TEXT NOT NULL, kind TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(well_id, kind));
        CREATE TABLE IF NOT EXISTS versions(
            well_id TEXT NOT NULL, dataset TEXT NOT NULL, version INTEGER NOT NULL, created_at TEXT NOT NULL,
            source TEXT NOT NULL, rows INTEGER NOT NULL, PRIMARY KEY(well_id, dataset, version));
        CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, field TEXT NOT NULL DEFAULT '',
            scenario_id TEXT, well_id TEXT, parent_id TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL,
            started_at TEXT, finished_at TEXT, params TEXT NOT NULL DEFAULT '{}', inputs TEXT NOT NULL DEFAULT '{}',
            outputs TEXT, summary TEXT, error TEXT);
        CREATE INDEX IF NOT EXISTS runs_recent ON runs(created_at);
        CREATE TABLE IF NOT EXISTS scenarios(
            id TEXT PRIMARY KEY, name TEXT NOT NULL, field TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '',
            facility TEXT NOT NULL DEFAULT '{}', policy TEXT NOT NULL DEFAULT 'hold', created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, UNIQUE(field, name));
    """)
    columns = {r["name"] for r in db.execute("PRAGMA table_info(wells)")}
    if "code" not in columns:  # migrate databases created before well IDs existed
        db.execute("ALTER TABLE wells ADD COLUMN code TEXT")
        db.execute("UPDATE wells SET code = name WHERE code IS NULL")
        db.execute("CREATE UNIQUE INDEX IF NOT EXISTS wells_code ON wells(code COLLATE NOCASE)")
        db.commit()
    if "easting" not in columns:  # migrate: well position on the field canvas (same units as the well map)
        db.execute("ALTER TABLE wells ADD COLUMN easting REAL")
        db.execute("ALTER TABLE wells ADD COLUMN northing REAL")
        db.commit()
    return db


# ---------- wells ----------

def _well(row, counts=None, results=None):
    out = dict(row)
    out["settings"] = json.loads(out["settings"])
    out["role"] = WELL_TYPES[out["type"]]["role"]
    out["lift"] = WELL_TYPES[out["type"]]["lift"]
    if counts is not None:
        out["counts"] = counts
    if results is not None:
        out["results"] = results
    return out


def _number(value, label):
    try:
        number = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        raise UserError(f"{label} must be a number.")
    if not math.isfinite(number):
        raise UserError(f"{label} must be a finite number.")
    return number


def _clean_well(body, existing=None):
    old = dict(existing) if existing else {}
    name = str(body.get("name", old.get("name", ""))).strip()
    if not name or len(name) > 80:
        raise UserError("Enter a well name of 1 to 80 characters.")
    code = str(body.get("code", old.get("code") or "") or name).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,39}", code):
        raise UserError("Well ID must be 1–40 letters, digits, spaces, dots, dashes or underscores.")
    kind = body.get("type", old.get("type"))
    if kind not in WELL_TYPES:
        raise UserError("Choose a well type.")
    if existing and kind != existing["type"] and WELL_TYPES[kind]["role"] != WELL_TYPES[existing["type"]]["role"]:
        raise UserError("A producer's lift can be added, switched or removed, but a well cannot change between producer and injector.")
    status = body.get("status", old.get("status", "active"))
    if status not in STATUSES:
        raise UserError("Choose a valid well status.")
    field = str(body.get("field", old.get("field", "")) or "").strip()[:80]
    notes = str(body.get("notes", old.get("notes", "")) or "").strip()[:2000]
    given = body.get("settings")
    if given is None:
        given = json.loads(old.get("settings", "{}"))  # keys that do not apply to a new lift type are dropped below
    settings = {}
    for spec in SETTINGS.get(kind, []):
        value = given.get(spec["key"], spec.get("default"))
        if value is None or value == "":
            if "default" in spec:
                value = spec["default"]
            else:
                continue
        value = _number(value, spec["label"])
        if value < spec.get("min", -math.inf) or value > spec.get("max", math.inf):
            raise UserError(f"{spec['label']} must be between {spec.get('min', '-∞')} and {spec.get('max', '∞')} {spec.get('unit', '')}.")
        settings[spec["key"]] = value
    for lo, hi, label in (("env_min_hz", "env_max_hz", "frequency"), ("env_min_gas", "env_max_gas", "lift gas")):
        if lo in settings and hi in settings and settings[lo] > settings[hi]:
            raise UserError(f"The minimum {label} is above the maximum.")
    position = {}
    for axis in ("easting", "northing"):
        value = body.get(axis, old.get(axis))
        position[axis] = None if value is None or value == "" else _number(value, axis.capitalize())
    if (position["easting"] is None) != (position["northing"] is None):
        raise UserError("Give both easting and northing, or neither.")
    return dict(name=name, code=code, type=kind, status=status, field=field, notes=notes, settings=json.dumps(settings), **position)


def _counts(db, well_id=None):
    counts = {}
    sql = ("SELECT r.well_id, r.dataset, COUNT(*) n, MIN(r.k) first, MAX(r.k) last FROM rows r"
           + (" WHERE r.well_id=?" if well_id else "") + " GROUP BY r.well_id, r.dataset")
    for r in db.execute(sql, (well_id,) if well_id else ()):
        counts.setdefault(r["well_id"], {})[r["dataset"]] = dict(rows=r["n"], first=r["first"], last=r["last"])
    for r in db.execute("SELECT well_id, dataset, MAX(version) v FROM versions" + (" WHERE well_id=?" if well_id else "")
                        + " GROUP BY well_id, dataset", (well_id,) if well_id else ()):
        entry = counts.setdefault(r["well_id"], {}).setdefault(r["dataset"], dict(rows=0, first=None, last=None))
        entry["version"] = r["v"]
    return counts


def list_wells(db, field=None):
    """All wells, or the wells of one field (field='' means wells with no field)."""
    counts = _counts(db)
    summaries = {}
    for r in db.execute("SELECT well_id, kind, created_at, payload FROM results WHERE well_id != ?", (FIELD,)):
        payload = json.loads(r["payload"])
        summaries.setdefault(r["well_id"], {})[r["kind"]] = dict(created_at=r["created_at"], run_id=payload.get("run_id"),
                                                                 warnings=payload.get("warnings", []), **payload.get("summary", {}))
    sql = "SELECT * FROM wells" + (" WHERE field=?" if field is not None else "") + " ORDER BY field COLLATE NOCASE, name COLLATE NOCASE"
    return [_well(r, counts.get(r["id"], {}), summaries.get(r["id"], {})) for r in db.execute(sql, (field,) if field is not None else ())]


def get_well(db, well_id):
    row = db.execute("SELECT * FROM wells WHERE id=?", (well_id,)).fetchone()
    if row is None:
        raise UserError("Well not found. It may have been deleted.", 404)
    return row


def well_detail(db, well_id):
    row = get_well(db, well_id)
    results = {r["kind"]: dict(created_at=r["created_at"], **json.loads(r["payload"]))
               for r in db.execute("SELECT kind, created_at, payload FROM results WHERE well_id=?", (well_id,))}
    return _well(row, _counts(db, well_id).get(well_id, {}), results)


def create_well(db, body):
    values = _clean_well(body)
    values.update(id=uuid.uuid4().hex[:12], created_at=now(), updated_at=now())
    try:
        with db:
            db.execute("INSERT INTO wells(id,name,code,type,field,status,notes,settings,easting,northing,created_at,updated_at) "
                       "VALUES(:id,:name,:code,:type,:field,:status,:notes,:settings,:easting,:northing,:created_at,:updated_at)", values)
    except sqlite3.IntegrityError:
        raise UserError(f"A well with the name {values['name']} or ID {values['code']} already exists.")
    return well_detail(db, values["id"])


def update_well(db, well_id, body):
    existing = get_well(db, well_id)
    values = _clean_well(body, existing)
    values.update(id=well_id, updated_at=now())
    try:
        with db:
            db.execute("UPDATE wells SET name=:name, code=:code, type=:type, field=:field, status=:status, notes=:notes, "
                       "settings=:settings, easting=:easting, northing=:northing, updated_at=:updated_at WHERE id=:id", values)
            if values["type"] != existing["type"]:  # cached results of the old lift no longer describe this well
                db.execute("DELETE FROM results WHERE well_id=?", (well_id,))
    except sqlite3.IntegrityError:
        raise UserError(f"A well with the name {values['name']} or ID {values['code']} already exists.")
    return well_detail(db, well_id)


def delete_well(db, well_id):
    get_well(db, well_id)
    with db:
        db.execute("DELETE FROM wells WHERE id=?", (well_id,))
        db.execute("DELETE FROM results WHERE well_id=?", (well_id,))
        db.execute("DELETE FROM versions WHERE well_id=?", (well_id,))
    return {"deleted": well_id}


def find_well(db, ident):
    """Match a well by ID code or name, case-insensitively."""
    return db.execute("SELECT * FROM wells WHERE code=? COLLATE NOCASE OR name=? COLLATE NOCASE",
                      (str(ident).strip(), str(ident).strip())).fetchone()


# ---------- datasets ----------

def dataset_for(well, dataset):
    if dataset not in DATASETS or dataset not in WELL_TYPES[well["type"]]["datasets"]:
        raise UserError(f"{WELL_TYPES[well['type']]['label']} wells do not use this dataset.", 404)
    return DATASETS[dataset]


def parse_time(value, kind):
    """Return (stored text, sort seconds) for a date or UTC timestamp."""
    if isinstance(value, datetime):
        stamp = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("is empty")
        try:
            stamp = datetime.fromisoformat(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a valid {'date (YYYY-MM-DD)' if kind == 'date' else 'timestamp (YYYY-MM-DD HH:MM:SS)'}")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)
    if kind == "date":
        day = stamp.date()
        return day.isoformat(), datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    return stamp.isoformat(timespec="seconds").replace("+00:00", "Z"), stamp.timestamp()


_parse_time = parse_time  # backwards-compatible name


def _blank(value):
    return value is None or (isinstance(value, float) and math.isnan(value)) or (isinstance(value, str) and not value.strip())


def clean_row(spec, raw):
    """Validate one row. Returns (key text, sort key, data dict) or raises ValueError with a readable reason."""
    data = {}
    for col in spec["columns"]:
        value = raw.get(col["key"])
        if _blank(value):
            if col["required"]:
                raise ValueError(f"{col['label']} is required")
            data[col["key"]] = None
            continue
        kind = col["kind"]
        if kind in ("date", "time"):
            data[col["key"]] = parse_time(value, kind)
        elif kind == "text":
            data[col["key"]] = str(value).strip()[:500]
        elif kind == "choice":
            text = str(value).strip().lower().replace(" ", "_")
            text = {"true": "yes", "1": "yes", "y": "yes", "false": "no", "0": "no", "n": "no"}.get(text, text) \
                if col["options"] == ["yes", "no"] else text
            if text not in col["options"]:
                raise ValueError(f"{col['label']} '{value}' must be one of: {', '.join(col['options'])}")
            data[col["key"]] = text
        else:
            try:
                number = float(str(value).replace(",", "")) if isinstance(value, str) else float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{col['label']} '{value}' is not a number")
            if not math.isfinite(number):
                raise ValueError(f"{col['label']} must be a finite number")
            if "min" in col and number < col["min"]:
                raise ValueError(f"{col['label']} {number:g} is below {col['min']:g} {col.get('unit', '')}".strip())
            if "max" in col and number > col["max"]:
                raise ValueError(f"{col['label']} {number:g} is above {col['max']:g} {col.get('unit', '')}".strip())
            data[col["key"]] = number
    key = spec["key"]
    if isinstance(data[key], tuple):
        text, sort = data[key]
        data[key] = text
    else:
        sort = data[key]
        text = repr(float(sort))
    return text, sort, data


def _bump(db, well_id, dataset, source):
    rows = db.execute("SELECT COUNT(*) FROM rows WHERE well_id=? AND dataset=?", (well_id, dataset)).fetchone()[0]
    version = (db.execute("SELECT MAX(version) FROM versions WHERE well_id=? AND dataset=?", (well_id, dataset)).fetchone()[0] or 0) + 1
    db.execute("INSERT INTO versions(well_id,dataset,version,created_at,source,rows) VALUES(?,?,?,?,?,?)",
               (well_id, dataset, version, now(), source[:200], rows))
    db.execute("UPDATE wells SET updated_at=? WHERE id=?", (now(), well_id))
    return version


def upsert_rows(db, well_id, dataset, rows, replace_id=None, source="manual entry"):
    """Insert or update rows keyed by the dataset key. Returns counts and per-row problems."""
    well = get_well(db, well_id)
    spec = dataset_for(well, dataset)
    inserted = updated = 0
    problems = []
    with db:
        for index, raw in enumerate(rows):
            line = raw.pop("_line", index + 1) if isinstance(raw, dict) else index + 1
            try:
                k, sk, data = clean_row(spec, raw)
            except ValueError as error:
                problems.append(f"Row {line}: {error}")
                continue
            if replace_id is not None:
                clash = db.execute("SELECT id FROM rows WHERE well_id=? AND dataset=? AND k=? AND id!=?",
                                   (well_id, dataset, k, replace_id)).fetchone()
                if clash:
                    raise UserError(f"Another row already uses {data[spec['key']]}.")
                cur = db.execute("UPDATE rows SET k=?, sk=?, data=? WHERE id=? AND well_id=? AND dataset=?",
                                 (k, sk, json.dumps(data), replace_id, well_id, dataset))
                if cur.rowcount == 0:
                    raise UserError("Row not found. Refresh the table.", 404)
                updated += 1
                continue
            exists = db.execute("SELECT 1 FROM rows WHERE well_id=? AND dataset=? AND k=?", (well_id, dataset, k)).fetchone()
            db.execute("INSERT INTO rows(well_id,dataset,k,sk,data) VALUES(?,?,?,?,?) "
                       "ON CONFLICT(well_id,dataset,k) DO UPDATE SET sk=excluded.sk, data=excluded.data",
                       (well_id, dataset, k, sk, json.dumps(data)))
            if exists:
                updated += 1
            else:
                inserted += 1
        version = _bump(db, well_id, dataset, source) if inserted or updated else None
    return {"inserted": inserted, "updated": updated, "rejected": len(problems), "problems": problems[:50], "version": version}


def delete_rows(db, well_id, dataset, ids=None, all_rows=False):
    well = get_well(db, well_id)
    dataset_for(well, dataset)
    with db:
        if all_rows:
            cur = db.execute("DELETE FROM rows WHERE well_id=? AND dataset=?", (well_id, dataset))
        else:
            ids = [int(i) for i in ids or []]
            if not ids:
                raise UserError("Select at least one row to delete.")
            marks = ",".join("?" * len(ids))
            cur = db.execute(f"DELETE FROM rows WHERE well_id=? AND dataset=? AND id IN ({marks})", (well_id, dataset, *ids))
        if cur.rowcount:
            _bump(db, well_id, dataset, "cleared" if all_rows else f"deleted {cur.rowcount} rows")
    return {"deleted": cur.rowcount}


def read_rows(db, well_id, dataset):
    """All rows in key order as plain dicts (for models and exports)."""
    return [json.loads(r["data"]) for r in db.execute(
        "SELECT data FROM rows WHERE well_id=? AND dataset=? ORDER BY sk", (well_id, dataset))]


def current_version(db, well_id, dataset):
    return db.execute("SELECT MAX(version) FROM versions WHERE well_id=? AND dataset=?", (well_id, dataset)).fetchone()[0] or 0


def page_rows(db, well_id, dataset, offset=0, limit=100):
    well = get_well(db, well_id)
    spec = dataset_for(well, dataset)
    offset, limit = max(0, int(offset)), min(max(1, int(limit)), 1000)
    total = db.execute("SELECT COUNT(*) FROM rows WHERE well_id=? AND dataset=?", (well_id, dataset)).fetchone()[0]
    newest_first = spec["columns"][0]["kind"] in ("date", "time")
    page = [dict(id=r["id"], **json.loads(r["data"])) for r in db.execute(
        f"SELECT id, data FROM rows WHERE well_id=? AND dataset=? ORDER BY sk {'DESC' if newest_first else 'ASC'} LIMIT ? OFFSET ?",
        (well_id, dataset, limit, offset))]
    step = max(1, math.ceil(total / 1500))
    series = [json.loads(r["data"]) for r in db.execute(
        "SELECT data FROM (SELECT data, sk, ROW_NUMBER() OVER (ORDER BY sk) - 1 AS n FROM rows "
        "WHERE well_id=? AND dataset=?) WHERE n % ? = 0 ORDER BY sk", (well_id, dataset, step))]
    keep = [spec["key"], *spec["chart"]]
    history = [dict(r) for r in db.execute("SELECT version, created_at, source, rows FROM versions WHERE well_id=? AND dataset=? "
                                           "ORDER BY version DESC LIMIT 10", (well_id, dataset))]
    return {"total": total, "offset": offset, "limit": limit, "rows": page, "newest_first": newest_first,
            "series": [{k: r.get(k) for k in keep} for r in series], "versions": history}


def dataset_registry(db, field=None):
    sql = ("SELECT w.id well_id, w.name, w.code, w.field, v.dataset, v.version, v.created_at, v.source, v.rows FROM versions v "
           "JOIN wells w ON w.id = v.well_id WHERE v.version = (SELECT MAX(version) FROM versions x WHERE x.well_id=v.well_id "
           "AND x.dataset=v.dataset)" + (" AND w.field=?" if field else "") + " ORDER BY w.name COLLATE NOCASE, v.dataset")
    return [dict(r, label=DATASETS[r["dataset"]]["label"]) for r in db.execute(sql, (field,) if field else ())]


def quality(db, well_id):
    """Data-quality summary per dataset: coverage, gaps, missing values."""
    well = get_well(db, well_id)
    report = []
    for dataset in WELL_TYPES[well["type"]]["datasets"]:
        spec = DATASETS[dataset]
        rows = read_rows(db, well_id, dataset)
        entry = dict(dataset=dataset, label=spec["label"], rows=len(rows), version=current_version(db, well_id, dataset), issues=[])
        if rows and spec["columns"][0]["kind"] in ("date", "time"):
            key = spec["key"]
            stamps = [parse_time(r[key], spec["columns"][0]["kind"])[1] for r in rows]
            entry["first"], entry["last"] = rows[0][key], rows[-1][key]
            if len(stamps) > 2:
                steps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
                typical = steps[len(steps) // 2]
                gaps = [(rows[i][key], rows[i + 1][key]) for i, (a, b) in enumerate(zip(stamps, stamps[1:])) if typical and b - a > 3 * typical]
                entry["typical_step_s"] = typical
                if gaps:
                    entry["issues"].append(f"{len(gaps)} gap(s) longer than 3× the usual spacing, first {gaps[0][0]} → {gaps[0][1]}")
        for col in spec["columns"]:
            if not col["required"] and col["kind"] == "number" and rows:
                present = sum(r.get(col["key"]) is not None for r in rows)
                if present < len(rows):
                    entry.setdefault("optional_coverage", {})[col["label"]] = round(present / len(rows), 3)
        if not rows:
            entry["issues"].append("No data")
        report.append(entry)
    return report


# ---------- latest-result cache (well cards) ----------

def save_result(db, well_id, kind, payload):
    with db:
        db.execute("INSERT INTO results(well_id,kind,created_at,payload) VALUES(?,?,?,?) "
                   "ON CONFLICT(well_id,kind) DO UPDATE SET created_at=excluded.created_at, payload=excluded.payload",
                   (well_id, kind, now(), json.dumps(payload, allow_nan=False)))


def drop_result(db, well_id, kind):
    with db:
        db.execute("DELETE FROM results WHERE well_id=? AND kind=?", (well_id, kind))


def get_result(db, well_id, kind):
    row = db.execute("SELECT created_at, payload FROM results WHERE well_id=? AND kind=?", (well_id, kind)).fetchone()
    return dict(created_at=row["created_at"], **json.loads(row["payload"])) if row else None


# ---------- scenarios ----------

def _scenario(row):
    out = dict(row)
    out["facility"] = json.loads(out["facility"])
    return out


def _clean_facility(given):
    facility = {}
    for spec in FACILITY:
        value = (given or {}).get(spec["key"], spec.get("default"))
        if value is None or value == "":
            continue
        value = _number(value, spec["label"])
        if value < 0:
            raise UserError(f"{spec['label']} cannot be negative.")
        facility[spec["key"]] = value
    return facility


def list_scenarios(db, field=None):
    ensure_base(db, field or "")
    sql = "SELECT * FROM scenarios" + (" WHERE field=?" if field is not None else "") + " ORDER BY field, created_at"
    return [_scenario(r) for r in db.execute(sql, (field,) if field is not None else ())]


def ensure_base(db, field):
    if not db.execute("SELECT 1 FROM scenarios WHERE field=?", (field,)).fetchone():
        with db:
            db.execute("INSERT OR IGNORE INTO scenarios(id,name,field,notes,facility,policy,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                       (uuid.uuid4().hex[:12], "Base", field, "Current facility limits", json.dumps(_clean_facility({})),
                        "hold", now(), now()))


def get_scenario(db, scenario_id):
    row = db.execute("SELECT * FROM scenarios WHERE id=?", (scenario_id,)).fetchone()
    if row is None:
        raise UserError("Scenario not found.", 404)
    return _scenario(row)


def save_scenario(db, body, scenario_id=None):
    old = get_scenario(db, scenario_id) if scenario_id else {}
    source = get_scenario(db, body["copy_from"]) if body.get("copy_from") else {}
    name = str(body.get("name", old.get("name", "")) or "").strip()
    if not name or len(name) > 60:
        raise UserError("Enter a scenario name of 1 to 60 characters.")
    field = old.get("field", body.get("field", source.get("field", "")) or "")
    policy = body.get("policy", old.get("policy", source.get("policy", "hold")))
    if policy not in POLICIES:
        raise UserError("Choose a failure policy.")
    facility = _clean_facility(body.get("facility", old.get("facility", source.get("facility", {}))))
    notes = str(body.get("notes", old.get("notes", source.get("notes", ""))) or "")[:1000]
    try:
        with db:
            if scenario_id:
                db.execute("UPDATE scenarios SET name=?, notes=?, facility=?, policy=?, updated_at=? WHERE id=?",
                           (name, notes, json.dumps(facility), policy, now(), scenario_id))
            else:
                scenario_id = uuid.uuid4().hex[:12]
                db.execute("INSERT INTO scenarios(id,name,field,notes,facility,policy,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                           (scenario_id, name, field, notes, json.dumps(facility), policy, now(), now()))
    except sqlite3.IntegrityError:
        raise UserError(f"A scenario named {name} already exists in this field.")
    return get_scenario(db, scenario_id)


def delete_scenario(db, scenario_id):
    s = get_scenario(db, scenario_id)
    if db.execute("SELECT COUNT(*) FROM scenarios WHERE field=?", (s["field"],)).fetchone()[0] <= 1:
        raise UserError("A field needs at least one scenario.")
    with db:
        db.execute("DELETE FROM scenarios WHERE id=?", (scenario_id,))
    return {"deleted": scenario_id}


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "well"
