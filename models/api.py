"""Command entry point for the Rust API.

Usage: python models/api.py <command>  (JSON request on stdin, one JSON reply on stdout)
Exit 0 = success, 2 = user error ({"error", "status"}), 1 = internal failure (details on stderr).
"""
import json
import sys
import traceback

REPLY = sys.stdout
sys.stdout = sys.stderr  # model libraries print progress; keep stdout for the reply only


def main():
    import store
    from store import UserError
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    text = sys.stdin.read()
    body = json.loads(text) if text.strip() else {}
    db = store.connect()

    def req(key):
        if body.get(key) in (None, ""):
            raise UserError(f"Missing {key} in request")
        return body[key]

    def mod(name):
        return __import__(name)

    def fields():
        rows = db.execute("SELECT field, COUNT(*) n, SUM(type='injector') inj FROM wells GROUP BY field ORDER BY field COLLATE NOCASE")
        out = {r["field"]: dict(name=r["field"], wells=r["n"], injectors=r["inj"], producers=r["n"] - r["inj"]) for r in rows}
        mod("crmdata")._table(db)
        for r in db.execute("SELECT DISTINCT field FROM crm_data"):  # fields that only have a CRM dataset
            out.setdefault(r["field"], dict(name=r["field"], wells=0, injectors=0, producers=0))
        return sorted(out.values(), key=lambda f: f["name"].lower())

    commands = {
        "schema": lambda: mod("schema").public(),
        "fields.list": fields,
        "wells.list": lambda: store.list_wells(db, body["field"] if "field" in body else None),
        "wells.get": lambda: store.well_detail(db, req("id")),
        "wells.create": lambda: store.create_well(db, body),
        "wells.update": lambda: store.update_well(db, req("id"), body),
        "wells.delete": lambda: store.delete_well(db, req("id")),
        "wells.quality": lambda: store.quality(db, req("id")),
        "rows.page": lambda: store.page_rows(db, req("id"), req("dataset"), body.get("offset", 0), body.get("limit", 100)),
        "rows.add": lambda: store.upsert_rows(db, req("id"), req("dataset"), body.get("rows", [])),
        "rows.update": lambda: store.upsert_rows(db, req("id"), req("dataset"), [body.get("row", {})], replace_id=int(req("row_id")),
                                                 source="manual edit"),
        "rows.delete": lambda: store.delete_rows(db, req("id"), req("dataset"), body.get("ids"), bool(body.get("all"))),
        "rows.import": lambda: mod("excel").import_file(db, req("id"), req("dataset"), req("path"), body.get("filename", "")),
        "rows.export": lambda: mod("excel").export_dataset(db, req("id"), req("dataset"), body.get("format", "csv")),
        "template": lambda: mod("excel").template(req("dataset"), body.get("format", "xlsx")),
        "datasets.list": lambda: store.dataset_registry(db, body.get("field") or None),
        "imports.preview": lambda: mod("intake").preview(db, req("kind"), req("path"), body.get("filename", ""), body.get("field", "")),
        "imports.commit": lambda: mod("intake").commit(db, req("token"), body),
        "scenarios.list": lambda: store.list_scenarios(db, body.get("field", "")),
        "scenarios.create": lambda: store.save_scenario(db, body),
        "scenarios.update": lambda: store.save_scenario(db, body, req("id")),
        "scenarios.delete": lambda: store.delete_scenario(db, req("id")),
        "runs.list": lambda: mod("runs").list_runs(db, body),
        "runs.get": lambda: mod("runs").get(db, req("id")),
        "runs.create": lambda: mod("runs").create(db, body),
        "runs.execute": lambda: mod("runs").execute(db, req("id")),
        "runs.fail": lambda: mod("runs").fail(db, req("id"), body.get("error") or "The run failed."),
        "runs.recover": lambda: mod("runs").recover(db),
        "runs.rename": lambda: mod("runs").rename(db, req("id"), body),
        "runs.delete": lambda: mod("runs").delete(db, req("id")),
        "runs.export": lambda: mod("excel").export_run(db, mod("runs").get(db, req("id")), body.get("format", "xlsx")),
        "well.export": lambda: mod("excel").export_well(db, req("id")),
        "field.export": lambda: mod("excel").export_field(db, body.get("field") or None),
        "crm.get": lambda: mod("crmdata").get(db, body.get("field", "") or ""),
        "crm.stage": lambda: mod("crmdata").stage(db, body.get("field", "") or "", req("kind"), req("path")),
        "crm.unstage": lambda: mod("crmdata").unstage(body.get("field", "") or ""),
        "crm.commit": lambda: mod("crmdata").commit(db, body),
        "crm.column": lambda: mod("crmdata").set_column(db, body),
        "crm.staged_data": lambda: mod("crmdata").staged_data(body.get("field", "") or "", req("kind")),
        "crm.mpi": lambda: mod("crmdata").mpi(db, body),
        "crm.bl": lambda: mod("crmdata").buckley_leverett(db, body),
        "crm.aquifer": lambda: mod("crmdata").aquifer(db, body),
        "demo.create": lambda: mod("demo").create(db),
        "demo.remove": lambda: mod("demo").remove(db),
    }
    if command not in commands:
        raise UserError(f"Unknown command {command}", 404)
    return commands[command]()


if __name__ == "__main__":
    try:
        result = main()
        REPLY.write(json.dumps(result, allow_nan=False, default=str))
        sys.exit(0)
    except Exception as error:
        from store import UserError
        if isinstance(error, UserError):
            REPLY.write(json.dumps({"error": str(error), "status": error.status}))
            sys.exit(2)
        traceback.print_exc()
        sys.exit(1)
