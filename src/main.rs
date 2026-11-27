use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post, put},
    Json, Router,
};
use serde_json::{json, Map, Value};
use std::{
    collections::HashMap,
    env,
    path::PathBuf,
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{fs, io::AsyncWriteExt, process::Command, sync::Semaphore};

struct App {
    root: PathBuf,
    runtime: PathBuf,
    python: String,
    jobs: Semaphore,
    uploads: AtomicU64,
}
type Shared = Arc<App>;
type ApiError = (StatusCode, Json<Value>);
type ApiResult = Result<Json<Value>, ApiError>;
type Params = Query<HashMap<String, String>>;

const QUICK: u64 = 60;
const IMPORT: u64 = 300;
const HEAVY: u64 = 1800;
const MAX_UPLOAD: usize = 64 * 1024 * 1024;

fn error(code: StatusCode, message: &str) -> ApiError {
    (code, Json(json!({ "error": message })))
}

/// Run `models/api.py <command>` with a JSON request on stdin.
/// Exit 0 returns the reply, exit 2 is a user error with its own message and status, anything else is a 500.
async fn py(app: &App, command: &str, body: Value, secs: u64) -> Result<Value, ApiError> {
    let mut cmd = Command::new(&app.python);
    cmd.current_dir(&app.root)
        .args(["models/api.py", command])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "Python worker unavailable. Check FIELD_PYTHON."))?;
    let input = serde_json::to_vec(&body).unwrap_or_default();
    if let Some(mut stdin) = child.stdin.take() {
        // The worker reads all of stdin before writing anything, so this cannot deadlock.
        stdin
            .write_all(&input)
            .await
            .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "Python worker closed its input"))?;
    }
    // On timeout the future (and child) is dropped, and kill_on_drop stops the process.
    let output = tokio::time::timeout(Duration::from_secs(secs), child.wait_with_output())
        .await
        .map_err(|_| error(StatusCode::GATEWAY_TIMEOUT, "The request timed out."))?
        .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "Python worker unavailable"))?;
    match output.status.code() {
        Some(0) => serde_json::from_slice(&output.stdout)
            .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "The worker returned an invalid reply")),
        Some(2) => {
            let reply: Value = serde_json::from_slice(&output.stdout).unwrap_or(json!({}));
            let code = reply["status"]
                .as_u64()
                .and_then(|c| StatusCode::from_u16(c as u16).ok())
                .unwrap_or(StatusCode::BAD_REQUEST);
            Err(error(code, reply["error"].as_str().unwrap_or("The request was rejected")))
        }
        _ => {
            eprintln!("[{command}] {}", String::from_utf8_lossy(&output.stderr));
            Err(error(StatusCode::INTERNAL_SERVER_ERROR, "The model worker failed. Details are in runtime/api-errors.log."))
        }
    }
}

async fn quick(app: &App, command: &str, body: Value) -> ApiResult {
    py(app, command, body, QUICK).await.map(Json)
}

fn check_id(value: &str) -> Result<(), ApiError> {
    let ok = !value.is_empty()
        && value.len() <= 64
        && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok { Ok(()) } else { Err(error(StatusCode::BAD_REQUEST, "Invalid identifier")) }
}

/// Query parameters become a JSON object for the worker, merged with fixed fields.
fn with_query(query: HashMap<String, String>, extra: Value) -> Value {
    let mut map: Map<String, Value> = query.into_iter().map(|(k, v)| (k, Value::String(v))).collect();
    if let Value::Object(fields) = extra {
        map.extend(fields);
    }
    Value::Object(map)
}

/// Send a file the worker wrote under the runtime folder.
async fn send_file(app: &App, reply: Value) -> Result<Response, ApiError> {
    let path = PathBuf::from(reply["path"].as_str().unwrap_or_default());
    let inside = match (fs::canonicalize(&path).await, fs::canonicalize(&app.runtime).await) {
        (Ok(file), Ok(base)) => file.starts_with(base),
        _ => false,
    };
    if !inside {
        return Err(error(StatusCode::INTERNAL_SERVER_ERROR, "Export file is missing"));
    }
    let bytes = fs::read(&path)
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "Export file is missing"))?;
    let name: String = reply["filename"]
        .as_str()
        .unwrap_or("export.xlsx")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || " ._-".contains(c) { c } else { '-' })
        .collect();
    let kind = if name.ends_with(".csv") {
        "text/csv; charset=utf-8"
    } else {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    };
    Ok((
        [
            (header::CONTENT_TYPE, kind.to_string()),
            (header::CONTENT_DISPOSITION, format!("attachment; filename=\"{name}\"")),
            (header::CACHE_CONTROL, "no-store".to_string()),
        ],
        bytes,
    )
        .into_response())
}

/// Store an upload under runtime/uploads and return its path; the caller removes it.
async fn store_upload(app: &App, headers: &HeaderMap, body: &Bytes) -> Result<(PathBuf, String), ApiError> {
    if body.is_empty() {
        return Err(error(StatusCode::BAD_REQUEST, "The uploaded file is empty"));
    }
    let filename = headers.get("x-filename").and_then(|v| v.to_str().ok()).unwrap_or("upload.csv").to_string();
    let ext = filename.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    if !["xlsx", "xlsm", "xls", "csv", "txt", "tsv"].contains(&ext.as_str()) {
        return Err(error(StatusCode::BAD_REQUEST, "Upload a CSV (.csv) or Excel (.xlsx) file."));
    }
    let folder = app.runtime.join("uploads");
    fs::create_dir_all(&folder)
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "Cannot store the upload"))?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let path = folder.join(format!("upload-{stamp}-{}.{ext}", app.uploads.fetch_add(1, Ordering::Relaxed)));
    fs::write(&path, body)
        .await
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "Cannot store the upload"))?;
    Ok((path, filename))
}

// ---------- registry: fields, wells, datasets ----------

async fn schema(State(app): State<Shared>) -> ApiResult {
    quick(&app, "schema", json!({})).await
}
async fn fields(State(app): State<Shared>) -> ApiResult {
    quick(&app, "fields.list", json!({})).await
}
async fn list_wells(State(app): State<Shared>, Query(q): Params) -> ApiResult {
    quick(&app, "wells.list", with_query(q, json!({}))).await
}
async fn create_well(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    quick(&app, "wells.create", body).await
}
async fn get_well(State(app): State<Shared>, Path(id): Path<String>) -> ApiResult {
    check_id(&id)?;
    quick(&app, "wells.get", json!({ "id": id })).await
}
async fn update_well(State(app): State<Shared>, Path(id): Path<String>, Json(mut body): Json<Value>) -> ApiResult {
    check_id(&id)?;
    body["id"] = json!(id);
    quick(&app, "wells.update", body).await
}
async fn delete_well(State(app): State<Shared>, Path(id): Path<String>) -> ApiResult {
    check_id(&id)?;
    quick(&app, "wells.delete", json!({ "id": id })).await
}
async fn well_quality(State(app): State<Shared>, Path(id): Path<String>) -> ApiResult {
    check_id(&id)?;
    quick(&app, "wells.quality", json!({ "id": id })).await
}
async fn export_well(State(app): State<Shared>, Path(id): Path<String>) -> Result<Response, ApiError> {
    check_id(&id)?;
    let reply = py(&app, "well.export", json!({ "id": id }), IMPORT).await?;
    send_file(&app, reply).await
}
async fn datasets(State(app): State<Shared>, Query(q): Params) -> ApiResult {
    quick(&app, "datasets.list", with_query(q, json!({}))).await
}

async fn page_rows(State(app): State<Shared>, Path((id, dataset)): Path<(String, String)>, Query(q): Params) -> ApiResult {
    check_id(&id)?;
    check_id(&dataset)?;
    let offset: u64 = q.get("offset").and_then(|v| v.parse().ok()).unwrap_or(0);
    let limit: u64 = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(100);
    quick(&app, "rows.page", json!({ "id": id, "dataset": dataset, "offset": offset, "limit": limit })).await
}
async fn add_rows(State(app): State<Shared>, Path((id, dataset)): Path<(String, String)>, Json(body): Json<Value>) -> ApiResult {
    check_id(&id)?;
    check_id(&dataset)?;
    let rows = body.get("rows").cloned().unwrap_or(json!([]));
    quick(&app, "rows.add", json!({ "id": id, "dataset": dataset, "rows": rows })).await
}
async fn update_row(
    State(app): State<Shared>,
    Path((id, dataset, row_id)): Path<(String, String, u64)>,
    Json(body): Json<Value>,
) -> ApiResult {
    check_id(&id)?;
    check_id(&dataset)?;
    let row = body.get("row").cloned().unwrap_or(json!({}));
    quick(&app, "rows.update", json!({ "id": id, "dataset": dataset, "row_id": row_id, "row": row })).await
}
async fn delete_rows(State(app): State<Shared>, Path((id, dataset)): Path<(String, String)>, Json(body): Json<Value>) -> ApiResult {
    check_id(&id)?;
    check_id(&dataset)?;
    quick(&app, "rows.delete", json!({ "id": id, "dataset": dataset, "ids": body.get("ids"), "all": body.get("all") })).await
}
async fn upload_rows(
    State(app): State<Shared>,
    Path((id, dataset)): Path<(String, String)>,
    headers: HeaderMap,
    body: Bytes,
) -> ApiResult {
    check_id(&id)?;
    check_id(&dataset)?;
    let (path, filename) = store_upload(&app, &headers, &body).await?;
    let request = json!({ "id": id, "dataset": dataset, "path": path.to_string_lossy(), "filename": filename });
    let result = py(&app, "rows.import", request, IMPORT).await;
    let _ = fs::remove_file(&path).await;
    result.map(Json)
}
async fn export_rows(State(app): State<Shared>, Path((id, dataset)): Path<(String, String)>, Query(q): Params) -> Result<Response, ApiError> {
    check_id(&id)?;
    check_id(&dataset)?;
    let reply = py(&app, "rows.export", with_query(q, json!({ "id": id, "dataset": dataset })), IMPORT).await?;
    send_file(&app, reply).await
}
async fn template(State(app): State<Shared>, Path(dataset): Path<String>, Query(q): Params) -> Result<Response, ApiError> {
    check_id(&dataset)?;
    let reply = py(&app, "template", with_query(q, json!({ "dataset": dataset })), QUICK).await?;
    send_file(&app, reply).await
}

// ---------- staged CSV/Excel intake ----------

async fn import_preview(State(app): State<Shared>, Path(kind): Path<String>, Query(q): Params, headers: HeaderMap, body: Bytes) -> ApiResult {
    check_id(&kind)?;
    let (path, filename) = store_upload(&app, &headers, &body).await?;
    let request = with_query(q, json!({ "kind": kind, "path": path.to_string_lossy(), "filename": filename }));
    let result = py(&app, "imports.preview", request, IMPORT).await; // the worker keeps its own staged copy
    let _ = fs::remove_file(&path).await;
    result.map(Json)
}
async fn import_commit(State(app): State<Shared>, Path(token): Path<String>, Json(mut body): Json<Value>) -> ApiResult {
    check_id(&token)?;
    body["token"] = json!(token);
    py(&app, "imports.commit", body, IMPORT).await.map(Json)
}

// ---------- scenarios ----------

async fn list_scenarios(State(app): State<Shared>, Query(q): Params) -> ApiResult {
    quick(&app, "scenarios.list", with_query(q, json!({}))).await
}
async fn create_scenario(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    quick(&app, "scenarios.create", body).await
}
async fn update_scenario(State(app): State<Shared>, Path(id): Path<String>, Json(mut body): Json<Value>) -> ApiResult {
    check_id(&id)?;
    body["id"] = json!(id);
    quick(&app, "scenarios.update", body).await
}
async fn delete_scenario(State(app): State<Shared>, Path(id): Path<String>) -> ApiResult {
    check_id(&id)?;
    quick(&app, "scenarios.delete", json!({ "id": id })).await
}

// ---------- runs (background jobs) ----------

async fn list_runs(State(app): State<Shared>, Query(q): Params) -> ApiResult {
    quick(&app, "runs.list", with_query(q, json!({}))).await
}
async fn get_run(State(app): State<Shared>, Path(id): Path<String>) -> ApiResult {
    check_id(&id)?;
    quick(&app, "runs.get", json!({ "id": id })).await
}
/// Validate and queue a run, return it immediately, and execute it in the background.
/// At most two runs execute at once; the rest wait their turn.
async fn create_run(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    let run = py(&app, "runs.create", body, QUICK).await?;
    let id = run["id"].as_str().unwrap_or_default().to_string();
    let worker = app.clone();
    tokio::spawn(async move {
        let _slot = worker.jobs.acquire().await;
        if let Err((_, Json(reply))) = py(&worker, "runs.execute", json!({ "id": id }), HEAVY).await {
            let message = reply["error"].as_str().unwrap_or("The run failed").to_string();
            let _ = py(&worker, "runs.fail", json!({ "id": id, "error": message }), QUICK).await;
        }
    });
    Ok(Json(run))
}
async fn rename_run(State(app): State<Shared>, Path(id): Path<String>, Json(mut body): Json<Value>) -> ApiResult {
    check_id(&id)?;
    body["id"] = json!(id);
    quick(&app, "runs.rename", body).await
}
async fn delete_run(State(app): State<Shared>, Path(id): Path<String>) -> ApiResult {
    check_id(&id)?;
    quick(&app, "runs.delete", json!({ "id": id })).await
}
async fn export_run(State(app): State<Shared>, Path(id): Path<String>, Query(q): Params) -> Result<Response, ApiError> {
    check_id(&id)?;
    let reply = py(&app, "runs.export", with_query(q, json!({ "id": id })), IMPORT).await?;
    send_file(&app, reply).await
}

// ---------- Deep-CRM data (pywaterflood layout) and companion models ----------

async fn crm_get(State(app): State<Shared>, Query(q): Params) -> ApiResult {
    quick(&app, "crm.get", with_query(q, json!({}))).await
}
async fn crm_upload(State(app): State<Shared>, Path(kind): Path<String>, Query(q): Params, headers: HeaderMap, body: Bytes) -> ApiResult {
    check_id(&kind)?;
    let (path, _) = store_upload(&app, &headers, &body).await?;
    let request = with_query(q, json!({ "kind": kind, "path": path.to_string_lossy() }));
    let result = py(&app, "crm.stage", request, IMPORT).await; // the worker keeps its own staged copy
    let _ = fs::remove_file(&path).await;
    result.map(Json)
}
async fn crm_unstage(State(app): State<Shared>, Query(q): Params) -> ApiResult {
    quick(&app, "crm.unstage", with_query(q, json!({}))).await
}
async fn crm_staged_data(State(app): State<Shared>, Path(kind): Path<String>, Query(q): Params) -> ApiResult {
    check_id(&kind)?;
    quick(&app, "crm.staged_data", with_query(q, json!({ "kind": kind }))).await
}
async fn crm_column(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    quick(&app, "crm.column", body).await
}
async fn crm_commit(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    py(&app, "crm.commit", body, IMPORT).await.map(Json)
}
async fn crm_mpi(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    quick(&app, "crm.mpi", body).await
}
async fn crm_bl(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    quick(&app, "crm.bl", body).await
}
async fn crm_aquifer(State(app): State<Shared>, Json(body): Json<Value>) -> ApiResult {
    py(&app, "crm.aquifer", body, IMPORT).await.map(Json)
}

// ---------- field report, demo, reference replay ----------

async fn field_export(State(app): State<Shared>, Query(q): Params) -> Result<Response, ApiError> {
    let reply = py(&app, "field.export", with_query(q, json!({})), IMPORT).await?;
    send_file(&app, reply).await
}
async fn demo_create(State(app): State<Shared>) -> ApiResult {
    py(&app, "demo.create", json!({}), IMPORT).await.map(Json)
}
async fn demo_remove(State(app): State<Shared>) -> ApiResult {
    quick(&app, "demo.remove", json!({})).await
}
async fn overview(State(app): State<Shared>) -> ApiResult {
    let bytes = fs::read(app.runtime.join("snapshot.json"))
        .await
        .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "Reference runs are loading. Refresh in a moment."))?;
    serde_json::from_slice(&bytes)
        .map(Json)
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "Invalid replay snapshot"))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = env::current_dir()?;
    let runtime = env::var("FIELD_RUNTIME").map(PathBuf::from).unwrap_or_else(|_| root.join("runtime"));
    let app = Arc::new(App {
        root,
        runtime,
        python: env::var("FIELD_PYTHON").unwrap_or("python".into()),
        jobs: Semaphore::new(2),
        uploads: AtomicU64::new(0),
    });
    // Runs left queued or running by a previous process can never finish.
    if let Err((_, Json(e))) = py(&app, "runs.recover", json!({}), QUICK).await {
        eprintln!("Could not recover interrupted runs: {e}");
    }
    let worker = app.clone();
    tokio::spawn(async move {
        loop {
            let mut cmd = Command::new(&worker.python);
            cmd.current_dir(&worker.root).arg("models/replay.py").kill_on_drop(true);
            match tokio::time::timeout(Duration::from_secs(300), cmd.status()).await {
                Ok(Ok(status)) if status.success() => {}
                Ok(_) => eprintln!("Replay refresh failed; retaining last completed snapshot"),
                Err(_) => eprintln!("Replay refresh timed out after 300 s; retaining last completed snapshot"),
            }
            tokio::time::sleep(Duration::from_secs(600)).await;
        }
    });
    let routes = Router::new()
        .route("/api/schema", get(schema))
        .route("/api/fields", get(fields))
        .route("/api/wells", get(list_wells).post(create_well))
        .route("/api/wells/{id}", get(get_well).put(update_well).delete(delete_well))
        .route("/api/wells/{id}/quality", get(well_quality))
        .route("/api/wells/{id}/export", get(export_well))
        .route("/api/wells/{id}/data/{dataset}", get(page_rows).post(add_rows).delete(delete_rows))
        .route("/api/wells/{id}/data/{dataset}/{row_id}", put(update_row))
        .route("/api/wells/{id}/data/{dataset}/upload", post(upload_rows).layer(DefaultBodyLimit::max(MAX_UPLOAD)))
        .route("/api/wells/{id}/data/{dataset}/export", get(export_rows))
        .route("/api/datasets", get(datasets))
        .route("/api/templates/{dataset}", get(template))
        .route("/api/imports/{kind}/preview", post(import_preview).layer(DefaultBodyLimit::max(MAX_UPLOAD)))
        .route("/api/imports/{token}/commit", post(import_commit))
        .route("/api/scenarios", get(list_scenarios).post(create_scenario))
        .route("/api/scenarios/{id}", put(update_scenario).delete(delete_scenario))
        .route("/api/runs", get(list_runs).post(create_run))
        .route("/api/runs/{id}", get(get_run).put(rename_run).delete(delete_run))
        .route("/api/runs/{id}/export", get(export_run))
        .route("/api/crm", get(crm_get))
        .route("/api/crm/upload/{kind}", post(crm_upload).layer(DefaultBodyLimit::max(MAX_UPLOAD)))
        .route("/api/crm/staged", axum::routing::delete(crm_unstage))
        .route("/api/crm/staged/{kind}", get(crm_staged_data))
        .route("/api/crm/column", post(crm_column))
        .route("/api/crm/commit", post(crm_commit))
        .route("/api/crm/mpi", post(crm_mpi))
        .route("/api/crm/bl", post(crm_bl))
        .route("/api/crm/aquifer", post(crm_aquifer))
        .route("/api/field/export", get(field_export))
        .route("/api/demo", post(demo_create).delete(demo_remove))
        .route("/api/overview", get(overview))
        .with_state(app);
    let address = env::var("FIELD_API_ADDR").unwrap_or("127.0.0.1:8787".into());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    println!("Field API on http://{address}");
    axum::serve(listener, routes).await?;
    Ok(())
}
