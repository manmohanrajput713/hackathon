use axum::{extract::State, http::StatusCode, routing::{get, post}, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{env, path::PathBuf, sync::Arc, time::Duration};
use tokio::{fs, process::Command, sync::Semaphore};

struct App { root: PathBuf, python: String, jobs: Semaphore }
type Shared = Arc<App>;
type Error = (StatusCode, Json<Value>);

fn error(code: StatusCode, message: &str) -> Error {
    (code, Json(json!({"error":message})))
}
async fn overview(State(app): State<Shared>) -> Result<Json<Value>, Error> {
    let bytes = fs::read(app.root.join("runtime/snapshot.json")).await
        .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE, "Replay is loading. Refresh in a moment."))?;
    serde_json::from_slice(&bytes).map(Json).map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR, "Invalid replay snapshot"))
}
#[derive(Deserialize)]
struct Scenario { gas: f64, energy_weight: f64, emissions_weight: f64 }
async fn allocate(State(app): State<Shared>, Json(s): Json<Scenario>) -> Result<Json<Value>, Error> {
    if [s.gas,s.energy_weight,s.emissions_weight].iter().any(|n| !n.is_finite() || *n < 0.0) {
        return Err(error(StatusCode::BAD_REQUEST, "Enter finite, nonnegative limits and weights"));
    }
    let _slot = app.jobs.try_acquire().map_err(|_| error(StatusCode::TOO_MANY_REQUESTS,"Allocator is busy"))?;
    let mut cmd = Command::new(&app.python);
    cmd.current_dir(&app.root).kill_on_drop(true).args(["models/allocate.py",
        &s.gas.to_string(), &s.energy_weight.to_string(), &s.emissions_weight.to_string()]);
    let output = tokio::time::timeout(Duration::from_secs(60), cmd.output()).await
        .map_err(|_| error(StatusCode::GATEWAY_TIMEOUT,"Allocation timed out"))?
        .map_err(|_| error(StatusCode::SERVICE_UNAVAILABLE,"Python worker unavailable"))?;
    if !output.status.success() {
        eprintln!("{}",String::from_utf8_lossy(&output.stderr));
        return Err(error(StatusCode::INTERNAL_SERVER_ERROR,"Allocation failed. Check the operating curves."));
    }
    let result: Value = serde_json::from_slice(&output.stdout)
        .map_err(|_| error(StatusCode::INTERNAL_SERVER_ERROR,"Invalid allocation result"))?;
    Ok(Json(result))
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = Arc::new(App { root: env::current_dir()?, python: env::var("FIELD_PYTHON").unwrap_or("python".into()), jobs: Semaphore::new(2) });
    let worker = app.clone();
    tokio::spawn(async move {
        loop {
            let run = Command::new(&worker.python).current_dir(&worker.root)
                .arg("models/replay.py").kill_on_drop(true).status().await;
            if !matches!(run, Ok(status) if status.success()) { eprintln!("Replay refresh failed; retaining last completed snapshot"); }
            tokio::time::sleep(Duration::from_secs(600)).await;
        }
    });
    let routes = Router::new().route("/api/overview",get(overview)).route("/api/allocate",post(allocate)).with_state(app);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8787").await?;
    println!("Field API on http://127.0.0.1:8787");
    axum::serve(listener,routes).await?;
    Ok(())
}

