# Field Console

A local Next.js dashboard with a Rust/Axum API and Python model workers. Open **http://127.0.0.1:3210** on the computer running the app. From this directory on the prepared Windows machine, run `./Start.ps1`. The launcher prints both process IDs and writes logs under `runtime/`.

## Layout

The middle area is the **field canvas**. Click empty space to add a well at those coordinates (they also become its position for the CRM geometry seed and MPI). Right-click a producer to add, switch or remove **ESP** (purple square, "E") or **gas lift** (dashed ring, "G"); injectors are blue triangles. Right-click any well to fill its column of CRM production/injection rates, and ESP/gas-lift wells for **VFM**, **Well Models** and **Failure**. The **Deep-CRM**, **Facility**, **Optimizer** and **Results** items on the canvas open their options on right-click; each choice opens in a panel over the canvas. Wells without a position wait in the "Not on the map" tray.

Every CSV input also has an editable **table** in exactly the CSV's layout (Fill table): type the data, or import a CSV into the table, check it, then save. Top bar: Field, Scenario, Active well, Mode (Edit/Review). Left: project tree. Right: inspector.

| Module | What it does |
| --- | --- |
| Deep‑CRM | Follows the pywaterflood user guide: Data → History Match → Connectivity → Forecast → Companion models. Input is the pywaterflood CSV layout: headerless `production.csv` (steps × producers), `injection.csv` (steps × injectors), `time.csv`, optional `pressure.csv` (producer BHP, same shape as production) and `locations.csv` (`name,easting,northing,kind`). Fit CRM or CrmCompensated with every guide option (primary, per-pair/per-producer τ, constraints, PINN training knobs, uniform/random/geometry seed, held-out steps); read gains, τ, injector sums and primary terms; forecast by holding the last injection or uploading a future injection CSV; MPI, Buckley-Leverett and aquifer pseudo-injector. `test-data/` holds a complete sample (pywaterflood 4-producer/5-injector set, guide BHP and well map). |
| VFM | Virtual liquid rate vs well tests for ESP and gas-lift producers. ESP: the pinn checkpoint on high-frequency pump data, with sensor coverage and intervals with insufficient sample history. Gas lift: the gas_pinc model on 60 s valve and pressure telemetry. Casing and tubing-head pressure (plus bottom-hole pressure when a gauge exists) fix the plant state, the gas_pinc algebraic network gives liquid rate, lift gas and BHP, and samples without pressures are carried forward one control step by the gas_pinc PINC (or the gas_pinc plant while the PINC checkpoint fails its accuracy check). |
| Well Models | Nodal analysis at the bottom hole: the inflow curve (IPR) against outflow curves (VLP), one per pump frequency (ESP, pinn BCS equations) or lift-gas rate (gas lift, gas_pinc plant at steady state), from per-well reservoir, PVT and equipment inputs. Each operating point is the stable IPR/VLP crossing; gas-lift points whose plant eigenvalues are unstable are flagged as casing heading. The optimizer uses these operating points. |
| Failure | ESP and Gas Lift sub-tabs: risk timeline with 7-day horizon, alert evidence, maintenance events, casing heading, and the operating envelope the scenario's policy applies. |
| Facility | Per-scenario lift-gas, compressor, separator and power limits, emissions factor and failure policy; capacity used, gas disposition, energy and emissions from the latest optimizer run. |
| Optimizer | Current vs recommended setpoints per well within the facility limits and policy envelopes, binding constraints, and the reason when no recommendation is possible (optional shut-in and unstable points). |
| Results | Every run is saved with its pinned dataset versions, model versions, scenario snapshot and upstream runs. Reopen, rename, compare runs on common axes, replay time series, export Excel/CSV and charts (PNG/SVG). Runs whose inputs changed are flagged for review; their stored inputs are kept. |

Deep-CRM input is CSV only (pywaterflood layout). Per-well data uploads accept CSV (Excel also works). The five-well demo (tree → Demo) is opt-in, lives in its own field and can be removed. Data lives in `runtime/field.db` (SQLite).

Build with `./Build.ps1` (stop the running console first), then start with `./Start.ps1`.

## Deploy on Vercel

The root `vercel.json` deploys one Vercel project with two services on one domain:

| Service | Source | Runtime | Public path |
| --- | --- | --- | --- |
| `api` | repo root, `Dockerfile` | Container (Rust API + Python 3.12 model workers) | `/api/*` |
| `web` | `web/` | Next.js | everything else |

The browser calls `/api/...` directly, so the services need no bindings. The model code reads the pipeline sources from `pipelines/` (`pinn`, `gas_pinc`, `failure_1`, `failure_2`, `pywaterflood`). The image builds pywaterflood from that folder and precomputes `crm.json`, `pump.json` and `curves.json`.

To deploy, import the repository in Vercel and leave **Root Directory** at the repository root. No environment variables are required.

Limits to know about:

- **Data resets.** Containers can only write to `/tmp`, which is cleared when an instance stops (after about 5 idle minutes in production). Wells, runs and uploads are lost then, and the demo can be rebuilt from the tree.
- **Uploads are capped at 4.5 MB** per request, the Vercel limit, instead of the local 64 MB.
- **Requests stop after 300 s** on Hobby (up to 800 s on Pro). Long CRM fits can hit this.
- The Container runtime is in Beta.

To test both services locally with Docker running, use `vercel dev`. The API image alone runs with `docker build -t field-api . && docker run -p 8787:80 field-api`.

## What the replay shows

- **Well performance** runs the saved ESP checkpoint on historical input windows. The reference line is the corresponding simulation output. Deep-CRM fits the first 80% of the separate reservoir history offline and forecasts the final 20% with its closed-form predictor. The gas-lift pressure plot reads the saved MPC simulation and its target.
- **Equipment health** reconstructs the ESP classifier on the supplied training split using the saved hyperparameters and scores the holdout installations. The reported holdout precision for the seven-day warning class is about 5.2%. Valve scores come from the supplied synthetic replay. Neither dataset shares well identifiers with the simulated production wells.
- **Lift allocation** builds response curves for two gas-lift wells using the numerical plant. The mixed-integer solver selects one tested valve setting per well under the total gas budget. Oil, energy and emissions are daily-equivalent rates at the ten-minute simulation endpoint. The UI states the water cut, power and emissions assumptions. The supplied example wells can both favor their lowest tested gas setting. An infeasible budget returns an explanation. Recommendations are advisory.

The API refreshes the replay every ten minutes. Offline fitting uses separate commands. A successful refresh atomically replaces the previous snapshot. API and web listeners bind to loopback.

## Set up a fresh workstation

Use Python 3.12, Node.js 22.12+ and a working 64-bit Rust/Cargo toolchain with its native linker. The five source pipeline folders are in `pipelines/`; folders placed alongside this project are used when `pipelines/` is absent, and `FIELD_PIPELINES` overrides both. In PowerShell, from the field-console directory:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt ./pipelines/pywaterflood
./.venv/Scripts/python.exe models/prepare.py
./.venv/Scripts/python.exe models/curves.py
cargo build --locked
npm.cmd --prefix web ci
npm.cmd --prefix web run build
./Start.ps1
```

On other systems use the environment's Python path, start the API from this directory with `FIELD_PYTHON` set to that interpreter, and run `npm start -- -H 127.0.0.1 -p 3210` from `web/`. To regenerate valve replay scores use the scoring entry point in the sibling failure_2 project before refreshing.

## Matched client-well replay

The optional importer runs both models on a common client export and joins results by producer ID and time. Provide three files in one directory:

1. `field.json` describes producer/injector column names, training cutoff, maximum forecast age, checkpoint path and the ESP checkpoint's input sampling interval. Use trusted checkpoints compatible with the existing ESP network and normalization.
2. `reservoir.csv` has `timestamp` followed by the producer and injector columns named in the manifest. Supply aligned rates on a documented, common volume basis in m³/h, using the appropriate fluid-property conversion upstream. The displayed model comparison assumes the ESP flow and well test use that same basis.
3. `telemetry.csv` has `well_id,timestamp,frequency_hz,choke_percent,manifold_pa,reservoir_pa,bottomhole_pa,wellhead_pa,test_m3h`. Supply absolute pressures and UTC or offset timestamps. Leave missing tests blank. Reservoir pressure comes from the supplied field input.

```json
{
  "producers": ["P1"],
  "injectors": ["I1"],
  "training_end": "2025-06-30T23:59:59Z",
  "crm_max_age_seconds": 86400,
  "esp": {"P1": {"checkpoint": "models/P1.pt", "sample_seconds": 1}}
}
```

Run `./.venv/Scripts/python.exe models/import_field.py PATH_TO_EXPORT`, then `./.venv/Scripts/python.exe models/replay.py`. Open **Client wells**. Each row shows the ESP estimate, a timed well test when available, and the most recent eligible CRM forecast with its own timestamp. Gaps break the ESP window. The import is historical and uses the recorded injection schedule. Field-specific equipment calibration, operating envelopes, energy meters and facility constraints are needed before adapting the demonstration allocator for client recommendations.
