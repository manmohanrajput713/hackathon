# Field Console

A local Next.js dashboard with a Rust/Axum API and Python model workers. Open **http://127.0.0.1:3210** on the computer running the app. From this directory on the prepared Windows machine, run `./Start.ps1`. The launcher prints both process IDs and writes logs under `runtime/`.

## What the replay shows

- **Well performance** runs the saved ESP checkpoint on historical input windows. The reference line is the corresponding simulation output. Deep-CRM fits the first 80% of the separate reservoir history offline and forecasts the final 20% with its closed-form predictor. The gas-lift pressure plot reads the saved MPC simulation and its target.
- **Equipment health** reconstructs the ESP classifier on the supplied training split using the saved hyperparameters and scores the holdout installations. The reported holdout precision for the seven-day warning class is about 5.2%. Valve scores come from the supplied synthetic replay. Neither dataset shares well identifiers with the simulated production wells.
- **Lift allocation** builds response curves for two gas-lift wells using the numerical plant. The mixed-integer solver selects one tested valve setting per well under the total gas budget. Oil, energy and emissions are daily-equivalent rates at the ten-minute simulation endpoint. The UI states the water cut, power and emissions assumptions. The supplied example wells can both favor their lowest tested gas setting. An infeasible budget returns an explanation. Recommendations are advisory.

The API refreshes the replay every ten minutes. Offline fitting uses separate commands. A successful refresh atomically replaces the previous snapshot. API and web listeners bind to loopback.

## Set up a fresh workstation

Use Python 3.12, Node.js 22.12+ and a working 64-bit Rust/Cargo toolchain with its native linker. Keep the five source pipeline folders alongside this project. In PowerShell, from the field-console directory:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt ../pywaterflood
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
