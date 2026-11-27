# Continuous gas-lift valve failure forecasting

`failure_2` predicts confirmed valve leaks or malfunctions within a configurable horizon, seven days by default. It follows the local `failure_1` workflow of schema checks, grouped feature engineering, Optuna tuning, frozen threshold selection, and numbered result tables. Training uses local CSVs and scoring loads a frozen bundle.

Mustaqim and Yasutra (2024), IPA24-E-314, *Unlocking Efficiency in Gas Lift System Operation: A Data Driven Approach for Early Problem Identification Using Machine Learning and Digitalized Two-Pen Recorder Chart*, motivates the Random Forest baseline. Its five inputs are casing pressure, tubing-head pressure, casing minus tubing-head pressure, and one-observation backward changes in the two pressures. This implementation uses those inputs for a future, valve-specific outcome. The IPA publisher page provides the abstract and lists the full text behind member login. [Indonesian Petroleum Association](https://www.ipa.or.id/id/publications/unlocking-efficiency-in-gas-lift-system-operation-a-data-driven-approach-for-early-problem-identification-using-machine-learning-and-digitalized-two-pen-recorder-chart)

## Run

The saved bundle was built with Python 3.12 and the library versions in `requirements.txt`. Training and scoring run offline once those libraries are installed. Load joblib bundles from trusted sources.

```bash
cd /Users/rohit/failure_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Generate and evaluate the included synthetic case.

```bash
python main.py make-fixture
python main.py train \
  --telemetry dataset/synthetic/telemetry.csv \
  --maintenance dataset/synthetic/maintenance.csv \
  --data-kind synthetic --n-trials 4 --n-estimators 60
python main.py score \
  --telemetry dataset/synthetic/telemetry.csv \
  --maintenance dataset/synthetic/maintenance.csv \
  --bundle results/synthetic/model_bundle.joblib \
  --output results/synthetic/frozen_scores.csv
```

Supply real input paths and a site-specific mapping when you have field data.

```bash
python main.py train \
  --telemetry /path/to/telemetry.csv --maintenance /path/to/maintenance.csv \
  --config /path/to/field_config.json --data-kind field
python main.py score \
  --telemetry /path/to/recent_and_historical_telemetry.csv \
  --maintenance /path/to/known_maintenance.csv \
  --bundle results/field/model_bundle.joblib \
  --from-timestamp 2026-11-22T00:00:00Z \
  --output results/field/scores.csv --heading-output results/field/live_casing_heading.csv
```

Run `python main.py train --help` for the forecast horizon, maximum telemetry gap, rolling window, holdout fraction, split dates, and tuning budget. `--feature-set baseline` limits the comparison to the five paper inputs. The optional scoring cut `--from-timestamp` filters the output after feature computation. For recurring scoring, include completed prior observations for the lookback window and the latest known valve service history. Use the same input column mapping and units recorded in the bundle.

## Input schema

Both inputs are CSV. Timestamps are parsed as UTC and an explicit offset is converted to UTC. Naive timestamps are treated as UTC. Use well IDs consistently in both files. Each telemetry `(well_id, timestamp)` pair must be unique.

| Canonical telemetry field | Requirement and interpretation |
| --- | --- |
| `well_id`, `timestamp` | Required text ID and observation time. |
| `casing_pressure`, `tubing_head_pressure` | Required for training. Both must be available at the current and previous sample to score. Default is gauge kPa. Units can be `pa`, `kpa`, `mpa`, `bar`, or `psi`. Set `reference` independently to `absolute` or `gauge`. Supply site ambient pressure in `ambient_kpa` for gauge measurements. The default offset is 101.325 kPa. |
| `lift_gas_rate` | Optional measured injection at standard conditions. Units are `sm3/d`, `scf/d`, `mscf/d`, or `mmscf/d`. Use a consistent standard temperature and pressure across feeds. |
| `choke_position` | Optional measured position of a consistent choke tag. Set units to `percent` or `fraction`. |
| `operating_state` | Optional as-of state such as `continuous_gas_lift`, `transition`, or `shut_in`. Use observations or states known at the sample time. |
| `nodal_residual` | Optional signed pressure residual. Join measured minus predicted pressure at the same well, location, reference and timestamp in the telemetry source. Use a prediction from the existing frozen model built from as-of inputs. Set `nodal_model_id` to the checkpoint version or hash and `nodal_residual_unit` to a pressure unit. |

| Canonical maintenance field | Requirement and interpretation |
| --- | --- |
| `well_id`, `event_type` | Required. Qualifying types are `valve_leak` and `valve_malfunction` with explicit confirmation. `installation`, `replacement`, and `planned_replacement` are effective service resets. `tubing_leak`, `casing_heading`, `other`, and `coverage_end` are recognized. Enter planned replacements only once performed. |
| `confirmed` | `true`/`false`, `yes`/`no`, or `1`/`0`. A blank is unconfirmed. |
| `detected_at`, `failed_at`, `changeout_at` | Supply the earliest defensible times for confirmed faults. An actual valve `changeout_at` on a valve-fault record also opens a new service interval. |
| `event_id` | Stable episode identifier within a well. Use the same ID across detection, failure, confirmation, and changeout reports for one episode. |
| `timestamp` | Effective time for service resets and the verified-through timestamp of a `coverage_end` record. For a valve fault with a reliable generic event time, map the source column to `detected_at` or `failed_at`. |
| `recorded_at` | Time the service change or confirmed fault became available to the forecasting feed. If blank, effective times are used as the availability timestamps. Provide this field to reproduce reporting delays during replay. |

Use the JSON keys `telemetry_columns` and `maintenance_columns` to map canonical fields to source headers. Use `event_types` to map source event codes to the canonical values. See [the field config example](docs/field_config.example.json). By default, provide one `coverage_end` record per well attesting that the maintenance log is complete from the first telemetry observation through that timestamp. Where this is guaranteed by the upstream process, the config flag `assume_log_complete_through_telemetry` allows implicit coverage for wells missing the record.

The inspected local `gas_pinc` checkpoint produces simulation pressure outputs and has no matching field measurements for an as-of residual. The synthetic run uses the pressure and measured-channel schema without a nodal residual. A field export can supply this optional feature with the checkpoint ID and the measurement used to construct it.

## Labels and features

- A sample at time `t` is positive for a qualifying episode in `(t, t + horizon]` in its valve service interval. The target uses the earliest reliable recorded detection or failure time, consolidated by episode ID. Later timestamps after a known changeout are treated as unreliable. A confirmed changeout-only episode targets time to changeout and is identified separately in the event inventory, metrics counts, and bundle metadata.
- Each well is partitioned at actual valve installations and changeouts. A well without records uses a single unknown-start interval. A fault concurrent with changeout belongs to the ending interval. Forecast samples at or after the first event are excluded until a verified reset. Repeated reports within one unresolved service interval count as one episode. The audit counts subsequent events whose new service interval could not be established.
- Negative labels require verified log coverage and telemetry follow-up through the full horizon with no gap longer than `--max-gap-hours` and no service reset through the horizon end. An unresolved, undated confirmed fault makes negative windows unverifiable for that well. A confirmed event can establish a positive horizon through a telemetry gap.
- Pressures are converted to absolute kPa before differencing. The five baseline features require two consecutive complete observations in the same well and interval and at most 36 hours apart by default. Extended candidates add trailing pressure means and standard deviations within a gap-free run, plus optional channels present in their training rows. The training pipeline fits numeric medians, available missingness indicators, and categories on training rows. Optional channels can be absent at score time.
- Casing heading is screened separately in `casing_heading.csv`. The causal screen looks for two completed, similarly spaced, amplitude-qualified casing-pressure cycles within the last 24 hours. By default it checks periods of 2–8 hours and an amplitude of at least 25 kPa. It needs at least four observations per shortest configured cycle and returns `unavailable` when the cadence or history is insufficient.

Scoring returns exactly `well_id,timestamp,risk_score,alert_status`. Status is `alert` at or above the frozen threshold, `no_alert` below it, and `unavailable` with a blank score when baseline inputs are insufficient or an event is already known for the current interval. The scoring replay uses maintenance information as of each sample. A fault that was confirmed later can therefore have scores before that confirmation in the scoring replay. Historical evaluation uses the reconciled event time to exclude event-time and post-event rows.

## Validation and artifacts

The simulator smoke test and field commands use the same pipeline. The CLI reserves whole wells if enough remain to fit both classes in training and validation. For both test cohorts it uses the same future period. Development wells provide the chronological backtest, and every well in the held-out cohort is excluded from all fitting and model selection. Default validation and test calendar cutoffs are 55% and 78% of the way through the observed time range. Supply prespecified dates with `--validation-start` and `--test-start`. A full horizon is embargoed before each cut, including samples whose horizon ends exactly on the cut. Training and validation also omit confirmed labels first recorded at or after their respective cutoff.

Random Forest and XGBoost are tuned on the same training and validation splits, first on the identical five inputs and then on the same extended inputs when available. Optuna ranks trials by validation average precision. The selected candidate has the highest validation average precision, with validation F1 as a tie-breaker. Each candidate's alert threshold maximizes validation sample F1. The fitted pipeline and threshold are frozen before either test cohort is scored.

| File under `results/synthetic/` | Contents |
| --- | --- |
| `processing_audit.json`, `table2_data_exploration.csv` | Input and missingness counts, feature summaries, label exclusions, target basis, and split policy. |
| `table3_class_distribution.csv`, `event_inventory.csv` | Well, row, and episode counts by split, plus reconciled confirmed events. |
| `table5_best_hyperparameters.json`, `table6_classification_report.csv` | Validation search and threshold selection, plus metrics for each evaluated candidate and partition. |
| `fig9_precision_recall_curve.png`, `sample_predictions.csv`, `event_evaluation.csv` | Precision–recall curves, sample forecasts and exclusions, and episode detections with first-alert lead times. |
| `model_bundle.joblib`, `model_bundle.json` | Frozen fitted pipeline and readable metadata with feature schema and units, horizon, threshold, column mapping, and library versions. |
| `casing_heading.csv`, `frozen_scores.csv`, `latest_scores.csv` | Separate pressure-cycle screening, full scoring replay, and a replay filtered to the last two days. |

PR-AUC is average precision. Sample precision and recall use scored rows with known labels. Episode recall counts each confirmed episode with at least one labeled opportunity in the split once, even when all of its opportunities were unscorable. Lead time is from the first in-window alert to the recorded target time. The summary reports the median among detected events. Consecutive false-alert samples within the same service interval count as one false-alert episode, with gaps, unavailable rows and intervening non-alerts breaking the episode. Well-week exposure sums the elapsed time between adjacent scored, labeled observations within that same gap policy. Undefined metrics are blank.

## Field validation inputs

The checked-in results come from the deterministic synthetic fixture in `dataset/synthetic/`. No paired field telemetry and confirmed valve-event labels were available for this run. Field validation needs synchronized casing and tubing-head pressure with units and pressure references, actual valve installation and replacement times, confirmed and classified fault reports with the earliest reliable detection or failure time, verified maintenance coverage for event-free windows, and enough event and event-free well history to populate both fitting periods and independent tests. Lift-gas rate, choke position, operating state, and a matched frozen-model residual can extend the comparison where available. A field run without both classes in training and validation writes an audit and exits before fitting.
