"""Regression tests for forecast windows, well isolation, and frozen scoring."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from main import feature_frame
from src.evaluate import metrics_for_split
from src.features import BASELINE, casing_heading, model_inputs
from src.model import assign_splits, bundle_metadata, make_pipeline, score_rows
from src.preprocessing import attach_failure_labels, load_config, maintenance_timelines, read_maintenance, read_telemetry


ROOT = Path(__file__).resolve().parents[1]
ORIGIN = pd.Timestamp("2024-01-01T00:00:00Z")
SETTINGS = {
    "horizon_days": 7,
    "max_gap_hours": 36,
    "rolling_hours": 72,
    "heading": {"min_period_hours": 2, "max_period_hours": 8, "window_hours": 24,
                "max_step_minutes": 60, "min_amplitude_kpa": 25},
}


def at(day):
    return ORIGIN + pd.Timedelta(days=day)


def telemetry(well, days, base=400):
    return pd.DataFrame({
        "well_id": well,
        "timestamp": [at(day) for day in days],
        "casing_kpa": base + np.asarray(days, dtype=float),
        "tubing_kpa": base / 2 + np.asarray(days, dtype=float) * 0.25,
    })


def timelines(tmp_path, entries):
    path = tmp_path / "maintenance.csv"
    pd.DataFrame(entries).to_csv(path, index=False)
    return maintenance_timelines(read_maintenance(path, load_config(None)))


def label(tmp_path, measurements, entries, *, assume_complete=False):
    resets, events, coverage, audit = timelines(tmp_path, entries)
    features = feature_frame(measurements, resets, events, SETTINGS)
    return attach_failure_labels(
        features, resets, events, coverage, pd.Timedelta(days=7), pd.Timedelta(hours=36),
        assume_log_complete=assume_complete,
        undated_wells=audit["wells_with_undated_confirmed_events"],
    )


def record(well, kind, **fields):
    return {"well_id": well, "event_type": kind, **fields}


def test_forecast_boundary_and_earliest_confirmed_time(tmp_path):
    measurements = telemetry("A", range(18))
    entries = [
        record("A", "installation", timestamp=at(0)),
        record("A", "valve_malfunction", event_id="1", confirmed=True, changeout_at=at(12)),
        record("A", "valve_malfunction", event_id="1", confirmed=True, detected_at=at(10), failed_at=at(9)),
        record("A", "valve_leak", confirmed=False, detected_at=at(4)),
        record("A", "tubing_leak", confirmed=True, detected_at=at(3)),
        record("A", "casing_heading", confirmed=True, detected_at=at(5)),
        record("A", "coverage_end", timestamp=at(17)),
    ]
    labeled, events, audit = label(tmp_path, measurements, entries)
    labels = labeled.set_index("timestamp")["label"]
    assert labels.loc[at(1)] == 0
    assert labels.loc[at(2)] == 1  # Exactly seven days out.
    assert labels.loc[at(8)] == 1
    assert labels.loc[at(9):at(11)].isna().all()
    assert events[["event_at", "target_basis"]].values.tolist() == [[at(9), "detection_or_failure"]]
    assert len(events) == audit["timed_event_reports_merged_by_event_id"] == 1


def test_changeout_target_and_replacement_boundary(tmp_path):
    measurements = telemetry("A", range(24))
    labeled, events, _ = label(tmp_path, measurements, [
        record("A", "installation", timestamp=at(0)),
        record("A", "valve_leak", confirmed=True, changeout_at=at(8)),
        record("A", "planned_replacement", confirmed=True, timestamp=at(18)),
        record("A", "coverage_end", timestamp=at(23)),
    ])
    by_time = labeled.set_index("timestamp")
    assert by_time.loc[at(1), "label"] == 1
    assert by_time.loc[at(8), "label_status"] == "event_or_post_event"
    assert by_time.loc[at(9), "label"] == 0
    assert by_time.loc[at(11), "label_status"] == "incomplete_followup"
    assert events.iloc[0]["target_basis"] == "changeout"


def test_incomplete_followup_and_telemetry_gaps(tmp_path):
    measurements = pd.concat([telemetry(well, [*range(10), *range(13, 28)]) for well in ("A", "B", "C")])
    labeled, _, _ = label(tmp_path, measurements, [
        record("A", "coverage_end", timestamp=at(24)),
        record("B", "coverage_end", timestamp=at(27)),
        record("B", "valve_leak", confirmed=True),
    ])
    labels = labeled.set_index(["well_id", "timestamp"])["label"]
    assert labels["A", at(2)] == 0
    assert pd.isna(labels["A", at(3)])  # Gap before completion.
    assert labels["A", at(17)] == 0
    assert pd.isna(labels["A", at(18)])  # Telemetry extends past verified log coverage.
    assert labels.loc["B"].isna().all()  # A confirmed, undated fault precludes verified negatives.
    assert labels.loc["C"].isna().all()  # Coverage must be declared.
    assumed, _, _ = label(tmp_path, measurements, [
        record("A", "coverage_end", timestamp=at(24)),
    ], assume_complete=True)
    labels = assumed.set_index(["well_id", "timestamp"])["label"]
    assert labels["C", at(20)] == 0
    assert pd.isna(labels["C", at(21)])  # Dataset tail lacks seven days of telemetry.


def test_failure_on_initial_installation_closes_forecasting_until_changeout(tmp_path):
    measurements = telemetry("A", range(20))
    labeled, _, _ = label(tmp_path, measurements, [
        record("A", "installation", timestamp=at(0)),
        record("A", "valve_leak", confirmed=True, detected_at=at(0), changeout_at=at(7)),
        record("A", "coverage_end", timestamp=at(19)),
    ])
    assert labeled.loc[:6, "label_status"].eq("event_or_post_event").all()
    assert labeled.loc[:6, "feature_status"].eq("known_event_or_post_event").all()
    assert labeled.loc[7, "feature_status"] == "first_observation"
    assert labeled.loc[8, "label"] == 0


def test_features_stay_within_well_and_as_of_service(tmp_path):
    measurements = pd.concat([telemetry("A", [0, 1, 2, 3, 4, 5, 8, 9]),
                              telemetry("B", range(10), base=2_000)])
    resets, events, _, _ = timelines(tmp_path, [
        record("A", "replacement", timestamp=at(2), recorded_at=at(3)),
        record("A", "valve_leak", confirmed=True, detected_at=at(4), recorded_at=at(5)),
    ])
    features = feature_frame(measurements, resets, events, SETTINGS)
    by_time = features.set_index(["well_id", "timestamp"])
    assert by_time.loc[("B", at(0)), "feature_status"] == "first_observation"
    assert by_time.loc[("B", at(1)), "casing_change_kpa"] == 1
    assert by_time.loc[("A", at(2)), "feature_status"] == "ready"
    assert by_time.loc[("A", at(3)), "feature_status"] == "first_observation"
    assert pd.isna(by_time.loc[("A", at(4)), "casing_mean_kpa"])
    assert by_time.loc[("A", at(4)), "feature_status"] == "ready"
    assert by_time.loc[("A", at(5)), "feature_status"] == "known_event_or_post_event"
    prefix = feature_frame(measurements.loc[measurements["timestamp"].le(at(4))], resets, events, SETTINGS)
    pd.testing.assert_frame_equal(
        prefix, features.loc[features["timestamp"].le(at(4))].reset_index(drop=True),
    )


def test_well_holdout_and_two_purged_time_cuts():
    data = pd.concat([telemetry(str(i), range(48)) for i in range(6)], ignore_index=True)
    data["label"] = data["timestamp"].dt.day % 2
    data["feature_status"] = "ready"
    horizon = pd.Timedelta(days=7)
    validation_start, test_start = at(20), at(35)
    split, policy, usable = assign_splits(data, horizon, 0.3, validation_start, test_start, seed=42)
    train, validation = data.loc[split.eq("train")], data.loc[split.eq("validation")]
    held = data.loc[split.eq("held_well_test")]
    assert usable and not held.empty
    assert set(held["well_id"]) == set(policy["holdout_wells"])
    assert set(held["well_id"]).isdisjoint(set(train["well_id"]) | set(validation["well_id"]))
    assert train["timestamp"].max() + horizon < validation_start
    assert validation["timestamp"].max() + horizon < test_start
    changed = data.copy()
    changed.loc[changed["timestamp"].ge(test_start), "label"] = np.nan
    pd.testing.assert_series_equal(
        split, assign_splits(changed, horizon, 0.3, validation_start, test_start, seed=42)[0],
    )
    changed["label_available_at"] = changed["timestamp"].where(
        ~changed["label"].eq(1), test_start,
    )
    assert not assign_splits(changed, horizon, 0.3, validation_start, test_start, seed=42)[2]


def test_distinct_event_and_false_alert_counts():
    days = [1, 2, 3, 4, 5, 6, 7]
    rows = telemetry("A", days).assign(
        service_start=0,
        label=[0, 0, 0, 1, 1, 0, 0],
        event_at=at(11),
        target_basis="detection_or_failure",
    )
    scores = pd.DataFrame({"risk_score": [0.9, 0.8, 0.1, 0.9, 0.8, 0.9, 0.7],
                           "alert_status": ["alert", "alert", "no_alert", "alert", "alert", "alert", "alert"]})
    metrics, events = metrics_for_split(rows, scores, pd.Timedelta(hours=36))
    assert metrics["events_detected"] == metrics["events"] == len(events) == 1
    assert metrics["false_alert_episodes"] == 2
    assert metrics["false_alerts_per_well_week"] == pytest.approx(2 / (6 / 7))
    assert metrics["median_lead_time_hours"] == 7 * 24


@pytest.mark.parametrize("use_optional", [False, True])
def test_saved_model_scoring(tmp_path, use_optional):
    mapping = tmp_path / "config.json"
    mapping.write_text(json.dumps({
        "telemetry_columns": {"casing_pressure": "case", "tubing_head_pressure": "tube"},
        "casing_pressure": {"unit": "bar", "reference": "gauge", "ambient_kpa": 100},
        "tubing_head_pressure": {"unit": "kpa", "reference": "absolute"},
    }))
    config = load_config(mapping)
    telemetry_path = tmp_path / "telemetry.csv"
    pd.DataFrame({
        "well_id": "001", "timestamp": [at(day) for day in (0, 1, 2, 3, 4, 7, 8)],
        "case": [4, 4.1, 4.2, 4.5, 4.9, 4.8, 4.7], "tube": [200, 200, 201, 205, 212, 220, 223],
    }).to_csv(telemetry_path, index=False)
    canonical, _ = read_telemetry(telemetry_path, config)
    resets, events, _, _ = maintenance_timelines(read_maintenance(None, config))
    features = feature_frame(canonical, resets, events, SETTINGS)
    assert features.loc[1, "casing_minus_tubing_kpa"] == pytest.approx(310)
    assert features.loc[1, "casing_change_kpa"] == pytest.approx(10)
    assert features.loc[5, "feature_status"] == "gap"
    fitting = features.loc[features["feature_status"].eq("ready")].assign(
        operating_state=["flowing", "flowing", "startup", "flowing", "flowing"],
        lift_gas_sm3d=[100, 99, np.nan, 106, 95],
    )
    schema = [*BASELINE, "operating_state", "lift_gas_sm3d"] if use_optional else BASELINE
    y = pd.Series([0, 0, 1, 1, 0], index=fitting.index)
    args = SimpleNamespace(n_estimators=16, jobs=1, seed=42)
    pipe = make_pipeline("random_forest", schema, y, {"max_depth": 4}, args)
    pipe.fit(model_inputs(fitting, schema), y)
    candidate = {"pipeline": pipe, "schema": schema, "threshold": 0.4}
    bundle = tmp_path / "model.joblib"
    joblib.dump(bundle_metadata("test/random_forest", candidate, SETTINGS, config, "synthetic")
                | {"pipeline": pipe}, bundle)
    original_hash = hashlib.sha256(bundle.read_bytes()).digest()
    score_path = tmp_path / "scores.csv"
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "main.py"), "score", "--telemetry", str(telemetry_path),
         "--bundle", str(bundle), "--output", str(score_path)],
        cwd=ROOT, check=True,
    )
    actual = pd.read_csv(score_path, parse_dates=["timestamp"], dtype={"well_id": str})
    expected = score_rows(features, pipe, schema, 0.4)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.loc[0, "alert_status"] == actual.loc[5, "alert_status"] == "unavailable"
    assert hashlib.sha256(bundle.read_bytes()).digest() == original_hash


def test_heading_is_causal_and_requires_sampling_cadence():
    days = np.arange(49) / 48
    rows = telemetry("A", days).assign(
        service_start=0, casing_kpa=400 + 40 * np.sin(12 * np.pi * days),
        timestamp=pd.date_range(ORIGIN, periods=len(days), freq="30min"),
    )
    full = casing_heading(rows, SETTINGS["heading"])["casing_heading_status"]
    prefix = casing_heading(rows.iloc[:30], SETTINGS["heading"])["casing_heading_status"]
    pd.testing.assert_series_equal(prefix, full.iloc[:30])
    assert full.eq("possible_heading").any()
    assert casing_heading(rows.iloc[::4], SETTINGS["heading"])["casing_heading_status"].eq("unavailable").all()


def test_training_without_timed_events_writes_audit(tmp_path):
    measurements = telemetry("A", range(40)).rename(columns={
        "casing_kpa": "casing_pressure", "tubing_kpa": "tubing_head_pressure",
    })
    telemetry_path = tmp_path / "telemetry.csv"
    maintenance_path = tmp_path / "maintenance.csv"
    results = tmp_path / "results"
    measurements.to_csv(telemetry_path, index=False)
    pd.DataFrame([record("A", "coverage_end", timestamp=at(39))]).to_csv(maintenance_path, index=False)
    run = subprocess.run(
        [sys.executable, "-B", str(ROOT / "main.py"), "train", "--telemetry", str(telemetry_path),
         "--maintenance", str(maintenance_path), "--output-dir", str(results)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert run.returncode == 2
    assert "need both labeled classes" in run.stderr
    audit = json.loads((results / "processing_audit.json").read_text())
    assert audit["maintenance"]["qualifying_confirmed_rows"] == 0
    assert not (results / "model_bundle.joblib").exists()
