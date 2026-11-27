"""Train and score a continuous gas-lift valve failure forecast from local CSVs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd

from src.evaluate import PARTITIONS, metrics_for_split, plot_precision_recall, split_summary
from src.features import FEATURE_UNITS, build_features, casing_heading
from src.fixture import write_fixture
from src.model import (
    RANDOM_STATE, assign_splits, bundle_metadata, eligible, fit_candidates, load_bundle, score_rows,
)
from src.preprocessing import (
    align_service_as_of, attach_failure_labels, load_config, maintenance_timelines, read_maintenance, read_telemetry,
)


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("failure_2")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(required=True, dest="action")
    train = actions.add_parser("train", help="Train and evaluate offline")
    score = actions.add_parser("score", help="Score using a frozen model bundle")
    fixture = actions.add_parser("make-fixture", help="Write labeled synthetic input CSVs")
    for action in (train, score, fixture):
        action.add_argument("--root", type=Path, default=ROOT, help="Default dataset and results root")
    for action in (train, score):
        action.add_argument("--telemetry", type=Path, required=True)
    train.add_argument("--maintenance", type=Path, required=True)
    train.add_argument("--config", type=Path, help="Input column mappings and units JSON")
    train.add_argument("--data-kind", choices=("field", "synthetic"), default="field")
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--skip-download", action="store_true", help="Accepted for failure_1 compatibility. Runs are offline.")
    train.add_argument("--n-trials", type=int, default=20)
    train.add_argument("--n-estimators", type=int, default=200)
    train.add_argument("--jobs", type=int, default=2)
    train.add_argument("--seed", type=int, default=RANDOM_STATE)
    train.add_argument("--feature-set", choices=("baseline", "extended"), default="extended")
    train.add_argument("--horizon-days", type=float, default=7)
    train.add_argument("--max-gap-hours", type=float, default=36)
    train.add_argument("--rolling-hours", type=float, default=72)
    train.add_argument("--holdout-fraction", type=float, default=0.30)
    train.add_argument("--validation-start")
    train.add_argument("--test-start")
    train.add_argument("--heading-min-period-hours", type=float, default=2)
    train.add_argument("--heading-max-period-hours", type=float, default=8)
    train.add_argument("--heading-window-hours", type=float, default=24)
    train.add_argument("--heading-max-step-minutes", type=float, default=60)
    train.add_argument("--heading-min-amplitude-kpa", type=float, default=25)
    score.add_argument("--maintenance", type=Path, help="Service and known event records if available")
    score.add_argument("--bundle", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--heading-output", type=Path)
    score.add_argument("--from-timestamp", help="Return forecasts at or after this UTC time while using prior input history")
    fixture.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def get_settings(args) -> dict:
    values = {
        "horizon_days": args.horizon_days, "max_gap_hours": args.max_gap_hours, "rolling_hours": args.rolling_hours,
        "heading": {
            "min_period_hours": args.heading_min_period_hours, "max_period_hours": args.heading_max_period_hours,
            "window_hours": args.heading_window_hours, "max_step_minutes": args.heading_max_step_minutes,
            "min_amplitude_kpa": args.heading_min_amplitude_kpa,
        },
    }
    for value in (args.horizon_days, args.max_gap_hours, args.rolling_hours, *values["heading"].values()):
        if not 0 < value < float("inf"):
            raise ValueError("Horizons, windows, gaps, and heading settings must be positive and finite")
    if not args.heading_min_period_hours <= args.heading_max_period_hours <= args.heading_window_hours / 2:
        raise ValueError("Heading period bounds must be ordered and the window must cover two maximum-length cycles")
    return values


def feature_frame(telemetry, resets, events, settings):
    return build_features(
        align_service_as_of(telemetry, resets, events),
        pd.Timedelta(hours=settings["max_gap_hours"]),
        pd.Timedelta(hours=settings["rolling_hours"]),
    )


def train(args) -> int:
    if min(args.n_trials, args.n_estimators, args.jobs) < 1:
        raise ValueError("n-trials, n-estimators, and jobs must be positive")
    config = load_config(args.config)
    settings = get_settings(args)
    horizon = pd.Timedelta(days=args.horizon_days)
    gap = pd.Timedelta(hours=args.max_gap_hours)
    directory = args.output_dir or args.root / "results" / args.data_kind
    directory.mkdir(parents=True, exist_ok=True)

    telemetry, telemetry_audit = read_telemetry(args.telemetry, config)
    log = read_maintenance(args.maintenance, config)
    resets, events, coverage, maintenance_audit = maintenance_timelines(log)
    features = feature_frame(telemetry, resets, events, settings)
    labeled, episodes, label_audit = attach_failure_labels(
        features, resets, events, coverage, horizon, gap,
        assume_log_complete=config["assume_log_complete_through_telemetry"],
        undated_wells=maintenance_audit["wells_with_undated_confirmed_events"],
    )
    heading = casing_heading(features, settings["heading"])
    write_csv(heading, directory / "casing_heading.csv")
    write_csv(episodes[["well_id", "event_id", "event_type", "event_at", "target_basis",
                        "service_index", "precedes_telemetry"]], directory / "event_inventory.csv")
    audit = {
        "data_kind": args.data_kind,
        "inputs": {"telemetry": str(args.telemetry), "maintenance": str(args.maintenance)},
        "settings": settings,
        "input_config": config,
        "telemetry": telemetry_audit,
        "maintenance": maintenance_audit,
        "labeling": label_audit,
        "rows_by_feature_status": features["feature_status"].value_counts().to_dict(),
        "casing_heading": heading["casing_heading_status"].value_counts().to_dict(),
    }
    audit_path = directory / "processing_audit.json"
    parsed = [pd.to_datetime(raw, utc=True) if raw else None for raw in (args.validation_start, args.test_start)]
    try:
        labeled["split"], split_policy, supports_fit = assign_splits(labeled, horizon, args.holdout_fraction, *parsed, args.seed)
    except ValueError:
        write_json(audit_path, audit)
        raise
    audit["split_policy"] = split_policy
    late = pd.Series(False, index=labeled.index)
    for split, boundary in (("train", "validation_start"), ("validation", "test_start")):
        late |= labeled["split"].eq(split) & labeled["label_available_at"].ge(pd.Timestamp(split_policy[boundary]))
    labeled.loc[late, "label"] = float("nan")
    labeled.loc[late, "label_status"] = "label_not_yet_available_at_split_cutoff"
    audit["rows_with_label_unavailable_at_split_cutoff"] = int(late.sum())
    audit["training_status"] = "incomplete"
    write_json(audit_path, audit)
    exploration = features.reindex(columns=[name for name in FEATURE_UNITS if name in features and name != "operating_state"])
    write_csv(exploration.describe().T.rename_axis("feature").reset_index(), directory / "table2_data_exploration.csv")
    summary = split_summary(labeled)
    summary.insert(0, "data_kind", args.data_kind)
    write_csv(summary, directory / "table3_class_distribution.csv")
    if not supports_fit:
        raise ValueError(
            "Training and validation each need both labeled classes. "
            f"See {audit_path} and table3_class_distribution.csv for coverage and split counts."
        )
    LOGGER.info("Prepared %d rows from %d wells (%s data).", len(labeled), telemetry["well_id"].nunique(), args.data_kind)
    development = labeled.loc[eligible(labeled)]
    candidates, search = fit_candidates(
        development.loc[development["split"].eq("train")],
        development.loc[development["split"].eq("validation")],
        args,
    )
    rank = sorted(search, key=lambda row: (-row["validation_pr_auc"], -row["validation_f1_at_threshold"], row["candidate"]))
    selected = rank[0]["candidate"]
    best = candidates[selected]
    metadata = bundle_metadata(selected, best, settings, config, args.data_kind)
    metadata["events_by_target_basis"] = label_audit["events_by_target_basis"]
    metrics = []
    event_metrics = []
    predictions = []
    for split in PARTITIONS:
        rows = labeled.loc[labeled["split"].eq(split)].copy()
        if rows.empty:
            continue
        retrospective_exclusions = rows["label_status"].isin(["event_or_post_event", "service_reset_not_yet_recorded"])
        rows.loc[retrospective_exclusions, "feature_status"] = rows.loc[retrospective_exclusions, "label_status"]
        for name, candidate in candidates.items():
            scores = score_rows(rows, candidate["pipeline"], candidate["schema"], candidate["threshold"])
            result, event_rows = metrics_for_split(rows, scores, gap)
            identifiers = {
                "data_kind": args.data_kind, "split": split, "candidate": name, "selected": name == selected,
            }
            metrics.append(identifiers | {"alert_threshold": candidate["threshold"]} | result)
            event_rows = event_rows.drop(columns="service_start")
            event_metrics.append(event_rows.assign(**identifiers))
            predictions.append(scores.assign(
                **identifiers, label=rows["label"], label_status=rows["label_status"], feature_status=rows["feature_status"],
            ))
    scored = pd.concat(predictions, ignore_index=True)
    write_csv(scored, directory / "sample_predictions.csv")
    write_csv(pd.concat(event_metrics, ignore_index=True), directory / "event_evaluation.csv")
    write_csv(pd.DataFrame(metrics), directory / "table6_classification_report.csv")
    write_json(directory / "table5_best_hyperparameters.json", {
        "data_kind": args.data_kind, "selection": "maximum validation PR-AUC then F1",
        "threshold_selection": "maximum validation sample F1, ties prefer higher threshold",
        "n_trials_per_candidate": args.n_trials, "n_estimators": args.n_estimators, "seed": args.seed,
        "selected_candidate": selected, "comparisons": rank,
    })
    plot_precision_recall(scored, directory / "fig9_precision_recall_curve.png", args.data_kind)
    with NamedTemporaryFile(dir=directory, prefix=".model_bundle.", suffix=".tmp", delete=False) as temporary:
        staging = Path(temporary.name)
    try:
        joblib.dump(metadata | {"pipeline": best["pipeline"]}, staging, compress=3)
        staging.replace(directory / "model_bundle.joblib")
    finally:
        staging.unlink(missing_ok=True)
    write_json(directory / "model_bundle.json", metadata)
    audit["training_status"] = "complete"
    write_json(audit_path, audit)
    LOGGER.info("Selected %s on validation. Results and frozen bundle written to %s.", selected, directory)
    return 0


def score(args) -> int:
    bundle = load_bundle(args.bundle)
    config, settings = bundle["input_config"], bundle["settings"]
    telemetry, _ = read_telemetry(args.telemetry, config, scoring=True)
    log = read_maintenance(args.maintenance, config)
    resets, events, _, _ = maintenance_timelines(log)
    features = feature_frame(telemetry, resets, events, settings)
    schema = [field["name"] for field in bundle["feature_schema"]]
    forecasts = score_rows(features, bundle["pipeline"], schema, bundle["alert_threshold"])
    keep = (
        forecasts["timestamp"].ge(pd.to_datetime(args.from_timestamp, utc=True))
        if args.from_timestamp else slice(None)
    )
    write_csv(forecasts.loc[keep], args.output)
    if args.heading_output:
        write_csv(casing_heading(features, settings["heading"]).loc[keep], args.heading_output)
    LOGGER.info("Wrote %d forecasts from the %s bundle to %s.", len(forecasts.loc[keep]), bundle["data_kind"], args.output)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.action == "train":
            return train(args)
        if args.action == "score":
            return score(args)
        directory = args.output_dir or args.root / "dataset" / "synthetic"
        write_fixture(directory)
        LOGGER.info("Wrote synthetic telemetry and maintenance fixture to %s.", directory)
        return 0
    except (OSError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
