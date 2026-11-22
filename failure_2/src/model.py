"""Purged chronological splits, whole-well holdout, and frozen classifier selection."""

from __future__ import annotations

import logging
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .features import OPTIONAL_CATEGORICAL, FEATURE_UNITS, model_inputs, select_features
from .preprocessing import KEYS


LOGGER = logging.getLogger(__name__)
RANDOM_STATE = 42
BUNDLE_VERSION = 1
LIBRARIES = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")
MODEL_NAMES = ("random_forest", "xgboost")


def ready(frame: pd.DataFrame) -> pd.Series:
    return frame["feature_status"].eq("ready")


def eligible(frame: pd.DataFrame) -> pd.Series:
    return ready(frame) & frame["label"].notna()


def assign_splits(
    frame: pd.DataFrame,
    horizon: pd.Timedelta,
    holdout_fraction: float,
    validation_start: pd.Timestamp | None,
    test_start: pd.Timestamp | None,
    seed: int,
) -> tuple[pd.Series, dict, bool]:
    """Reserve wells first, then use two embargoed cuts over the development wells."""
    times = frame["timestamp"]
    start, end = times.min(), times.max()
    validation_start = validation_start if validation_start is not None else start + (end - start) * 0.55
    test_start = test_start if test_start is not None else start + (end - start) * 0.78
    if not start < validation_start < test_start <= end:
        raise ValueError("Require telemetry start < validation start < test start <= telemetry end")
    pre_val = times.lt(validation_start - horizon)
    validation = times.ge(validation_start) & times.lt(test_start - horizon)
    future = times.ge(test_start)
    used = eligible(frame)
    wells = sorted(frame.loc[future, "well_id"].unique())
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout-fraction must be in [0, 1)")

    # Check only whether development training and validation retain both classes.
    label_available = frame.get("label_available_at", times)
    train_ready = used & pre_val & label_available.lt(validation_start)
    val_ready = used & validation & label_available.lt(test_start)
    train_classes = {label: set(frame.loc[train_ready & frame["label"].eq(label), "well_id"])
                     for label in (0, 1)}
    val_classes = {label: set(frame.loc[val_ready & frame["label"].eq(label), "well_id"])
                   for label in (0, 1)}
    requirements = [*train_classes.values(), *val_classes.values()]
    supports_fit = all(requirements)
    held = set()
    holdout_status = "disabled_by_configuration"
    if holdout_fraction > 0:
        holdout_status = "insufficient_development_wells_or_classes"
        if wells and supports_fit and frame["well_id"].nunique() > 1:
            rng = np.random.default_rng(seed)
            count = min(max(1, len(wells) - 1), max(1, round(len(wells) * holdout_fraction)))
            for size in range(count, 0, -1):
                draws = ({well} for well in rng.permutation(wells)) if size == 1 else (
                    set(rng.choice(wells, size=size, replace=False)) for _ in range(64)
                )
                for candidate in draws:
                    if all(options - candidate for options in requirements):
                        held = candidate
                        break
                if held:
                    holdout_status = "held_out"
                    break

    held_mask = frame["well_id"].isin(held)
    split = pd.Series("embargo", index=frame.index, dtype=object)
    split.loc[pre_val & ~held_mask] = "train"
    split.loc[validation & ~held_mask] = "validation"
    split.loc[future & ~held_mask] = "chronological_test"
    split.loc[held_mask] = "held_well_history"
    split.loc[future & held_mask] = "held_well_test"
    return split, {
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "embargo_days": horizon / pd.Timedelta(days=1),
        "holdout_status": holdout_status,
        "requested_holdout_fraction": holdout_fraction,
        "holdout_candidate_wells": len(wells),
        "holdout_wells": sorted(held),
        "development_wells": sorted(set(frame["well_id"]) - held),
    }, supports_fit


def best_f1_threshold(labels, scores) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    # Prefer the higher threshold when F1 is tied.
    index = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[index]), float(f1[index])


def suggest(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "random_forest":
        return {
            "max_depth": trial.suggest_categorical("max_depth", [5, 9, 15, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 25, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.75, 1.0]),
        }
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }


def make_pipeline(model_name: str, schema: list[str], labels: pd.Series, params: dict, args) -> Pipeline:
    numeric = [name for name in schema if name not in OPTIONAL_CATEGORICAL]
    categorical = [name for name in schema if name in OPTIONAL_CATEGORICAL]
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    shared = {"n_estimators": args.n_estimators, "n_jobs": args.jobs, "random_state": args.seed}
    if model_name == "random_forest":
        classifier = RandomForestClassifier(**shared, **params, class_weight="balanced")
    else:
        classifier = XGBClassifier(
            **shared, **params, scale_pos_weight=float(labels.eq(0).sum() / labels.eq(1).sum()),
            tree_method="hist", objective="binary:logistic", eval_metric="aucpr", verbosity=0,
        )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def fit_candidates(train: pd.DataFrame, validation: pd.DataFrame, args) -> tuple[dict, list[dict]]:
    """Compare both classifiers on the same rows and pressure features first."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    variants = {"paper_baseline": select_features(train, extended=False)}
    extended = select_features(train, extended=True)
    if args.feature_set == "extended" and extended != variants["paper_baseline"]:
        variants["extended"] = extended
    candidates = {}
    search_results = []
    y_train = train["label"].astype(int)
    y_val = validation["label"].astype(int)
    for variant, schema in variants.items():
        x_train = model_inputs(train, schema)
        x_val = model_inputs(validation, schema)
        for name in MODEL_NAMES:
            identifier = f"{variant}/{name}"

            def objective(trial):
                params = suggest(trial, name)
                pipeline = make_pipeline(name, schema, y_train, params, args)
                pipeline.fit(x_train, y_train)
                probability = pipeline.predict_proba(x_val)[:, 1]
                return float(average_precision_score(y_val, probability))

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
            study.optimize(objective, n_trials=args.n_trials)
            pipeline = make_pipeline(name, schema, y_train, study.best_params, args)
            pipeline.fit(x_train, y_train)
            probability = pipeline.predict_proba(x_val)[:, 1]
            threshold, f1 = best_f1_threshold(y_val, probability)
            candidates[identifier] = {"pipeline": pipeline, "schema": schema, "threshold": threshold}
            search_results.append({
                "candidate": identifier, "model": name, "feature_set": variant,
                "feature_schema": schema, "best_params": study.best_params,
                "validation_pr_auc": study.best_value, "validation_f1_at_threshold": f1,
                "alert_threshold": threshold,
                "trials": [{"params": t.params, "validation_pr_auc": t.value} for t in study.trials],
            })
            LOGGER.info("%s validation PR-AUC %.4f. Threshold %.4f.", identifier, study.best_value, threshold)
    return candidates, search_results


def score_rows(frame: pd.DataFrame, pipeline: Pipeline, schema: list[str], threshold: float) -> pd.DataFrame:
    scores = frame[KEYS].copy()
    scores["risk_score"] = np.nan
    scores["alert_status"] = "unavailable"
    available = ready(frame)
    if available.any():
        predicted = pipeline.predict_proba(model_inputs(frame.loc[available], schema))[:, 1]
        scores.loc[available, "risk_score"] = predicted
        scores.loc[available, "alert_status"] = np.where(predicted >= threshold, "alert", "no_alert")
    return scores


def bundle_metadata(selected: str, candidate: dict, settings: dict, config: dict, data_kind: str) -> dict:
    names = candidate["schema"]
    return {
        "bundle_version": BUNDLE_VERSION,
        "selected_candidate": selected,
        "data_kind": data_kind,
        "feature_schema": [{"name": name, "unit": FEATURE_UNITS[name]} for name in names],
        "alert_threshold": candidate["threshold"],
        "settings": settings,
        "input_config": config,
        "library_versions": {name: version(name) for name in LIBRARIES},
    }


def load_bundle(path: Path) -> dict:
    bundle = joblib.load(path)
    if bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"Unsupported model bundle version in {path}")
    return bundle
