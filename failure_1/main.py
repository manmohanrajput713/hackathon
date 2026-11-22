"""Run the Abdalla et al. (2022) KDD pipeline on the PGEHackathon ESP archive."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from src.download_data import download_hackathon, download_paper, schema_check
from src.evaluate import (
    best_f1_threshold,
    classification_table,
    plot_fig8_roc,
    plot_fig9_precision_recall,
)
from src.features import (
    FEATURE_COLUMNS,
    N_COMPONENTS,
    build_feature_matrix,
    plot_fig4_normalized_difference,
    plot_fig5_pca_geometry,
    plot_fig6_pca_projection,
    plot_fig7_explained_variance,
    plot_kdd_workflow,
)
from src.model import (
    N_ESTIMATORS,
    N_SPLITS,
    PC_COLUMNS,
    RANDOM_STATE,
    PUBLISHED_SEARCH_SPACE,
    SEARCH_SPACE,
    TEST_SIZE,
    assign_esp_folds,
    cross_validate,
    fit_final,
    holdout_esp_ids,
    scale_pos_weight,
    tune_hyperparameters,
)
from src.preprocessing import (
    PRE_EVENT_WINDOW_DAYS,
    SENSOR_COLUMNS,
    WELL_CONTEXT,
    class_distribution,
    exploration_table,
    plot_fig2_before_outliers,
    plot_fig3_after_outliers,
    prepare_frames,
)

logger = logging.getLogger("failure_1")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, float_format="%.6f")


def main() -> int:
    args = _parse()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    root = args.root.resolve()
    data_dir = root / "dataset"
    docs_dir = root / "docs"
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_paper(docs_dir / "acsomega.1c05881.pdf", force=args.force_download)
        download_hackathon(data_dir, force=args.force_download)

    daily = pd.read_csv(data_dir / "dailyData.csv")
    well = pd.read_csv(data_dir / "wellData.csv")
    solution = pd.read_csv(data_dir / "solution.csv")
    schema = schema_check(daily, well, solution)

    prepared = prepare_frames(daily, well, solution)
    plot_kdd_workflow(results / "fig1_kdd_workflow.png")
    plot_fig2_before_outliers(prepared.before_outliers, results / "fig2_box_before_outliers.png")
    plot_fig3_after_outliers(prepared.cleaned, results / "fig3_box_after_outliers.png")
    exploration = exploration_table(prepared.cleaned)
    classes = class_distribution(prepared.cleaned)
    _write_table(exploration, results / "table2_data_exploration.csv")
    _write_table(classes, results / "table3_class_distribution.csv")

    test_ids = holdout_esp_ids(prepared.cleaned, test_size=TEST_SIZE, seed=RANDOM_STATE)
    featured = build_feature_matrix(prepared.cleaned, test_ids, n_components=N_COMPONENTS)
    plot_fig4_normalized_difference(featured.frame, results / "fig4_box_normalized_diff.png")
    plot_fig5_pca_geometry(results / "fig5_pca_geometry.png")
    plot_fig6_pca_projection(featured.frame, results / "fig6_pca_projection.png")
    plot_fig7_explained_variance(featured.pca, results / "fig7_pca_explained_variance.png")
    _write_table(featured.loadings, results / "table4_pca_loadings.csv")

    train = featured.frame.loc[featured.frame["split"].eq("train")].reset_index(drop=True)
    test = featured.frame.loc[featured.frame["split"].eq("test")].reset_index(drop=True)
    esp_cohort = train.groupby("esp_id")["cohort"].first()
    fold_map = assign_esp_folds(
        esp_cohort.index.tolist(),
        esp_cohort.tolist(),
        n_splits=N_SPLITS,
        seed=RANDOM_STATE,
    )
    train["cv_fold"] = train["esp_id"].map(fold_map).astype(int)
    x_train = train[PC_COLUMNS].to_numpy(dtype=float)
    y_train = train["label"].to_numpy(dtype=int)
    logger.info(
        "Training matrix %s by %s. Positive rate %.4f. scale_pos_weight %.2f.",
        x_train.shape[0],
        x_train.shape[1],
        float(y_train.mean()),
        scale_pos_weight(y_train),
    )

    best_params, study = tune_hyperparameters(
        x_train,
        y_train,
        train["cv_fold"].to_numpy(),
        n_trials=args.n_trials,
        n_estimators=args.n_estimators,
        seed=RANDOM_STATE,
    )
    validation = cross_validate(
        x_train,
        y_train,
        train["cv_fold"].to_numpy(),
        best_params,
        n_estimators=args.n_estimators,
        store_curves=True,
    )
    for curve in validation.curves:
        logger.info("Reported fold %s ROC-AUC = %.4f.", curve["fold"], curve["auc"])
    mean_auc = plot_fig8_roc(validation.curves, results / "fig8_roc_10fold_cv.png")
    threshold, oof_f1 = best_f1_threshold(validation.oof_label, validation.oof_probability)
    logger.info("OOF F1-maximizing threshold = %.4f (F1 %.4f).", threshold, oof_f1)

    classifier, weight = fit_final(x_train, y_train, best_params, n_estimators=args.n_estimators)
    x_test = test[PC_COLUMNS].to_numpy(dtype=float)
    y_test = test["label"].to_numpy(dtype=int)
    probability = classifier.predict_proba(x_test)[:, 1]
    average_precision = plot_fig9_precision_recall(
        y_test, probability, results / "fig9_precision_recall_curve.png"
    )
    report = classification_table(y_test, probability, threshold)
    _write_table(report, results / "table6_classification_report.csv")
    positive = report.loc[report["label"].eq("7 days or less pre-event")].iloc[0]
    from sklearn.metrics import roc_auc_score

    test_auc = float(roc_auc_score(y_test, probability))
    card = {
        "reference": "Abdalla et al., ACS Omega 2022, 7, 17641-17651, doi:10.1021/acsomega.1c05881",
        "published_search_space": PUBLISHED_SEARCH_SPACE,
        "search_space": SEARCH_SPACE,
        "best_hyperparameters": {
            "max_depth": int(best_params["max_depth"]),
            "min_child_weight": float(best_params["min_child_weight"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "learning_rate": float(best_params["learning_rate"]),
            "gamma": int(best_params["gamma"]),
        },
        "fixed_parameters": {
            "n_estimators": int(args.n_estimators),
            "objective": "binary:logistic",
            "tree_method": "hist",
            "scale_pos_weight": weight,
            "n_components": N_COMPONENTS,
            "decision_threshold": threshold,
            "pre_event_window_days": PRE_EVENT_WINDOW_DAYS,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "feature_columns": FEATURE_COLUMNS,
            "model_columns": PC_COLUMNS,
        },
        "optuna": {
            "n_trials": int(args.n_trials),
            "direction": "maximize",
            "objective": "mean ROC-AUC over 10 ESP-grouped folds",
            "best_value": float(study.best_value),
        },
        "cross_validation": {
            "fold_auc": validation.fold_auc,
            "mean_auc": validation.mean_auc,
            "mean_curve_auc": mean_auc,
            "oof_best_f1": oof_f1,
        },
        "test": {
            "roc_auc": test_auc,
            "average_precision": average_precision,
            "positive_precision": float(positive["precision"]),
            "positive_recall": float(positive["recall"]),
            "positive_f1": float(positive["f1-score"]),
            "support_positive": float(positive["support"]),
        },
    }
    card_path = results / "table5_best_hyperparameters.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    test = test.copy()
    test["cv_fold"] = pd.Series(pd.NA, index=test.index, dtype="Int64")
    train["cv_fold"] = train["cv_fold"].astype("Int64")
    modeled = pd.concat([train, test], ignore_index=True)
    keep = [
        "Well_ID",
        "AL_Key",
        "esp_id",
        "day_index",
        "days_to_event",
        "cohort",
        "condition",
        "label",
        "split",
        "cv_fold",
        "OIL",
        *[column for column in WELL_CONTEXT if column in modeled.columns],
        *[f"raw_{column}" for column in SENSOR_COLUMNS],
        *SENSOR_COLUMNS,
        *[f"diff_{column}" for column in SENSOR_COLUMNS],
        *PC_COLUMNS,
    ]
    parquet_path = data_dir / "processed_esp_data.parquet"
    modeled.loc[:, keep].to_parquet(parquet_path, index=False)

    audit = {
        "schema": schema,
        "preprocess": prepared.audit,
        "fences": prepared.fences,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_esps": int(train["esp_id"].nunique()),
        "test_esps": int(test["esp_id"].nunique()),
        "pca_explained_variance_ratio": featured.pca.explained_variance_ratio_[:N_COMPONENTS].tolist(),
        "pca_cumulative_8": float(featured.pca.explained_variance_ratio_[:N_COMPONENTS].sum()),
    }
    (results / "processing_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info(
        "Holdout positive F1 %.3f, precision %.3f, recall %.3f, ROC-AUC %.3f, AP %.3f.",
        positive["f1-score"],
        positive["precision"],
        positive["recall"],
        test_auc,
        average_precision,
    )
    logger.info("Mean 10-fold ROC-AUC %.3f. Artifacts written to %s.", validation.mean_auc, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
