"""KDD phase 4: group holdout, Optuna search, and 10-fold XGBoost.

Equations (6) through (13) are Chen and Guestrin's XGBoost objective. The
leaf weight, similarity, and gain below are those formulas. The fitted model
delegates the same algebra to the XGBoost library.

Table 5 fixes the search space: max_depth, min_child_weight,
colsample_bytree, learning_rate, and gamma. A column subsample or a learning
rate of exactly zero does not define a tree update, so those two uniform
ranges are open at zero. The paper splits rows at random. This module splits
whole ESP installations so a well's history cannot sit on both sides.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from xgboost.core import XGBoostError

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SPLITS = 10
TEST_SIZE = 0.30
N_ESTIMATORS = 200
N_JOBS = min(4, os.cpu_count() or 1)
PC_COLUMNS = [f"PC{i + 1}" for i in range(8)]

# Published Table 5 bounds. min_child_weight is log-uniform on the printed range.
PUBLISHED_SEARCH_SPACE = {
    "max_depth": {"sampling": "int", "low": 2, "high": 10},
    "min_child_weight": {"sampling": "log_uniform", "low": 1e-10, "high": 1e10},
    "colsample_bytree": {"sampling": "uniform", "low": 0.0, "high": 1.0},
    "learning_rate": {"sampling": "uniform", "low": 0.0, "high": 0.1},
    "gamma": {"sampling": "int", "low": 0, "high": 5},
}
# Interior actually searched. A minimum child weight above the sum of the
# Hessians cannot split a node, and a column fraction or learning rate of
# zero does not define an update. Every bound below sits inside Table 5.
SEARCH_SPACE = {
    "max_depth": {"sampling": "int", "low": 2, "high": 10},
    "min_child_weight": {"sampling": "log_uniform", "low": 1e-2, "high": 20.0},
    "colsample_bytree": {"sampling": "uniform", "low": 0.5, "high": 1.0},
    "learning_rate": {"sampling": "uniform", "low": 0.02, "high": 0.1},
    "gamma": {"sampling": "int", "low": 0, "high": 5},
}


def optimal_leaf_weight(gradient, hessian, reg_lambda: float) -> float:
    """Equation (10): w* = -sum(g) / (sum(h) + lambda)."""
    return float(-np.sum(gradient) / (np.sum(hessian) + reg_lambda))


def similarity_score(gradient, hessian, reg_lambda: float) -> float:
    """Equation (12): (sum g)^2 / (sum h + lambda).

    The paper defines cover as this denominator minus lambda, which is the
    sum of the Hessians.
    """
    return float((np.sum(gradient) ** 2) / (np.sum(hessian) + reg_lambda))


def split_gain(gradient, hessian, left_mask, reg_lambda: float, gamma: float) -> float:
    """Split gain from Chen and Guestrin (2016), equation (13).

    Gain = 1/2 (S_left + S_right - S_parent) - gamma, where S is the
    similarity score. The one-half and gamma terms are the second-order
    objective after the constant part of the Taylor expansion is removed
    (equations 8 and 9).
    """
    left = np.asarray(left_mask, dtype=bool)
    parent = similarity_score(gradient, hessian, reg_lambda)
    left_score = similarity_score(np.asarray(gradient)[left], np.asarray(hessian)[left], reg_lambda)
    right_score = similarity_score(np.asarray(gradient)[~left], np.asarray(hessian)[~left], reg_lambda)
    return float(0.5 * (left_score + right_score - parent) - gamma)


def holdout_esp_ids(
    cleaned: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
) -> set[str]:
    """Hold out entire ESPs. Stratify on the failed and survivor cohorts."""
    labeled = cleaned.loc[cleaned["label"].notna(), ["esp_id", "cohort"]].drop_duplicates("esp_id")
    if labeled.empty:
        raise RuntimeError("No labeled ESPs are available for the holdout.")
    rng = np.random.default_rng(seed)
    test_ids: set[str] = set()
    for cohort in sorted(labeled["cohort"].unique()):
        members = sorted(labeled.loc[labeled["cohort"].eq(cohort), "esp_id"])
        if len(members) < 2:
            continue
        n_test = int(round(len(members) * test_size))
        n_test = min(max(n_test, 1), len(members) - 1)
        chosen = rng.permutation(len(members))[:n_test]
        test_ids.update(members[int(i)] for i in chosen)
    if not test_ids or len(test_ids) == len(labeled):
        raise RuntimeError("The group holdout did not produce both a train and a test set.")
    logger.info(
        "Group holdout: %s test ESPs and %s train ESPs.",
        len(test_ids),
        len(labeled) - len(test_ids),
    )
    return test_ids


def assign_esp_folds(
    esp_ids: list[str],
    cohorts: list[str],
    n_splits: int = N_SPLITS,
    seed: int = RANDOM_STATE,
) -> dict[str, int]:
    """Assign each ESP to a fold, stratified by cohort."""
    if n_splits < 2:
        raise ValueError("Cross-validation needs at least two folds.")
    rng = np.random.default_rng(seed)
    folds: dict[str, int] = {}
    paired = list(dict.fromkeys(zip(esp_ids, cohorts, strict=True)))
    for cohort in sorted({cohort for _, cohort in paired}):
        members = sorted(esp for esp, group in paired if group == cohort)
        order = rng.permutation(len(members))
        for position, index in enumerate(order):
            folds[members[int(index)]] = int(position % n_splits)
    return folds


def scale_pos_weight(y) -> float:
    """Negative-to-positive ratio. The paper declines SMOTE and keeps the imbalance explicit."""
    labels = np.asarray(y).astype(int)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise RuntimeError("scale_pos_weight needs both classes.")
    return negatives / positives


def make_classifier(params: dict, class_weight: float, n_estimators: int = N_ESTIMATORS) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        colsample_bytree=float(params["colsample_bytree"]),
        learning_rate=float(params["learning_rate"]),
        gamma=float(params["gamma"]),
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=float(class_weight),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=0,
    )


def _suggest(trial: optuna.Trial) -> dict:
    space = SEARCH_SPACE
    return {
        "max_depth": trial.suggest_int("max_depth", space["max_depth"]["low"], space["max_depth"]["high"]),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            space["min_child_weight"]["low"],
            space["min_child_weight"]["high"],
            log=True,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            space["colsample_bytree"]["low"],
            space["colsample_bytree"]["high"],
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            space["learning_rate"]["low"],
            space["learning_rate"]["high"],
        ),
        "gamma": trial.suggest_int("gamma", space["gamma"]["low"], space["gamma"]["high"]),
    }


@dataclass
class CVResult:
    fold_auc: list[float]
    mean_auc: float
    oof_probability: np.ndarray
    oof_label: np.ndarray
    curves: list[dict] = field(default_factory=list)


def cross_validate(
    X,
    y,
    fold_ids,
    params: dict,
    n_estimators: int = N_ESTIMATORS,
    store_curves: bool = True,
) -> CVResult:
    """Ten group folds. Each left-out fold contributes one ROC."""
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y).astype(int)
    folds = np.asarray(fold_ids).astype(int)
    oof = np.full(labels.shape, np.nan, dtype=float)
    fold_auc: list[float] = []
    curves: list[dict] = []
    for fold in range(int(folds.min()), int(folds.max()) + 1):
        test_mask = folds == fold
        train_mask = ~test_mask
        if test_mask.sum() == 0 or len(np.unique(labels[test_mask])) < 2:
            logger.warning("Fold %s does not contain both classes and was skipped.", fold)
            continue
        if len(np.unique(labels[train_mask])) < 2:
            logger.warning("Fold %s has a single-class training side and was skipped.", fold)
            continue
        classifier = make_classifier(params, scale_pos_weight(labels[train_mask]), n_estimators)
        classifier.fit(features[train_mask], labels[train_mask])
        probability = classifier.predict_proba(features[test_mask])[:, 1]
        oof[test_mask] = probability
        score = float(roc_auc_score(labels[test_mask], probability))
        fold_auc.append(score)
        if store_curves:
            curves.append(
                {"fold": int(fold), "y": labels[test_mask], "probability": probability, "auc": score}
            )
        logger.debug("Fold %s ROC-AUC = %.4f on %s rows.", fold, score, int(test_mask.sum()))
    if not fold_auc:
        raise RuntimeError("Every cross-validation fold was skipped.")
    return CVResult(
        fold_auc=fold_auc,
        mean_auc=float(np.mean(fold_auc)),
        oof_probability=oof,
        oof_label=labels,
        curves=curves,
    )


def tune_hyperparameters(
    X,
    y,
    fold_ids,
    n_trials: int,
    n_estimators: int = N_ESTIMATORS,
    seed: int = RANDOM_STATE,
) -> tuple[dict, optuna.Study]:
    """Maximize the mean 10-fold ROC-AUC inside the Table 5 ranges."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        try:
            result = cross_validate(
                X, y, fold_ids, params, n_estimators=n_estimators, store_curves=False
            )
        except XGBoostError as exc:
            logger.warning("Trial %s was rejected by XGBoost: %s", trial.number, exc)
            return 0.5
        return result.mean_auc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _log_trial(study, trial) -> None:
        logger.info(
            "Trial %s mean ROC-AUC %.4f. Best so far %.4f.",
            trial.number,
            trial.value,
            study.best_value,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[_log_trial])
    logger.info(
        "Best mean ROC-AUC %.4f with %s.",
        study.best_value,
        study.best_params,
    )
    return dict(study.best_params), study


def fit_final(X, y, params: dict, n_estimators: int = N_ESTIMATORS) -> tuple[XGBClassifier, float]:
    weight = scale_pos_weight(y)
    classifier = make_classifier(params, weight, n_estimators)
    classifier.fit(np.asarray(X, dtype=float), np.asarray(y).astype(int))
    return classifier, weight
