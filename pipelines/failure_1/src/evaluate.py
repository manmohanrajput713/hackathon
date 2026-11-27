"""KDD phase 5: ROC-AUC, the precision-recall curve, and the F1 report.

Equations (14) and (15) are precision and recall. Equation (16) is the true
positive rate, which equals recall. Equation (17) is the false positive rate.
The F1-score is the harmonic mean of precision and recall. Figure 8 is the
10-fold ROC. Figure 9 and Table 6 are computed on the ESP holdout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)


def precision_from_counts(tp: float, fp: float) -> float:
    """Equation (14): TP / (TP + FP)."""
    return float(tp / (tp + fp))


def recall_from_counts(tp: float, fn: float) -> float:
    """Equation (15), also the true positive rate in equation (16)."""
    return float(tp / (tp + fn))


def false_positive_rate(fp: float, tn: float) -> float:
    """Equation (17): FP / (FP + TN)."""
    return float(fp / (fp + tn))


def f1_from_precision_recall(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    return float(2.0 * precision * recall / (precision + recall))


def best_f1_threshold(y_true, probability) -> tuple[float, float]:
    """Operating point that maximizes positive-class F1 on out-of-fold scores.

    The threshold is chosen on training folds and then frozen for the holdout.
    """
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(probability, dtype=float)
    valid = np.isfinite(scores)
    precision, recall, thresholds = precision_recall_curve(labels[valid], scores[valid])
    if thresholds.size == 0:
        return 0.5, 0.0
    f1 = np.zeros_like(thresholds, dtype=float)
    denominator = precision[:-1] + recall[:-1]
    usable = denominator > 0
    f1[usable] = 2.0 * precision[:-1][usable] * recall[:-1][usable] / denominator[usable]
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def plot_fig8_roc(curves: list[dict], path) -> float:
    """Figure 8: one ROC per fold, plus the mean curve."""
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    mean_fpr = np.linspace(0.0, 1.0, 201)
    tprs = []
    aucs = []
    for curve in curves:
        fpr, tpr, _ = roc_curve(curve["y"], curve["probability"])
        score = float(auc(fpr, tpr))
        aucs.append(score)
        axis.plot(fpr, tpr, color="#4C78A8", alpha=0.35, linewidth=1.1)
        interpolated = np.interp(mean_fpr, fpr, tpr)
        interpolated[0] = 0.0
        tprs.append(interpolated)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    axis.plot(
        mean_fpr,
        mean_tpr,
        color="#E45756",
        linewidth=2.2,
        label=f"Mean ROC (AUC = {mean_auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#333333", linewidth=1, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Figure 8. ROC curves for 10 group folds")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, loc="lower right")
    axis.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return mean_auc


def plot_fig9_precision_recall(y_true, probability, path) -> float:
    """Figure 9: precision-recall curve of the holdout positive class."""
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(probability, dtype=float)
    precision, recall, _ = precision_recall_curve(labels, scores)
    average_precision = float(average_precision_score(labels, scores))
    prevalence = float(labels.mean())
    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    axis.plot(
        recall,
        precision,
        color="#E45756",
        linewidth=2.0,
        label=f"Pre-event (AP = {average_precision:.3f})",
    )
    axis.hlines(
        prevalence,
        0,
        1,
        colors="#333333",
        linestyles="--",
        linewidth=1,
        label=f"Prevalence ({prevalence:.3f})",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title("Figure 9. Precision-recall curve for the 7-day pre-event class")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return average_precision


def classification_table(y_true, probability, threshold: float) -> pd.DataFrame:
    """Table 6 rows, plus the macro and weighted averages."""
    labels = np.asarray(y_true).astype(int)
    predicted = (np.asarray(probability, dtype=float) >= threshold).astype(int)
    report = classification_report(
        labels,
        predicted,
        labels=[0, 1],
        target_names=["normal", "7 days or less pre-event"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for name in ["normal", "7 days or less pre-event", "macro avg", "weighted avg"]:
        entry = report[name]
        rows.append(
            {
                "label": name,
                "precision": entry["precision"],
                "recall": entry["recall"],
                "f1-score": entry["f1-score"],
                "support": entry["support"],
            }
        )
    return pd.DataFrame(rows)
