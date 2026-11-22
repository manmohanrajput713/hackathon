"""Sample and episode metrics for the validation and independent test partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from .model import eligible
from .preprocessing import KEYS, SERVICE


EPISODE_KEYS = ["well_id", "service_start", "event_at", "target_basis"]
PARTITIONS = ["validation", "chronological_test", "held_well_test"]


def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for split, rows in frame.groupby("split", sort=False):
        usable = eligible(rows)
        opportunities = rows.loc[rows["label"].eq(1), EPISODE_KEYS].drop_duplicates()
        scored_events = rows.loc[usable & rows["label"].eq(1), EPISODE_KEYS].drop_duplicates()
        records.append({
            "split": split, "rows": len(rows), "wells": rows["well_id"].nunique(),
            "well_service_intervals": len(rows[SERVICE].drop_duplicates()),
            "first_timestamp": rows["timestamp"].min(), "last_timestamp": rows["timestamp"].max(),
            "unknown_label_rows": int(rows["label"].isna().sum()),
            "unavailable_labeled_rows": int((rows["label"].notna() & ~usable).sum()),
            "positive_rows": int((usable & rows["label"].eq(1)).sum()),
            "negative_rows": int((usable & rows["label"].eq(0)).sum()),
            "events_with_labeled_opportunity": len(opportunities),
            "detection_or_failure_events": int(opportunities["target_basis"].eq("detection_or_failure").sum()),
            "changeout_events": int(opportunities["target_basis"].eq("changeout").sum()),
            "events_with_scored_opportunity": len(scored_events),
        })
    return pd.DataFrame(records)


def metrics_for_split(
    rows: pd.DataFrame, scores: pd.DataFrame, max_gap: pd.Timedelta,
) -> tuple[dict, pd.DataFrame]:
    measured = rows.join(scores[["risk_score", "alert_status"]])
    scorable = measured["risk_score"].notna() & measured["label"].notna()
    alerted = measured["alert_status"].eq("alert")
    is_positive = measured["label"].eq(1)
    is_negative = measured["label"].eq(0)
    tp = int((scorable & alerted & is_positive).sum())
    fp = int((scorable & alerted & is_negative).sum())
    positives = int((scorable & is_positive).sum())
    negatives = int((scorable & is_negative).sum())
    pr_auc = (
        float(average_precision_score(measured.loc[scorable, "label"], measured.loc[scorable, "risk_score"]))
        if positives and negatives else None
    )

    first_alerts = (
        measured.loc[scorable & alerted & is_positive].groupby(EPISODE_KEYS)["timestamp"]
        .min().rename("first_alert_at")
    )
    score_counts = (
        measured.loc[scorable & is_positive].groupby(EPISODE_KEYS)["timestamp"]
        .size().rename("scored_forecasts")
    )
    episodes = (
        measured.loc[is_positive].groupby(EPISODE_KEYS)["timestamp"].size().rename("labeled_opportunities")
        .to_frame().join(score_counts).join(first_alerts).reset_index()
    )
    episodes["scored_forecasts"] = episodes["scored_forecasts"].fillna(0).astype(int)
    episodes["detected"] = episodes["first_alert_at"].notna()
    leads = (episodes["event_at"] - episodes["first_alert_at"]) / pd.Timedelta(hours=1)
    episodes["lead_time_hours"] = leads

    grouped = measured.assign(scorable=scorable, false_alert=scorable & is_negative & alerted).groupby(SERVICE)
    duration = grouped["timestamp"].diff()
    contiguous = duration.notna() & duration.le(max_gap)
    previous_scorable = grouped["scorable"].shift(fill_value=False)
    previous_false_alert = grouped["false_alert"].shift(fill_value=False)
    exposure_weeks = float(duration.loc[contiguous & previous_scorable & scorable].sum() / pd.Timedelta(weeks=1))
    false_episodes = int((scorable & is_negative & alerted & ~(contiguous & previous_false_alert)).sum())
    detected = int(episodes["detected"].sum())
    return {
        "wells": int(measured["well_id"].nunique()),
        "scored_wells": int(measured.loc[scorable, "well_id"].nunique()),
        "scored_rows": positives + negatives,
        "positive_rows": positives,
        "negative_rows": negatives,
        "pr_auc": pr_auc,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None,
        "events": len(episodes),
        "events_detected": detected,
        "event_level_recall": detected / len(episodes) if len(episodes) else None,
        "false_alert_episodes": false_episodes,
        "observed_well_weeks": exposure_weeks,
        "false_alerts_per_well_week": false_episodes / exposure_weeks if exposure_weeks else None,
        "median_lead_time_hours": float(leads.median()) if detected else None,
        "true_positive_rows": tp,
        "false_positive_rows": fp,
    }, episodes


def plot_precision_recall(predictions: pd.DataFrame, path, data_kind: str) -> None:
    import matplotlib.pyplot as plt

    splits = predictions["split"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 4.5), squeeze=False)
    for axis, split in zip(axes[0], splits):
        rows = predictions.loc[predictions["split"].eq(split)].dropna(subset=["label", "risk_score"])
        for candidate, candidate_rows in rows.groupby("candidate", sort=False):
            y = candidate_rows["label"].astype(int)
            if y.nunique() < 2:
                continue
            p, r, _ = precision_recall_curve(y, candidate_rows["risk_score"])
            ap = average_precision_score(y, candidate_rows["risk_score"])
            axis.step(r, p, where="post", label=f"{candidate} ({ap:.3f})")
        if axis.lines:
            axis.legend(loc="lower left", fontsize=7)
        else:
            axis.text(0.5, 0.5, "Both classes required", ha="center")
        axis.set(title=split.replace("_", " ").capitalize(), xlabel="Recall", ylabel="Precision",
                 xlim=(0, 1.02), ylim=(0, 1.03))
        axis.grid(alpha=0.2)
    fig.suptitle(f"Precision–recall curves · {data_kind.capitalize()} data")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
