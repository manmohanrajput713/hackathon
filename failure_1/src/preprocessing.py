"""KDD phases 1 and 2: merge, failure labels, zero-production and IQR filters.

Abdalla et al. resample sensor streams with a one-hour moving median, drop
zero-oil measurements, and reject Tukey outliers. Equations (1) and (2) in
the paper are the lower and upper fences around the interquartile range.
The published wording also says the subsequent scaler divides by the
variance. Unit variance, which the same paragraph requires, is obtained by
dividing by the standard deviation. That scaler lives in ``features.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.download_data import SOLUTION_LABEL

logger = logging.getLogger(__name__)

# Eight dynamic channels. Five match the paper's tags exactly. Wellhead
# pressure, wellhead temperature, and casing-head pressure are not in the
# hackathon feed, so the remaining three slots are the measured pressure,
# temperature, and electrical channels that play the same role in the
# 8-level + 8-difference feature vector.
SENSOR_SOURCE = {
    "FRQ": "ESP Data - Output Frequency",
    "PDP": "Discharge_Pressure",
    "PIP": "ESP Data - Intake Pressure",
    "PAVG": "Pump_Average_Pressure",
    "MT": "ESP Data - Motor Winding Temperature",
    "ESPT": "ESP_Temperature",
    "CURRENT": "ESP Data - Drive Current",
    "VOLT": "ESP Data - Drive Voltage",
}
SENSOR_COLUMNS = list(SENSOR_SOURCE)
DISPLAY_NAMES = {
    "FRQ": "FRQ (Hz)",
    "PDP": "PDP (psi)",
    "PIP": "PIP (psi)",
    "PAVG": "PAVG (psi)",
    "MT": "MT (F)",
    "ESPT": "ESPT (F)",
    "CURRENT": "CURRENT (A)",
    "VOLT": "VOLT (V)",
}
# Inclusive physical limits applied before the statistical fences.
# Frequencies, pressures, and temperatures of zero are unphysical here.
# A drive can report zero current or voltage while a daily volume is allocated.
UNPHYSICAL_BOUNDS = {
    "FRQ": (0.0, 100.0, False),
    "PDP": (0.0, 20_000.0, False),
    "PIP": (0.0, 20_000.0, False),
    "PAVG": (0.0, 20_000.0, False),
    "MT": (0.0, 500.0, False),
    "ESPT": (0.0, 500.0, False),
    "CURRENT": (0.0, 2_000.0, True),
    "VOLT": (0.0, 15_000.0, True),
}
WELL_CONTEXT = [
    "Artificial_Lift_Type",
    "AL_Bottom_Depth",
    "ESP_Pump_Stages",
    "DLS_Critical",
    "ESP_Motor_Frequency_Rating",
    "ESP_Motor_Current_Rating",
    "ESP_Motor_Voltage_Rating",
    "ESP_Motor_Power_Rating",
    "DLS_at_Set_Depth",
]
# Positive class: the workover day and the preceding seven daily rows.
# Table 6 calls this pooled label "7 days or less pre-event".
PRE_EVENT_WINDOW_DAYS = 7
IQR_K = 1.5


@dataclass
class PreprocessResult:
    before_outliers: pd.DataFrame
    cleaned: pd.DataFrame
    audit: dict = field(default_factory=dict)
    fences: dict = field(default_factory=dict)


def iqr_bounds(values, k: float = IQR_K) -> tuple[float, float]:
    """Tukey fences, equations (1) and (2).

    lower = Q1 - k (Q3 - Q1)
    upper = Q3 + k (Q3 - Q1)
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("IQR fences require at least one finite value.")
    q1, q3 = np.quantile(array, [0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def resample_moving_median(df: pd.DataFrame, timestamp_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Median of each sensor inside successive one-hour bins.

    The paper resamples higher-frequency tags with a moving median in one-hour
    steps. The hackathon extract is already one row per day and has no
    timestamp, so the production path leaves those rows unchanged. This
    function is the hourly step, covered by a unit test.
    """
    frame = df.copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    frame = frame.sort_values(timestamp_col).set_index(timestamp_col)
    return frame[value_cols].resample("1h").median()


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Well_ID"] = pd.to_numeric(out["Well_ID"], errors="raise").astype(int)
    out["AL_Key"] = out["AL_Key"].astype(str).str.strip()
    out["esp_id"] = out["Well_ID"].astype(str) + "|" + out["AL_Key"]
    return out


def to_canonical_sensors(daily: pd.DataFrame) -> pd.DataFrame:
    """Rename hackathon tags to the short names used in the paper's tables."""
    out = daily.copy()
    rename = {}
    for canonical, source in SENSOR_SOURCE.items():
        if canonical in out.columns:
            continue
        if source not in out.columns:
            raise KeyError(f"Neither {canonical} nor {source!r} is present.")
        rename[source] = canonical
    out = out.rename(columns=rename)
    for column in ["OIL", *SENSOR_COLUMNS]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def assign_day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Daily time index from file order within each ESP.

    ``dailyData.csv`` has no calendar column. The repository states that each
    installed ESP contributes one row per day, and the file stores that
    sequence in order.
    """
    out = df.copy()
    out["day_index"] = out.groupby("esp_id", sort=False).cumcount().astype(int)
    return out


def attach_failure_labels(df: pd.DataFrame, solution: pd.DataFrame) -> pd.DataFrame:
    """Label the 7-day pre-event window on historical failures.

    Cohorts, from the hackathon README and ``solution.csv``:

    * ``failed``: the ESP is absent from ``solution.csv``. The series ends at
      the failure. ``days_to_event == 0`` is the workover day and
      ``1 <= days_to_event <= 7`` is preworkover.
    * ``survivor``: ``Fail in 30 days`` is 0. Every observed day stays normal.
    * ``prospective``: ``Fail in 30 days`` is 1. The failure is still in the
      future, so a 7-day label cannot be read from the observed series.
      Those rows are kept in the audit and excluded from supervised learning.
    """
    keys = _normalize_keys(solution)
    outcome = {
        esp: int(flag)
        for esp, flag in zip(keys["esp_id"], keys[SOLUTION_LABEL], strict=True)
    }
    out = df.copy()
    cohorts = []
    for esp in out["esp_id"]:
        flag = outcome.get(esp)
        if flag is None:
            cohorts.append("failed")
        elif flag == 0:
            cohorts.append("survivor")
        else:
            cohorts.append("prospective")
    out["cohort"] = cohorts
    event_day = (
        out.loc[out["cohort"].eq("failed")].groupby("esp_id")["day_index"].max()
    )
    mapped = out["esp_id"].map(event_day)
    out["days_to_event"] = np.where(out["cohort"].eq("failed"), mapped - out["day_index"], np.nan)

    conditions = np.full(len(out), "normal", dtype=object)
    labels = np.zeros(len(out), dtype=float)
    prospective = out["cohort"].eq("prospective").to_numpy()
    days = out["days_to_event"].to_numpy(dtype=float)
    conditions[prospective] = "excluded_prospective"
    labels[prospective] = np.nan
    pre = (~prospective) & np.isfinite(days) & (days >= 1) & (days <= PRE_EVENT_WINDOW_DAYS)
    work = (~prospective) & np.isfinite(days) & (days == 0)
    conditions[pre] = "preworkover"
    conditions[work] = "workover"
    labels[pre | work] = 1.0
    out["condition"] = conditions
    out["label"] = labels
    return out


def _physical_mask(df: pd.DataFrame) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    for column, (lower, upper, zero_ok) in UNPHYSICAL_BOUNDS.items():
        values = df[column]
        if zero_ok:
            keep &= values.ge(lower) & values.le(upper)
        else:
            keep &= values.gt(lower) & values.le(upper)
    return keep


def prepare_frames(daily: pd.DataFrame, well: pd.DataFrame, solution: pd.DataFrame) -> PreprocessResult:
    """Merge, label, and clean. Return the pre-outlier and post-outlier tables."""
    audit: dict = {"daily_rows_raw": int(len(daily))}
    frame = to_canonical_sensors(_normalize_keys(daily))
    well_norm = _normalize_keys(well)
    context = ["Well_ID", "AL_Key", *[c for c in WELL_CONTEXT if c in well_norm.columns]]
    # esp_id is rebuilt by _normalize_keys; drop the well copy before the join.
    well_part = well_norm[context].drop(columns=["esp_id"], errors="ignore")
    frame = frame.merge(well_part, on=["Well_ID", "AL_Key"], how="left", suffixes=("", "_well"))
    if "ARTIFICIAL_LIFT" in frame.columns:
        before_lift = len(frame)
        frame = frame.loc[frame["ARTIFICIAL_LIFT"].astype(str).str.upper().eq("ESP")].copy()
        audit["rows_non_esp_removed"] = int(before_lift - len(frame))
    frame = assign_day_index(frame)
    frame = attach_failure_labels(frame, solution)
    audit["esp_installations"] = int(frame["esp_id"].nunique())
    audit["cohort_esps"] = {
        key: int(value)
        for key, value in frame.groupby("esp_id")["cohort"].first().value_counts().items()
    }
    before = frame.copy()

    complete = frame[SENSOR_COLUMNS].notna().all(axis=1)
    audit["rows_incomplete_sensors_removed"] = int((~complete).sum())
    frame = frame.loc[complete].copy()

    positive_oil = frame["OIL"].notna() & frame["OIL"].gt(0)
    audit["rows_zero_or_missing_oil_removed"] = int((~positive_oil).sum())
    frame = frame.loc[positive_oil].copy()

    physical = _physical_mask(frame)
    audit["rows_unphysical_removed"] = int((~physical).sum())
    frame = frame.loc[physical].copy()

    keep = pd.Series(True, index=frame.index)
    fences: dict = {}
    for column in SENSOR_COLUMNS:
        lower, upper = iqr_bounds(frame[column].to_numpy(dtype=float))
        fences[column] = {"lower": lower, "upper": upper, "k": IQR_K}
        outside = ~frame[column].between(lower, upper, inclusive="both")
        audit[f"iqr_outside_{column}"] = int(outside.sum())
        keep &= ~outside
    audit["rows_iqr_removed"] = int((~keep).sum())
    cleaned = frame.loc[keep].copy()
    audit["rows_cleaned"] = int(len(cleaned))
    audit["cleaned_esps"] = int(cleaned["esp_id"].nunique())
    audit["class_counts_cleaned"] = {
        key: int(value) for key, value in cleaned["condition"].value_counts().items()
    }
    logger.info(
        "Cleaned corpus: %s rows, %s ESPs. Zero-oil removals: %s. IQR removals: %s.",
        audit["rows_cleaned"],
        audit["cleaned_esps"],
        audit["rows_zero_or_missing_oil_removed"],
        audit["rows_iqr_removed"],
    )
    return PreprocessResult(before_outliers=before, cleaned=cleaned, audit=audit, fences=fences)


def exploration_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Table 2: sensor summaries after outlier removal, before standardization."""
    summary = cleaned[SENSOR_COLUMNS].describe().T
    table = summary.loc[:, ["mean", "std", "min", "25%", "50%", "75%", "max"]].T
    table.index.name = "statistic"
    table.columns = [DISPLAY_NAMES[column] for column in table.columns]
    return table


def class_distribution(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Table 3: normal, preworkover, workover, and excluded prospective rows."""
    order = ["normal", "preworkover", "workover", "excluded_prospective"]
    counts = cleaned["condition"].value_counts()
    rows = [{"condition": name, "reported_data_points": int(counts.get(name, 0))} for name in order]
    return pd.DataFrame(rows)


def _boxplot(df: pd.DataFrame, columns: list[str], path, title: str, ylabel: str) -> None:
    labels = []
    series = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        series.append(values)
        base = column[5:] if column.startswith("diff_") else column
        text = DISPLAY_NAMES.get(base, base)
        labels.append(f"d {text}" if column.startswith("diff_") else text)
    fig_width = max(10.0, 0.72 * len(labels) + 2.0)
    fig, axis = plt.subplots(figsize=(fig_width, 5.8))
    axis.boxplot(
        series,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2.0, "markeredgecolor": "#4C78A8"},
        medianprops={"color": "#E45756", "linewidth": 1.4},
        boxprops={"color": "#333333"},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
    )
    axis.set_xticks(range(1, len(labels) + 1), labels, rotation=55, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fig2_before_outliers(before: pd.DataFrame, path) -> None:
    _boxplot(
        before,
        SENSOR_COLUMNS,
        path,
        "Figure 2. Sensor box plots before outlier removal",
        "Measured value",
    )


def plot_fig3_after_outliers(cleaned: pd.DataFrame, path) -> None:
    _boxplot(
        cleaned,
        SENSOR_COLUMNS,
        path,
        "Figure 3. Sensor box plots after zero-production and IQR removal",
        "Measured value",
    )
