"""Causal pressure features and an independent casing-heading screen."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import KEYS, SERVICE, timestamp_ns


BASELINE = ["casing_kpa", "tubing_kpa", "casing_minus_tubing_kpa",
            "casing_change_kpa", "tubing_change_kpa"]
ROLLING = ["casing_mean_kpa", "casing_std_kpa", "tubing_mean_kpa", "tubing_std_kpa"]
OPTIONAL_NUMERIC = ["lift_gas_sm3d", "choke_pct", "nodal_residual_kpa"]
OPTIONAL_CATEGORICAL = ["operating_state"]
FEATURE_UNITS = {
    **dict.fromkeys([*BASELINE, *ROLLING, "nodal_residual_kpa"], "kPa"),
    "casing_kpa": "kPa absolute",
    "tubing_kpa": "kPa absolute",
    "casing_mean_kpa": "kPa absolute",
    "tubing_mean_kpa": "kPa absolute",
    "lift_gas_sm3d": "Sm3/day",
    "choke_pct": "percent",
    "operating_state": "category",
}


def build_features(frame: pd.DataFrame, max_gap: pd.Timedelta, rolling_window: pd.Timedelta) -> pd.DataFrame:
    """Use the previous observation and trailing, time-based statistics within a service run."""
    output = frame.sort_values(KEYS).reset_index(drop=True).copy()
    grouped = output.groupby(SERVICE, sort=False)
    elapsed = grouped["timestamp"].diff()
    gap = elapsed.gt(max_gap)
    output["casing_minus_tubing_kpa"] = output["casing_kpa"] - output["tubing_kpa"]
    for kind in ("casing", "tubing"):
        output[f"{kind}_change_kpa"] = grouped[f"{kind}_kpa"].diff().where(elapsed.le(max_gap))
    complete = output[["casing_kpa", "tubing_kpa"]].notna().all(axis=1)
    # Missing baseline observations also break rolling history.
    run = (elapsed.isna() | gap | ~complete | ~complete.shift(fill_value=False)).cumsum()
    rolling = (
        output.assign(run=run).groupby("run", sort=False)
        .rolling(rolling_window, on="timestamp", min_periods=3)[["casing_kpa", "tubing_kpa"]]
    )
    for summary in ("mean", "std"):
        values = getattr(rolling, summary)().droplevel(0)
        for kind in ("casing", "tubing"):
            output[f"{kind}_{summary}_kpa"] = values[f"{kind}_kpa"].to_numpy()
    output["feature_status"] = np.select(
        [
            output["known_event_or_post_event"],
            ~complete,
            gap,
            elapsed.isna(),
            output[BASELINE].isna().any(axis=1),
        ],
        ["known_event_or_post_event", "missing_pressure", "gap", "first_observation", "missing_previous_pressure"],
        default="ready",
    )
    return output


def select_features(train: pd.DataFrame, extended: bool) -> list[str]:
    optional = [*ROLLING, *OPTIONAL_NUMERIC, *OPTIONAL_CATEGORICAL]
    available = [name for name in optional if name in train and train[name].notna().any()]
    return [*BASELINE, *available] if extended else BASELINE.copy()


def model_inputs(frame: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    inputs = frame.reindex(columns=schema).copy()
    for name in schema:
        if name in OPTIONAL_CATEGORICAL:
            text = inputs[name].astype(object)
            inputs[name] = text.where(text.notna(), "__missing__")
        else:
            inputs[name] = pd.to_numeric(inputs[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return inputs


def casing_heading(frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Screen for two completed pressure cycles at regular enough cadence."""
    min_period = pd.Timedelta(hours=settings["min_period_hours"]).value
    max_period = pd.Timedelta(hours=settings["max_period_hours"]).value
    window = pd.Timedelta(hours=settings["window_hours"]).value
    max_step = min(min_period // 4, pd.Timedelta(minutes=settings["max_step_minutes"]).value)
    amplitude = settings["min_amplitude_kpa"]
    output = frame[KEYS].copy()
    output["casing_heading_status"] = "unavailable"
    for _, rows in frame.groupby(SERVICE, sort=False):
        times = timestamp_ns(rows["timestamp"])
        values = rows["casing_kpa"].to_numpy()
        valid = np.isfinite(values)
        runs = np.split(
            np.arange(len(rows)),
            np.flatnonzero((np.diff(times) > max_step) | ~valid[1:] | ~valid[:-1]) + 1,
        )
        for indices in runs:
            if len(indices) < 3:
                continue
            t = times[indices]
            p = values[indices]
            ready = (t - t[0]) >= 2 * max_period
            signal = np.zeros(len(t), dtype=bool)
            peaks = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
            for left, center, right in zip(peaks, peaks[1:], peaks[2:]):
                previous_period = t[center] - t[left]
                next_period = t[right] - t[center]
                if not (min_period <= previous_period <= max_period and min_period <= next_period <= max_period):
                    continue
                if abs(previous_period - next_period) > max(previous_period, next_period) / 3:
                    continue
                if min(p[left], p[center]) - p[left:center].min() < amplitude:
                    continue
                if min(p[center], p[right]) - p[center:right].min() < amplitude:
                    continue
                # A local peak is known only after the next pressure observation.
                stop = np.searchsorted(t, t[left] + window, side="right")
                signal[right + 1:stop] = True
            status = np.where(ready, "no_cycles_detected", "unavailable")
            status[signal] = "possible_heading"
            output.loc[rows.index[indices], "casing_heading_status"] = status
    return output
