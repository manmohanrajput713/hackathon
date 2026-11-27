"""Input schema, pressure normalization, valve intervals, and right-censored labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PRESSURE_FACTORS = {"kpa": 1.0, "pa": 0.001, "mpa": 1_000.0, "bar": 100.0, "psi": 6.894757293168}
GAS_FACTORS = {"sm3/d": 1.0, "scf/d": 0.028316846592, "mscf/d": 28.316846592,
               "mmscf/d": 28_316.846592}
QUALIFYING = {"valve_leak", "valve_malfunction"}
RESET_TYPES = {"installation", "replacement", "planned_replacement"}
OTHER_TYPES = {"tubing_leak", "casing_heading", "other", "coverage_end"}
KEYS = ["well_id", "timestamp"]
SERVICE = ["well_id", "service_start"]
TIME_COLUMNS = ["timestamp", "detected_at", "failed_at", "changeout_at", "recorded_at"]
TELEMETRY_FIELDS = {*KEYS, "casing_pressure", "tubing_head_pressure", "lift_gas_rate",
                    "choke_position", "operating_state", "nodal_residual"}
MAINTENANCE_FIELDS = {"well_id", "event_type", "event_id", "confirmed", *TIME_COLUMNS}
DEFAULT_CONFIG = {
    "telemetry_columns": {},
    "maintenance_columns": {},
    "event_types": {},
    "casing_pressure": {"unit": "kpa", "reference": "gauge", "ambient_kpa": 101.325},
    "tubing_head_pressure": {"unit": "kpa", "reference": "gauge", "ambient_kpa": 101.325},
    "lift_gas_rate_unit": "sm3/d",
    "choke_position_unit": "percent",
    "nodal_residual_unit": "kpa",
    "nodal_model_id": None,
    "assume_log_complete_through_telemetry": False,
}
NO_RESET = np.iinfo(np.int64).min


def load_config(path: Path | None) -> dict:
    overrides = json.loads(path.read_text()) if path else {}
    unknown = overrides.keys() - DEFAULT_CONFIG.keys()
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    config = DEFAULT_CONFIG | overrides
    for key in ("casing_pressure", "tubing_head_pressure"):
        spec = DEFAULT_CONFIG[key] | config[key]
        if spec.keys() != DEFAULT_CONFIG[key].keys():
            raise ValueError(f"Unknown pressure settings for {key}")
        if spec["unit"] not in PRESSURE_FACTORS or spec["reference"] not in {"absolute", "gauge"}:
            raise ValueError(f"Invalid pressure unit or reference for {key}")
        if not np.isfinite(spec["ambient_kpa"]) or spec["ambient_kpa"] <= 0:
            raise ValueError(f"{key}.ambient_kpa must be positive and finite")
        config[key] = spec
    for key, valid in (("telemetry_columns", TELEMETRY_FIELDS), ("maintenance_columns", MAINTENANCE_FIELDS)):
        if config[key].keys() - valid or len(set(config[key].values())) != len(config[key]):
            raise ValueError(f"{key} must map distinct canonical fields to distinct input columns")
    config["event_types"] = {key.strip().lower(): value for key, value in config["event_types"].items()}
    if set(config["event_types"].values()) - (QUALIFYING | RESET_TYPES | OTHER_TYPES):
        raise ValueError("event_types values must use the canonical event types")
    if config["lift_gas_rate_unit"] not in GAS_FACTORS:
        raise ValueError("Invalid standard lift-gas rate unit")
    if config["choke_position_unit"] not in {"percent", "fraction"}:
        raise ValueError("Choke position unit must be percent or fraction")
    if config["nodal_residual_unit"] not in PRESSURE_FACTORS:
        raise ValueError("Nodal residual requires a pressure-difference unit")
    if not isinstance(config["assume_log_complete_through_telemetry"], bool):
        raise ValueError("assume_log_complete_through_telemetry must be boolean")
    return config


def _read(path: Path, mapping: dict, required: set, accepted: set) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    sources = {name: mapping.get(name, name) for name in accepted}
    present = {name: source for name, source in sources.items() if source in raw}
    missing = required - present.keys()
    if missing:
        raise ValueError(f"{path.name} missing input columns: {sorted(sources[name] for name in missing)}")
    if len(set(present.values())) != len(present):
        raise ValueError(f"{path.name} maps a source column to multiple canonical fields")
    frame = raw[list(present.values())].rename(columns={v: k for k, v in present.items()})
    for key in frame:
        frame[key] = frame[key].str.strip().replace("", pd.NA)
    if frame["well_id"].isna().any():
        raise ValueError(f"{path.name} contains empty well IDs")
    return frame


def _parse_time(values: pd.Series, name: str, required: bool = False) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, format="mixed", errors="coerce").astype("datetime64[ns, UTC]")
    invalid = parsed.isna() & (values.notna() | required)
    if invalid.any():
        raise ValueError(f"{name} contains {int(invalid.sum())} invalid or missing timestamps")
    return parsed


def timestamp_ns(values: pd.Series) -> np.ndarray:
    return values.dt.as_unit("ns").astype("int64").to_numpy()


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def read_telemetry(path: Path, config: dict, *, scoring: bool = False) -> tuple[pd.DataFrame, dict]:
    required = set(KEYS) if scoring else {*KEYS, "casing_pressure", "tubing_head_pressure"}
    raw = _read(path, config["telemetry_columns"], required, TELEMETRY_FIELDS)
    if raw.empty:
        raise ValueError("Telemetry has no rows")
    present = sorted(raw.columns)
    output = raw[KEYS].copy()
    output["timestamp"] = _parse_time(raw["timestamp"], "telemetry.timestamp", required=True)
    if output.duplicated(KEYS).any():
        raise ValueError("Duplicate (well_id, timestamp) telemetry rows. Resolve duplicates in the source.")
    for name, output_name in (("casing_pressure", "casing_kpa"), ("tubing_head_pressure", "tubing_kpa")):
        spec = config[name]
        values = _numeric(raw[name]) if name in raw else pd.Series(np.nan, index=raw.index)
        values = values * PRESSURE_FACTORS[spec["unit"]]
        if spec["reference"] == "gauge":
            values += spec["ambient_kpa"]
        output[output_name] = values.where(values.gt(0) & np.isfinite(values))
    if "lift_gas_rate" in raw:
        values = _numeric(raw["lift_gas_rate"]) * GAS_FACTORS[config["lift_gas_rate_unit"]]
        output["lift_gas_sm3d"] = values.where(values.ge(0) & np.isfinite(values))
    if "choke_position" in raw:
        values = _numeric(raw["choke_position"])
        if config["choke_position_unit"] == "fraction":
            values = values * 100
        output["choke_pct"] = values.where(values.between(0, 100))
    if "operating_state" in raw:
        output["operating_state"] = raw["operating_state"].str.lower()
    if "nodal_residual" in raw and raw["nodal_residual"].notna().any():
        if not config["nodal_model_id"]:
            raise ValueError("Set nodal_model_id to use a residual from a frozen upstream model")
        values = _numeric(raw["nodal_residual"]) * PRESSURE_FACTORS[config["nodal_residual_unit"]]
        output["nodal_residual_kpa"] = values.where(np.isfinite(values))
    audit = {
        "telemetry_rows": len(output),
        "telemetry_wells": int(output["well_id"].nunique()),
        "telemetry_start": output["timestamp"].min().isoformat(),
        "telemetry_end": output["timestamp"].max().isoformat(),
        "telemetry_columns_present": present,
        "missing_or_invalid_by_normalized_column": output.drop(columns=KEYS).isna().sum().to_dict(),
    }
    return output.sort_values(KEYS).reset_index(drop=True), audit


def read_maintenance(path: Path | None, config: dict) -> pd.DataFrame:
    if path:
        raw = _read(path, config["maintenance_columns"], {"well_id", "event_type"}, MAINTENANCE_FIELDS)
    else:
        raw = pd.DataFrame(columns=["well_id", "event_type"])
    for name in MAINTENANCE_FIELDS - set(raw):
        raw[name] = pd.NA
    raw["event_type"] = raw["event_type"].str.lower().map(
        lambda name: config["event_types"].get(name, name), na_action="ignore",
    )
    valid_types = QUALIFYING | RESET_TYPES | OTHER_TYPES
    if (~raw["event_type"].isin(valid_types)).any():
        invalid = raw.loc[~raw["event_type"].isin(valid_types), "event_type"].unique().tolist()
        raise ValueError(f"Map maintenance event types to canonical codes: {invalid}")
    for name in TIME_COLUMNS:
        raw[name] = _parse_time(raw[name], f"maintenance.{name}")
    flags = raw["confirmed"].str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    if (flags.notna() & ~flags.isin(mapping)).any():
        raise ValueError("maintenance.confirmed must be true/false, yes/no, 1/0, or blank")
    raw["confirmed"] = flags.map(mapping).eq(True)  # Blank means unconfirmed.
    reset_rows = raw["event_type"].isin(RESET_TYPES)
    if (reset_rows & raw["timestamp"].isna() & raw["changeout_at"].isna()).any():
        raise ValueError("A service reset requires timestamp or changeout_at")
    if (raw["event_type"].eq("coverage_end") & raw["timestamp"].isna()).any():
        raise ValueError("coverage_end requires timestamp")
    return raw


def _availability(event_time: pd.Series, recorded: pd.Series) -> pd.Series:
    return recorded.where(recorded.gt(event_time), event_time)


def maintenance_timelines(log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Return physical resets, confirmed event candidates, and verified follow-up."""
    reset_mask = log["event_type"].isin(RESET_TYPES)
    changeout_mask = log["event_type"].isin(QUALIFYING) & log["changeout_at"].notna()
    resets = log.loc[reset_mask | changeout_mask, ["well_id", "timestamp", "changeout_at", "recorded_at"]].copy()
    resets["reset_at"] = resets["changeout_at"].fillna(resets["timestamp"])
    resets["available_at"] = _availability(resets["reset_at"], resets["recorded_at"])
    resets["closes_previous"] = log.loc[resets.index, "event_type"].ne("installation")
    resets = (
        resets.sort_values("available_at").drop_duplicates(["well_id", "reset_at", "closes_previous"])
        [["well_id", "reset_at", "available_at", "closes_previous"]]
    )

    qualified = log["event_type"].isin(QUALIFYING)
    candidates = log.loc[qualified & log["confirmed"]].copy()
    reliable_times = candidates[["detected_at", "failed_at"]].copy()
    after_changeout = reliable_times.gt(candidates["changeout_at"], axis=0)
    reliable = reliable_times.mask(after_changeout).min(axis=1)
    candidates["event_at"] = reliable.fillna(candidates["changeout_at"])
    candidates["target_basis"] = np.where(reliable.notna(), "detection_or_failure", "changeout")
    # A confirmation entered later must not disable prospective scoring before it was known.
    candidates["available_at"] = _availability(candidates["event_at"], candidates["recorded_at"])
    undated = candidates.loc[candidates["event_at"].isna()]
    missing_event_time = len(undated)
    # An undated confirmed failure makes negative windows for that well unverifiable.
    candidates = candidates.dropna(subset=["event_at"])
    resolved_ids = pd.MultiIndex.from_frame(candidates.loc[candidates["event_id"].notna(), ["well_id", "event_id"]])
    unresolved = ~pd.MultiIndex.from_frame(undated[["well_id", "event_id"]]).isin(resolved_ids)
    undated_wells = sorted(undated.loc[unresolved, "well_id"].unique().tolist())
    # Keep all reports for as-of scoring. Training will reconcile duplicate event IDs.
    coverage = log.loc[log["event_type"].eq("coverage_end")].groupby("well_id")["timestamp"].max()
    audit = {
        "maintenance_rows": len(log),
        "maintenance_rows_by_type": log["event_type"].value_counts().to_dict(),
        "qualifying_confirmed_rows": int((qualified & log["confirmed"]).sum()),
        "qualifying_unconfirmed_rows": int((qualified & ~log["confirmed"]).sum()),
        "confirmed_rows_without_event_time": missing_event_time,
        "confirmed_rows_with_onset_timestamp_after_changeout": int(after_changeout.any(axis=1).sum()),
        "wells_with_undated_confirmed_events": undated_wells,
        "verified_service_resets": len(resets[["well_id", "reset_at"]].drop_duplicates()),
        "wells_with_explicit_coverage_end": len(coverage),
    }
    return resets, candidates, coverage, audit


def _known_time(timestamps: np.ndarray, updates: pd.DataFrame, time_field: str) -> np.ndarray:
    ordered = updates.sort_values("available_at")
    cutoff = timestamp_ns(ordered["available_at"])
    effective = timestamp_ns(ordered[time_field])
    return np.maximum.accumulate(np.r_[NO_RESET, effective])[np.searchsorted(cutoff, timestamps, side="right")]


def align_service_as_of(
    telemetry: pd.DataFrame, resets: pd.DataFrame, candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Align each sample using only resets and confirmations available by that sample."""
    reset_groups = dict(iter(resets.groupby("well_id", sort=False)))
    event_groups = dict(iter(candidates.groupby("well_id", sort=False)))
    pieces = []
    for well_id, rows in telemetry.groupby("well_id", sort=False):
        rows = rows.copy()
        times = timestamp_ns(rows["timestamp"])
        well_resets = reset_groups.get(well_id, resets.iloc[:0])
        known_reset = _known_time(times, well_resets, "reset_at")
        known_replacement = _known_time(times, well_resets.loc[well_resets["closes_previous"]], "reset_at")
        known_event = _known_time(times, event_groups.get(well_id, candidates.iloc[:0]), "event_at")
        rows["service_start"] = known_reset
        event_on_installation = (known_event == known_reset) & (known_reset > known_replacement)
        rows["known_event_or_post_event"] = (known_event > known_reset) | (known_event == times) | event_on_installation
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def _episode_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    # A later report for one episode may resolve a changeout proxy to a reliable onset.
    identified = candidates.loc[candidates["event_id"].notna()].copy()
    identified["is_changeout_proxy"] = identified["target_basis"].eq("changeout")
    identified = identified.sort_values(["is_changeout_proxy", "event_at", "available_at"], kind="stable")
    identified = identified.drop_duplicates(["well_id", "event_id"]).drop(columns="is_changeout_proxy")
    unidentified = candidates.loc[candidates["event_id"].isna()]
    return pd.concat([identified, unidentified]).sort_values(
        ["event_at", "target_basis"], ascending=[True, False], kind="stable",
    )


def attach_failure_labels(
    frame: pd.DataFrame,
    resets: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.Series,
    horizon: pd.Timedelta,
    max_gap: pd.Timedelta,
    *,
    assume_log_complete: bool = False,
    undated_wells: tuple | list = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Label (t, t + horizon] when an event is known, or a full negative window is observed."""
    episodes = _episode_candidates(candidates)
    reset_groups = dict(iter(resets.groupby("well_id", sort=False)))
    event_groups = dict(iter(episodes.groupby("well_id", sort=False)))
    pieces = []
    episode_pieces = []
    no_coverage = []
    for well_id, rows in frame.groupby("well_id", sort=False):
        rows = rows.copy()
        times = timestamp_ns(rows["timestamp"])
        well_resets = reset_groups.get(well_id, resets.iloc[:0])
        boundaries = np.unique(timestamp_ns(well_resets["reset_at"]))
        service_index = np.searchsorted(boundaries, times, side="right")
        physical = np.r_[NO_RESET, boundaries][service_index]
        service_ends = np.r_[boundaries, np.iinfo(np.int64).max][service_index]
        associated = event_groups.get(well_id, episodes.iloc[:0]).copy()
        event_times = timestamp_ns(associated["event_at"])
        # A fault simultaneous with changeout belongs to the interval that just ended.
        replacement_times = timestamp_ns(well_resets.loc[well_resets["closes_previous"], "reset_at"])
        associated["service_index"] = np.searchsorted(boundaries, event_times, side="right") - np.isin(
            event_times, replacement_times,
        )
        associated["precedes_telemetry"] = associated["event_at"].lt(rows["timestamp"].min())
        associated = associated.drop_duplicates("service_index")
        episode_pieces.append(associated)
        events_for_rows = associated.set_index("service_index").reindex(service_index).set_axis(rows.index)
        event_at = events_for_rows["event_at"]
        labels = np.full(len(rows), np.nan)
        status = np.full(len(rows), "incomplete_followup", dtype=object)
        # A gap ends observable telemetry follow-up. A confirmed event can still establish a positive.
        run = np.cumsum(np.r_[True, np.diff(times) > max_gap.value])
        last_observed = pd.Series(times).groupby(run).transform("max").to_numpy()
        if well_id in coverage.index:
            last_observed = np.minimum(last_observed, coverage.at[well_id].value)
        elif not assume_log_complete:
            last_observed = NO_RESET
            no_coverage.append(well_id)
        if well_id in undated_wells:
            last_observed = NO_RESET
        end = times + horizon.value
        negative = (end <= last_observed) & (end < service_ends)
        event_ns = timestamp_ns(event_at)
        positive = event_at.notna().to_numpy() & (event_ns > times) & (event_ns <= end)
        post = (event_at.notna().to_numpy() & (event_ns <= times)) | np.isin(times, event_times)
        misaligned = rows["service_start"].to_numpy() != physical
        labels[negative] = 0
        status[negative] = "negative"
        labels[positive] = 1
        status[positive] = "positive"
        labels[misaligned | post] = np.nan
        status[misaligned] = "service_reset_not_yet_recorded"
        status[post] = "event_or_post_event"
        rows["event_at"] = event_at
        rows["target_basis"] = events_for_rows["target_basis"]
        rows["label"] = labels
        rows["label_status"] = status
        rows["label_available_at"] = pd.Series(pd.to_datetime(end, utc=True), index=rows.index).where(
            labels == 0, events_for_rows["available_at"].where(labels == 1),
        )
        pieces.append(rows)

    labeled = pd.concat(pieces, ignore_index=True)
    event_inventory = pd.concat(episode_pieces, ignore_index=True)
    known_wells = set(frame["well_id"])
    counts = {
        "wells_without_verified_log_coverage": no_coverage,
        "wells_with_telemetry_before_first_known_installation": int(
            labeled.loc[labeled["service_start"].eq(NO_RESET), "well_id"].nunique()
        ),
        "timed_event_candidates_without_telemetry_well": int((~episodes["well_id"].isin(known_wells)).sum()),
        "later_timed_events_collapsed_within_service_interval": int(
            episodes["well_id"].isin(known_wells).sum() - len(event_inventory)
        ),
        "timed_event_reports_merged_by_event_id": len(candidates) - len(episodes),
        "events_by_target_basis": event_inventory["target_basis"].value_counts().to_dict(),
        "events_before_first_telemetry": int(event_inventory["precedes_telemetry"].sum()),
        "rows_by_label_status": labeled["label_status"].value_counts().to_dict(),
    }
    return labeled, event_inventory, counts
