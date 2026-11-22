"""Deterministic, synthetic telemetry and confirmed event records for the smoke run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_fixture(directory: Path) -> None:
    rng = np.random.default_rng(2498)
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    intervals = [0, 58, 115, 151, 185]
    telemetry = []
    maintenance = []

    def timestamp(day):
        return start + pd.Timedelta(days=day)

    for number in range(20):
        well = f"SYN-{number + 1:02d}"
        samples_per_day = 48 if number == 0 else 4
        steps = np.arange(185 * samples_per_day + 1)
        days = steps / samples_per_day
        casing = rng.normal(0, 12, len(days)) + rng.uniform(410, 510) + 8 * np.sin(days / 3)
        tubing = rng.normal(0, 7, len(days)) + rng.uniform(145, 205) + 5 * np.cos(days / 5)
        casing += rng.normal(0, 10, len(days)).cumsum() / np.sqrt(len(days))
        gas = rng.normal(55_000, 6_200, len(days))
        choke = rng.normal(44, 7, len(days))
        heading = (days >= 20) & (days < 24)
        if number == 0:
            casing[heading] += 42 * np.sin(days[heading] * 2 * np.pi * 6)
            maintenance.append({"well_id": well, "event_type": "casing_heading", "timestamp": timestamp(20)})
        maintenance.append({"well_id": well, "event_type": "installation", "timestamp": timestamp(0)})
        for interval, upper in enumerate(intervals[1:]):
            if (number + 2 * interval) % 5 >= 3:
                fault = upper - (5 + (number % 5))
                proxy = ((number + interval) % 5 == 0)
                onset = upper if proxy else fault
                severity = rng.uniform(16, 32)
                ramp = np.clip((days - (onset - 8)) / 6, 0, 1) * (days < upper)
                casing += severity * ramp
                tubing -= severity * 0.5 * ramp
                gas -= severity * 120 * ramp
                maintenance.append({
                    "well_id": well,
                    "event_type": "valve_leak" if number % 2 else "valve_malfunction",
                    "event_id": f"{well}-{interval}",
                    "confirmed": True,
                    "detected_at": None if proxy else timestamp(fault),
                    "failed_at": None if proxy else timestamp(fault + 0.25),
                    "changeout_at": timestamp(upper),
                })
            else:
                # Pressure disturbances with no qualifying valve failure.
                maintenance.append({"well_id": well, "event_type": "planned_replacement", "timestamp": timestamp(upper)})
                disturbance = upper - 9 - (number % 3)
                pulse = np.maximum(0, 1 - np.abs(days - disturbance) / 1.5)
                casing += 26 * pulse
                tubing -= 12 * pulse
                if number % 2:
                    maintenance.append({"well_id": well, "event_type": "tubing_leak", "timestamp": timestamp(disturbance)})
        maintenance.append({"well_id": well, "event_type": "coverage_end", "timestamp": timestamp(185)})
        mask = ~((days > 72) & (days < 74)) if number % 4 == 0 else np.ones(len(days), dtype=bool)
        if number % 4 == 0:
            gas[:] = np.nan
        if number % 3 == 0:
            choke[::21] = np.nan
        if number % 4 == 1:
            tubing[::127] = np.nan
        telemetry.append(pd.DataFrame({
            "well_id": well,
            "timestamp": start + pd.to_timedelta(steps[mask] * (1440 // samples_per_day), unit="min"),
            "casing_pressure": casing[mask],
            "tubing_head_pressure": tubing[mask],
            "lift_gas_rate": gas[mask],
            "choke_position": choke[mask],
            "operating_state": np.where(days[mask] % 37 < 1, "transition", "continuous_gas_lift"),
        }))
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(telemetry).to_csv(directory / "telemetry.csv", index=False, float_format="%.4f")
    pd.DataFrame(maintenance).to_csv(directory / "maintenance.csv", index=False)
