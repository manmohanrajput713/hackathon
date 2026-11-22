"""Geometry-based CRM connectivity initialization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from pywaterflood.crm import CRM, CrmCompensated, _PhysicalParameterHead
from pywaterflood.multiwellproductivity import (
    calc_gains_homogeneous,
    geometry_crm_gains,
    translate_locations,
)


def _locations_inverted_five_spot():
    """Four producers around one injector, labeled independently of DataFrame order."""
    locations = pd.DataFrame(
        {
            "X": [1.0, 20.0, 1.0, 20.0, 10.5],
            "Y": [1.0, 1.0, 20.0, 20.0, 10.5],
            "Type": ["Producer", "Producer", "Producer", "Producer", "Injector"],
        },
        index=["P0", "P1", "P2", "P3", "I0"],
    )
    return locations, 34.0, 22.0


def _locations_many_to_many():
    """Two producers and two injectors with IDs shuffled relative to map order."""
    locations = pd.DataFrame(
        {
            "X": [3.0, 19.0, 4.0, 16.0],
            "Y": [4.0, 3.0, 18.0, 17.0],
            "Type": ["Injector", "Producer", "Producer", "Injector"],
        },
        index=["I1", "P1", "P0", "I0"],
    )
    return locations, 24.0, 24.0


def _synthetic_rates(n_time: int, n_prod: int, n_inj: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    time = np.arange(n_time, dtype=np.float64)
    injection = 20.0 + rng.random((n_time, n_inj))
    production = 10.0 + rng.random((n_time, n_prod))
    return production, injection, time


def _head_gains(constraints: str, gains: np.ndarray, tau: np.ndarray) -> np.ndarray:
    n_prod, n_inj = gains.shape
    head = _PhysicalParameterHead(
        n_inj=n_inj,
        n_prod=n_prod,
        tau_selection="per-producer",
        constraints=constraints,
        primary=True,
        with_pressure=False,
    )
    head.initialize_from_numpy(gains=gains, tau=tau)
    with torch.no_grad():
        return head()["gains"].detach().cpu().numpy()


class TestGeometryCrmGains:
    def test_shape_matches_well_counts(self):
        locations, x_e, y_e = _locations_inverted_five_spot()
        gains = geometry_crm_gains(locations, x_e, y_e)
        assert gains.shape == (4, 1)
        assert gains.dtype == np.float64
        assert gains.flags["C_CONTIGUOUS"]

    def test_explicit_ids_reorder_without_transpose(self):
        locations, x_e, y_e = _locations_many_to_many()
        producer_ids = ["P0", "P1"]
        injector_ids = ["I0", "I1"]
        gains = geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=producer_ids,
            injector_ids=injector_ids,
        )
        lambda_df = calc_gains_homogeneous(locations, x_e, y_e)
        expected = -lambda_df.loc[producer_ids, injector_ids].to_numpy()
        np.testing.assert_allclose(gains, expected)
        assert gains.shape == (2, 2)
        shuffled = geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=["P1", "P0"],
            injector_ids=["I1", "I0"],
        )
        np.testing.assert_allclose(shuffled, expected[::-1, ::-1])
        np.testing.assert_allclose(gains[0, 1], -lambda_df.loc["P0", "I1"])
        np.testing.assert_allclose(gains[1, 0], -lambda_df.loc["P1", "I0"])

    @pytest.mark.parametrize(
        ("n_prod", "n_inj", "producer_xy", "injector_xy"),
        [
            (1, 1, [(5.0, 5.0)], [(12.0, 12.0)]),
            (1, 3, [(8.0, 8.0)], [(2.0, 2.0), (14.0, 2.0), (8.0, 14.0)]),
            (
                3,
                2,
                [(2.0, 2.0), (14.0, 2.0), (8.0, 14.0)],
                [(8.0, 2.0), (8.0, 8.0)],
            ),
        ],
        ids=["one-to-one", "one-to-many", "many-to-many"],
    )
    def test_well_configurations(self, n_prod, n_inj, producer_xy, injector_xy):
        rows = []
        index = []
        for i, (x, y) in enumerate(producer_xy):
            rows.append({"X": x, "Y": y, "Type": "Producer"})
            index.append(f"P{i}")
        for j, (x, y) in enumerate(injector_xy):
            rows.append({"X": x, "Y": y, "Type": "Injector"})
            index.append(f"I{j}")
        locations = pd.DataFrame(rows, index=index)
        gains = geometry_crm_gains(
            locations,
            20.0,
            20.0,
            producer_ids=[f"P{i}" for i in range(n_prod)],
            injector_ids=[f"I{j}" for j in range(n_inj)],
            n_prod=n_prod,
            n_inj=n_inj,
        )
        assert gains.shape == (n_prod, n_inj)
        assert np.all(np.isfinite(gains))
        assert np.all(gains >= 0)

    def test_missing_and_mismatched_geometry(self):
        locations, x_e, y_e = _locations_inverted_five_spot()
        with pytest.raises(ValueError, match="locations is required"):
            geometry_crm_gains(None, x_e, y_e)
        with pytest.raises(TypeError, match="pandas DataFrame"):
            geometry_crm_gains([1, 2, 3], x_e, y_e)
        with pytest.raises(ValueError, match="missing required columns"):
            geometry_crm_gains(pd.DataFrame({"X": [1.0], "Y": [1.0]}), x_e, y_e)
        with pytest.raises(ValueError, match="x_e and y_e"):
            geometry_crm_gains(locations, None, y_e)
        with pytest.raises(ValueError, match="finite and positive"):
            geometry_crm_gains(locations, 0.0, y_e)
        with pytest.raises(ValueError, match="n_producers"):
            geometry_crm_gains(locations, x_e, y_e, n_prod=3, n_inj=1)
        with pytest.raises(ValueError, match="Missing producers"):
            geometry_crm_gains(
                locations,
                x_e,
                y_e,
                producer_ids=["P0", "PX", "P2", "P3"],
                injector_ids=["I0"],
            )

    def test_zero_and_nonfinite_influence(self, monkeypatch):
        locations, x_e, y_e = _locations_many_to_many()
        producer_ids = ["P0", "P1"]
        injector_ids = ["I0", "I1"]
        dirty = pd.DataFrame(
            [[np.nan, 0.0], [-np.inf, 0.4]],
            index=producer_ids,
            columns=injector_ids,
        )

        def _fake_lambda(_locs, _xe, _ye):
            return dirty

        monkeypatch.setattr(
            "pywaterflood.multiwellproductivity.calc_gains_homogeneous",
            _fake_lambda,
        )
        gains = geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=producer_ids,
            injector_ids=injector_ids,
        )
        assert gains.shape == (2, 2)
        assert np.all(np.isfinite(gains))
        assert np.all(gains >= 0)
        assert gains[0, 0] == 0.0
        assert gains[1, 0] == 0.0


class TestCrmGeometryInitialization:
    def test_missing_geometry_raises_without_fallback(self):
        production, injection, time = _synthetic_rates(12, 4, 1)
        crm = CRM(primary=True, tau_selection="per-producer", constraints="positive")
        with pytest.raises(ValueError, match="Geometry-based initialization requires locations"):
            crm.fit(
                production,
                injection,
                time,
                initial_guess="geometry",
                options={"maxiter": 0},
            )
        crm.set_rates(production, injection, time)
        with pytest.raises(ValueError, match="requires x_e"):
            crm.fit(
                production,
                injection,
                time,
                initial_guess="geometry",
                locations=_locations_inverted_five_spot()[0],
                options={"maxiter": 0},
            )

    def test_invalid_initial_guess_string(self):
        production, injection, time = _synthetic_rates(8, 2, 2)
        crm = CRM()
        with pytest.raises(ValueError, match="initial_guess string must be one of"):
            crm.fit(production, injection, time, initial_guess="mpi", options={"maxiter": 0})

    @pytest.mark.parametrize("constraints", ["positive", "up-to one", "sum-to-one"])
    def test_shape_orientation_and_constraints(self, constraints):
        locations, x_e, y_e = _locations_inverted_five_spot()
        production, injection, time = _synthetic_rates(16, 4, 1)
        geo = geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=["P0", "P1", "P2", "P3"],
            injector_ids=["I0"],
        )
        crm = CRM(primary=True, tau_selection="per-producer", constraints=constraints)
        crm.set_rates(production, injection, time)
        x0 = crm._coerce_initial_guess(
            "geometry",
            random=False,
            geometry={
                "locations": locations,
                "x_e": x_e,
                "y_e": y_e,
                "producer_ids": ["P0", "P1", "P2", "P3"],
                "injector_ids": ["I0"],
            },
        )
        unpacked = crm._guess_matrices(x0)
        assert unpacked["gains"].shape == (4, 1)
        np.testing.assert_allclose(unpacked["gains"], geo)

        crm.fit(
            production,
            injection,
            time,
            initial_guess="geometry",
            locations=locations,
            x_e=x_e,
            y_e=y_e,
            producer_ids=["P0", "P1", "P2", "P3"],
            injector_ids=["I0"],
            options={"maxiter": 0},
        )
        assert crm.gains.shape == (4, 1)
        assert np.all(crm.gains >= 0)
        mapped = _head_gains(constraints, geo, unpacked["tau"])
        np.testing.assert_allclose(crm.gains, mapped, atol=1e-10)
        if constraints == "sum-to-one":
            np.testing.assert_allclose(crm.gains.sum(axis=1), 1.0, atol=1e-10)
            row = np.clip(geo, 1e-8, None)
            expected = row / row.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(crm.gains, expected, atol=1e-10)
        elif constraints == "up-to one":
            assert np.all(crm.gains.sum(axis=0) <= 1.0 + 1e-10)
        else:
            np.testing.assert_allclose(crm.gains, geo, atol=1e-8, rtol=1e-6)

    def test_well_ordering_follows_historical_columns(self):
        locations, x_e, y_e = _locations_many_to_many()
        production, injection, time = _synthetic_rates(10, 2, 2)
        producer_ids = ["P0", "P1"]
        injector_ids = ["I0", "I1"]
        crm = CRM(primary=False, tau_selection="per-pair", constraints="positive")
        crm.fit(
            production,
            injection,
            time,
            initial_guess="geometry",
            locations=locations,
            x_e=x_e,
            y_e=y_e,
            producer_ids=producer_ids,
            injector_ids=injector_ids,
            options={"maxiter": 0},
        )
        expected = geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=producer_ids,
            injector_ids=injector_ids,
        )
        np.testing.assert_allclose(crm.gains, expected, atol=1e-8, rtol=1e-6)
        reversed_crm = CRM(primary=False, tau_selection="per-pair", constraints="positive")
        reversed_crm.fit(
            production,
            injection,
            time,
            initial_guess="geometry",
            locations=locations,
            x_e=x_e,
            y_e=y_e,
            producer_ids=["P1", "P0"],
            injector_ids=["I1", "I0"],
            options={"maxiter": 0},
        )
        np.testing.assert_allclose(reversed_crm.gains, expected[::-1, ::-1], atol=1e-8, rtol=1e-6)

    def test_no_double_normalization_positive_constraint(self):
        locations, x_e, y_e = _locations_many_to_many()
        production, injection, time = _synthetic_rates(10, 2, 2)
        geo = geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=["P0", "P1"],
            injector_ids=["I0", "I1"],
        )
        row_normed = geo / geo.sum(axis=1, keepdims=True)
        crm = CRM(primary=True, tau_selection="per-producer", constraints="positive")
        crm.fit(
            production,
            injection,
            time,
            initial_guess="geometry",
            locations=locations,
            x_e=x_e,
            y_e=y_e,
            producer_ids=["P0", "P1"],
            injector_ids=["I0", "I1"],
            options={"maxiter": 0},
        )
        np.testing.assert_allclose(crm.gains, geo, atol=1e-8, rtol=1e-6)
        assert not np.allclose(crm.gains, row_normed)

    @pytest.mark.parametrize("random", [False, True])
    def test_uniform_and_random_regression(self, random):
        production, injection, time = _synthetic_rates(12, 3, 2)
        n_prod, n_inj = 3, 2
        crm = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
        crm.set_rates(production, injection, time)
        x0 = crm._get_initial_guess(random=random)
        gains = crm._guess_matrices(x0)["gains"]
        assert gains.shape == (n_prod, n_inj)
        np.testing.assert_allclose(gains.sum(axis=0), 1.0, atol=1e-12)
        if not random:
            np.testing.assert_allclose(gains, np.full((n_prod, n_inj), 1.0 / n_prod))

        mode = "random" if random else "uniform"
        crm.fit(
            production,
            injection,
            time,
            initial_guess=mode,
            options={"maxiter": 0},
        )
        assert crm.gains.shape == (n_prod, n_inj)
        assert np.all(crm.gains >= 0)

        crm_flag = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
        crm_flag.fit(
            production,
            injection,
            time,
            random=random,
            options={"maxiter": 0},
        )
        assert crm_flag.gains.shape == (n_prod, n_inj)

    def test_pinn_handoff_and_trainability(self):
        locations, x_e, y_e = _locations_inverted_five_spot()
        production, injection, time = _synthetic_rates(20, 4, 1)
        geo_kwargs = {
            "locations": locations,
            "x_e": x_e,
            "y_e": y_e,
            "producer_ids": ["P0", "P1", "P2", "P3"],
            "injector_ids": ["I0"],
        }
        crm = CRM(primary=True, tau_selection="per-producer", constraints="positive")
        crm.fit(
            production,
            injection,
            time,
            initial_guess="geometry",
            options={"maxiter": 0},
            seed=0,
            **geo_kwargs,
        )
        seeded = crm.gains.copy()
        geo = geometry_crm_gains(**geo_kwargs)
        np.testing.assert_allclose(seeded, geo, atol=1e-8, rtol=1e-6)

        head = crm.pinn_model_.parameter_head
        assert head.raw_gains.requires_grad
        assert head.raw_gains.shape == (4, 1)
        params = list(head.parameters())
        assert all(param.requires_grad for param in params)

        residual = head.constraint_penalty(head())
        residual.backward()
        assert head.raw_gains.grad is not None
        assert torch.any(head.raw_gains.grad != 0) or residual.item() == 0.0
        head.zero_grad()

        crm.fit(
            production,
            injection,
            time,
            initial_guess="geometry",
            options={"maxiter": 8},
            lr=0.05,
            seed=1,
            **geo_kwargs,
        )
        assert crm.gains.shape == (4, 1)
        assert crm.pinn_model_.parameter_head.raw_gains.requires_grad
        assert len(crm.loss_history_) == 8

    def test_compensated_geometry_init(self):
        locations, x_e, y_e = _locations_inverted_five_spot()
        production, injection, time = _synthetic_rates(12, 4, 1)
        pressure = 1000.0 - production
        crm = CrmCompensated(primary=True, tau_selection="per-producer", constraints="positive")
        crm.fit(
            production,
            pressure,
            injection,
            time,
            initial_guess="geometry",
            locations=locations,
            x_e=x_e,
            y_e=y_e,
            producer_ids=["P0", "P1", "P2", "P3"],
            injector_ids=["I0"],
            options={"maxiter": 0},
        )
        expected = geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=["P0", "P1", "P2", "P3"],
            injector_ids=["I0"],
        )
        assert crm.gains.shape == (4, 1)
        np.testing.assert_allclose(crm.gains, expected, atol=1e-8, rtol=1e-6)
        assert crm.pinn_model_.parameter_head.raw_gains.requires_grad

    def test_translated_locations_still_align(self):
        raw = pd.DataFrame(
            {
                "easting": [101.0, 120.0, 101.0, 120.0, 110.5],
                "northing": [51.0, 51.0, 70.0, 70.0, 60.5],
                "kind": ["Producer", "Producer", "Producer", "Producer", "Injector"],
            },
            index=["P0", "P1", "P2", "P3", "I0"],
        )
        locations = translate_locations(raw, "easting", "northing", "kind")
        production, injection, time = _synthetic_rates(8, 4, 1)
        crm = CRM(primary=True, tau_selection="per-producer", constraints="positive")
        crm.fit(
            production,
            injection,
            time,
            initial_guess="geometry",
            locations=locations,
            x_e=34.0,
            y_e=22.0,
            producer_ids=["P0", "P1", "P2", "P3"],
            injector_ids=["I0"],
            options={"maxiter": 0},
        )
        assert crm.gains.shape == (4, 1)
        assert np.all(crm.gains >= 0)
