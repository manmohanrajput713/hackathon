"""Unit tests for the KDD formulas, labels, and leakage boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download_data import schema_check
from src.evaluate import (
    f1_from_precision_recall,
    false_positive_rate,
    precision_from_counts,
    recall_from_counts,
)
from src.features import (
    FEATURE_COLUMNS,
    add_moving_differences,
    build_feature_matrix,
    correlation_loadings,
    covariance_matrix,
    fit_standardizer,
    apply_standardizer,
    project_principal_components,
)
from src.model import (
    assign_esp_folds,
    holdout_esp_ids,
    optimal_leaf_weight,
    similarity_score,
    split_gain,
)
from src.preprocessing import (
    SENSOR_COLUMNS,
    class_distribution,
    exploration_table,
    iqr_bounds,
    plot_fig2_before_outliers,
    prepare_frames,
)


def test_iqr_fences_match_quartiles():
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    lower, upper = iqr_bounds(values, k=1.5)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    assert lower == pytest.approx(q1 - 1.5 * iqr)
    assert upper == pytest.approx(q3 + 1.5 * iqr)
    assert lower == pytest.approx(-2.5)
    assert upper == pytest.approx(11.5)


def test_covariance_matches_numpy():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(40, 5))
    np.testing.assert_allclose(
        covariance_matrix(matrix),
        np.cov(matrix, rowvar=False),
        rtol=1e-10,
    )


def test_standardizer_has_unit_variance():
    rng = np.random.default_rng(8)
    matrix = rng.normal(loc=5.0, scale=2.0, size=(500, 3))
    mean, std = fit_standardizer(matrix)
    scaled = apply_standardizer(matrix, mean, std)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(scaled.std(axis=0, ddof=0), 1.0, atol=1e-10)


def test_loadings_and_projection_match_the_pca_formulas():
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(80, 6))
    # Sample standard deviation makes each column's n-1 variance equal to 1,
    # which is the divisor inside the PCA eigenvalues used by the loadings.
    scaled = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)
    pca = PCA(n_components=3, svd_solver="full").fit(scaled)
    loadings = correlation_loadings(pca.components_, pca.explained_variance_)
    expected = pca.components_.T * np.sqrt(pca.explained_variance_)
    np.testing.assert_allclose(loadings, expected)
    scores = project_principal_components(scaled - pca.mean_, pca.components_)
    np.testing.assert_allclose(scores, pca.transform(scaled))
    pearson = np.corrcoef(scaled[:, 0], scores[:, 0])[0, 1]
    assert loadings[0, 0] == pytest.approx(pearson, rel=1e-6, abs=1e-6)


def test_moving_difference_keeps_only_consecutive_days():
    frame = pd.DataFrame(
        {
            "esp_id": ["A", "A", "A", "A"],
            "day_index": [0, 1, 2, 4],
            "FRQ": [1.0, 2.0, 4.0, 10.0],
        }
    )
    result = add_moving_differences(frame, ["FRQ"])
    assert result["day_index"].tolist() == [1, 2]
    assert result["diff_FRQ"].tolist() == [1.0, 2.0]


def test_hourly_median_resample():
    stamps = pd.date_range("2024-01-01", periods=4, freq="30min")
    frame = pd.DataFrame({"time": stamps, "FRQ": [1.0, 3.0, 10.0, 14.0]})
    # Import through preprocessing to keep the public function available
    # under the features re-export used above. The implementation lives
    # with the cleaning stage; call it directly.
    from src.preprocessing import resample_moving_median as hourly

    out = hourly(frame, "time", ["FRQ"])
    assert out["FRQ"].iloc[0] == pytest.approx(2.0)
    assert out["FRQ"].iloc[1] == pytest.approx(12.0)


def test_zero_oil_unphysical_and_pre_event_labels():
    daily, well, solution = _labeled_archive()
    prepared = prepare_frames(daily, well, solution)
    cleaned = prepared.cleaned
    failed = cleaned.loc[cleaned["esp_id"].eq("1|ESP_1")]
    # Day 0 is zero oil and day 1 is an unphysical frequency, so both are gone.
    assert set(failed["day_index"]) == set(range(2, 12))
    assert (failed["OIL"] > 0).all()
    assert failed["FRQ"].max() < 100
    workover = failed.loc[failed["condition"].eq("workover"), "day_index"].tolist()
    preworkover = failed.loc[failed["condition"].eq("preworkover"), "day_index"].tolist()
    normal = failed.loc[failed["condition"].eq("normal"), "day_index"].tolist()
    assert workover == [11]
    assert preworkover == [4, 5, 6, 7, 8, 9, 10]
    assert normal == [2, 3]
    assert set(failed.loc[failed["label"].eq(1), "day_index"]) == set(range(4, 12))
    survivor = cleaned.loc[cleaned["esp_id"].eq("2|ESP_1")]
    assert (survivor["condition"] == "normal").all()
    assert (survivor["label"] == 0).all()
    prospective = cleaned.loc[cleaned["esp_id"].eq("3|ESP_1")]
    assert (prospective["condition"] == "excluded_prospective").all()
    assert prospective["label"].isna().all()
    counts = class_distribution(cleaned).set_index("condition")["reported_data_points"]
    assert counts["workover"] == 1
    assert counts["preworkover"] == 7
    table = exploration_table(cleaned)
    assert list(table.index) == ["mean", "std", "min", "25%", "50%", "75%", "max"]


def test_group_holdout_is_disjoint_and_stratified():
    rows = []
    for index in range(10):
        rows.append({"esp_id": f"f{index}", "cohort": "failed", "label": 1.0})
    for index in range(10):
        rows.append({"esp_id": f"s{index}", "cohort": "survivor", "label": 0.0})
    frame = pd.DataFrame(rows)
    test_ids = holdout_esp_ids(frame, test_size=0.3, seed=42)
    train_ids = set(frame["esp_id"]) - test_ids
    assert test_ids.isdisjoint(train_ids)
    assert 2 <= len(test_ids) <= 8
    cohorts = frame.drop_duplicates("esp_id").set_index("esp_id")["cohort"]
    assert set(cohorts.loc[list(test_ids)]) == {"failed", "survivor"}
    folds = assign_esp_folds(
        sorted(train_ids),
        [cohorts.loc[esp] for esp in sorted(train_ids)],
        n_splits=5,
        seed=42,
    )
    assert set(folds) == train_ids
    assert set(folds.values()) <= set(range(5))


def test_pca_fit_uses_training_esps_only():
    frame = _modeling_frame()
    baseline = build_feature_matrix(frame, {"B"}, n_components=4)
    corrupted = frame.copy()
    corrupted.loc[corrupted["esp_id"].eq("B"), SENSOR_COLUMNS] = 10_000.0
    other = build_feature_matrix(corrupted, {"B"}, n_components=4)
    pd.testing.assert_frame_equal(baseline.loadings, other.loadings)
    train_a = baseline.frame.loc[baseline.frame["esp_id"].eq("A"), "FRQ"].to_numpy()
    train_b = other.frame.loc[other.frame["esp_id"].eq("A"), "FRQ"].to_numpy()
    np.testing.assert_allclose(train_a, train_b)
    assert list(baseline.loadings.columns) == ["PC1", "PC2", "PC3", "PC4"]
    assert list(baseline.loadings.index) == FEATURE_COLUMNS


def test_classification_counts_and_tree_gain():
    assert precision_from_counts(8, 2) == pytest.approx(0.8)
    assert recall_from_counts(8, 2) == pytest.approx(0.8)
    assert false_positive_rate(2, 8) == pytest.approx(0.2)
    assert f1_from_precision_recall(0.8, 0.6) == pytest.approx(2 * 0.8 * 0.6 / 1.4)
    gradient = np.array([0.1, -0.2, 0.3])
    hessian = np.array([0.2, 0.2, 0.2])
    reg_lambda = 1.0
    assert similarity_score(gradient, hessian, reg_lambda) == pytest.approx((0.2**2) / 1.6)
    assert optimal_leaf_weight(gradient, hessian, reg_lambda) == pytest.approx(-0.2 / 1.6)
    gain = split_gain(gradient, hessian, [True, False, True], reg_lambda, gamma=0.1)
    left = (0.4**2) / (0.4 + 1.0)
    right = ((-0.2) ** 2) / (0.2 + 1.0)
    parent = (0.2**2) / 1.6
    assert gain == pytest.approx(0.5 * (left + right - parent) - 0.1)


def test_boxplot_writes_a_png(tmp_path):
    frame = _modeling_frame()
    target = tmp_path / "fig2.png"
    plot_fig2_before_outliers(frame, target)
    assert target.is_file()
    assert target.stat().st_size > 1000


def test_checked_in_hackathon_schema():
    daily_path = ROOT / "dataset" / "dailyData.csv"
    if not daily_path.exists():
        pytest.skip("hackathon extract is not in dataset/")
    daily = pd.read_csv(daily_path)
    well = pd.read_csv(ROOT / "dataset" / "wellData.csv")
    solution = pd.read_csv(ROOT / "dataset" / "solution.csv")
    report = schema_check(daily, well, solution)
    assert report["daily_rows"] > 1000
    assert report["solution_rows"] == 40


def _sensor_block(n_days: int, center: float = 50.0) -> dict[str, np.ndarray]:
    return {column: np.full(n_days, center, dtype=float) for column in SENSOR_COLUMNS}


def _labeled_archive():
    frames = []
    for well_id, n_days in ((1, 12), (2, 8), (3, 6)):
        block = _sensor_block(n_days)
        block.update(
            {
                "Well_ID": np.full(n_days, well_id),
                "AL_Key": np.array(["ESP_1"] * n_days),
                "OIL": np.full(n_days, 100.0),
            }
        )
        frames.append(pd.DataFrame(block))
    daily = pd.concat(frames, ignore_index=True)
    # Historical failure: a shut-in day and one impossible frequency.
    daily.loc[0, "OIL"] = 0.0
    daily.loc[1, "FRQ"] = 500.0
    well = pd.DataFrame({"Well_ID": [1, 2, 3], "AL_Key": ["ESP_1", "ESP_1", "ESP_1"]})
    solution = pd.DataFrame(
        {
            "Well_ID": [2, 3],
            "AL_Key": ["ESP_1", "ESP_1"],
            "Fail in 30 days": [0, 1],
        }
    )
    return daily, well, solution


def _modeling_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    frames = []
    for esp, well_id, failed in (("A", 1, True), ("B", 2, False)):
        n_days = 30
        day = np.arange(n_days)
        data = {
            "Well_ID": np.full(n_days, well_id),
            "AL_Key": np.array(["ESP_1"] * n_days),
            "esp_id": np.array([esp] * n_days),
            "day_index": day,
            "OIL": np.full(n_days, 80.0),
            "cohort": np.array(["failed" if failed else "survivor"] * n_days),
        }
        for index, column in enumerate(SENSOR_COLUMNS):
            data[column] = 40.0 + index * 5.0 + day * 0.2 + rng.normal(0, 0.05, n_days)
        if failed:
            days_to_event = (n_days - 1) - day
            condition = np.where(
                days_to_event == 0,
                "workover",
                np.where(days_to_event <= 7, "preworkover", "normal"),
            )
            label = (days_to_event <= 7).astype(float)
            data["days_to_event"] = days_to_event.astype(float)
        else:
            condition = np.array(["normal"] * n_days)
            label = np.zeros(n_days)
            data["days_to_event"] = np.full(n_days, np.nan)
        data["condition"] = condition
        data["label"] = label
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)
