"""KDD phases 2 and 3: standardization, moving differences, PCA, loadings.

The paper's principal-component section uses four steps. The covariance
matrix is equation (3). Scores are the projection Y = X W, written in the
paper as Y = W' X when samples are stored as columns. Component loadings are
the correlations between the original variables and the scores. For
standardized inputs those correlations equal V diag(sqrt(eigenvalues)),
which is the cross-covariance relationship in equation (5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.preprocessing import PRE_EVENT_WINDOW_DAYS, SENSOR_COLUMNS

logger = logging.getLogger(__name__)

N_COMPONENTS = 8
DIFF_COLUMNS = [f"diff_{column}" for column in SENSOR_COLUMNS]
FEATURE_COLUMNS = SENSOR_COLUMNS + DIFF_COLUMNS


@dataclass
class FeatureResult:
    frame: pd.DataFrame
    mean: np.ndarray
    std: np.ndarray
    pca: PCA
    loadings: pd.DataFrame
    train_esp_ids: set[str]
    test_esp_ids: set[str]


def covariance_matrix(X) -> np.ndarray:
    """Sample covariance, equation (3), with divisor n - 1."""
    array = np.asarray(X, dtype=float)
    centered = array - array.mean(axis=0, keepdims=True)
    n_samples = array.shape[0]
    if n_samples < 2:
        raise ValueError("Covariance requires at least two rows.")
    return (centered.T @ centered) / (n_samples - 1)


def fit_standardizer(X) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation.

    The paper asks for a standard scaler that produces zero mean and unit
    variance. That is (x - mean) / sigma. sigma is the square root of the
    population variance, matching ``sklearn.preprocessing.StandardScaler``.
    """
    array = np.asarray(X, dtype=float)
    mean = array.mean(axis=0)
    std = array.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std) -> np.ndarray:
    return (np.asarray(X, dtype=float) - mean) / std


def correlation_loadings(components, explained_variance) -> np.ndarray:
    """Correlation loadings, equation (5).

    Parameters
    ----------
    components:
        Eigenvectors as rows, shape (n_components, n_features), which is
        ``PCA.components_``.
    explained_variance:
        Matching eigenvalues.

    Returns
    -------
    ndarray
        Shape (n_features, n_components). Entry (j, k) is the correlation
        between variable j and component k when the inputs are standardized.
    """
    scales = np.sqrt(np.asarray(explained_variance, dtype=float))
    return np.asarray(components, dtype=float).T * scales


def project_principal_components(X_centered, components) -> np.ndarray:
    """Project samples onto the principal axes.

    With samples stored as rows this is Y = X W. The paper writes
    Y = W' · X for the variables-by-samples layout. ``components`` uses the
    sklearn row-per-component layout, so W is ``components.T``.
    """
    return np.asarray(X_centered, dtype=float) @ np.asarray(components, dtype=float).T


def add_moving_differences(
    df: pd.DataFrame,
    columns: list[str],
    group_col: str = "esp_id",
    order_col: str = "day_index",
) -> pd.DataFrame:
    """One-step difference inside each ESP.

    A difference is kept only when the daily index increases by exactly 1.
    Differencing across a gap left by the IQR filter would mix time scales,
    so those rows are dropped. The first day of every installation is dropped
    with them.
    """
    out = df.sort_values([group_col, order_col], kind="mergesort").copy()
    grouped = out.groupby(group_col, sort=False)
    day_delta = grouped[order_col].diff()
    diff_names = [f"diff_{column}" for column in columns]
    for column, name in zip(columns, diff_names, strict=True):
        out[name] = grouped[column].diff().where(day_delta.eq(1))
    ready = out[diff_names].notna().all(axis=1)
    return out.loc[ready].reset_index(drop=True)


def build_feature_matrix(
    cleaned: pd.DataFrame,
    test_esp_ids: set[str],
    n_components: int = N_COMPONENTS,
) -> FeatureResult:
    """Standardize on the training ESPs, difference, and fit PCA on train rows.

    Prospective rows (missing labels) never enter the scaler or the PCA.
    """
    labeled = cleaned.loc[cleaned["label"].notna()].copy()
    if labeled.empty:
        raise RuntimeError("No supervised rows remain after cleaning.")
    unknown = test_esp_ids - set(labeled["esp_id"])
    if unknown:
        raise KeyError(f"Holdout ids are not in the cleaned corpus: {sorted(unknown)[:5]}")
    labeled["split"] = np.where(labeled["esp_id"].isin(test_esp_ids), "test", "train")
    train_levels = labeled.loc[labeled["split"].eq("train"), SENSOR_COLUMNS]
    if train_levels.empty:
        raise RuntimeError("The training split has no rows.")
    mean, std = fit_standardizer(train_levels.to_numpy(dtype=float))
    scaled = apply_standardizer(labeled[SENSOR_COLUMNS].to_numpy(dtype=float), mean, std)
    for index, column in enumerate(SENSOR_COLUMNS):
        labeled[f"raw_{column}"] = labeled[column].to_numpy()
        labeled[column] = scaled[:, index]

    modeled = add_moving_differences(labeled, SENSOR_COLUMNS)
    train_mask = modeled["split"].eq("train").to_numpy()
    test_mask = modeled["split"].eq("test").to_numpy()
    if train_mask.sum() <= n_components or test_mask.sum() == 0:
        raise RuntimeError(
            "Feature matrix is too small after differencing: "
            f"train={int(train_mask.sum())}, test={int(test_mask.sum())}."
        )
    x_train = modeled.loc[train_mask, FEATURE_COLUMNS].to_numpy(dtype=float)
    pca = PCA(n_components=x_train.shape[1], svd_solver="full", random_state=42)
    pca.fit(x_train)
    scores = project_principal_components(
        modeled[FEATURE_COLUMNS].to_numpy(dtype=float) - pca.mean_,
        pca.components_,
    )
    for component in range(n_components):
        modeled[f"PC{component + 1}"] = scores[:, component]
    loading_matrix = correlation_loadings(
        pca.components_[:n_components],
        pca.explained_variance_[:n_components],
    )
    loadings = pd.DataFrame(
        loading_matrix,
        index=FEATURE_COLUMNS,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    train_ids = set(modeled.loc[train_mask, "esp_id"])
    test_ids = set(modeled.loc[test_mask, "esp_id"])
    logger.info(
        "Features: %s train rows, %s test rows, %s components. "
        "Eight-component cumulative variance = %.3f. Window = %s days.",
        int(train_mask.sum()),
        int(test_mask.sum()),
        n_components,
        float(pca.explained_variance_ratio_[:n_components].sum()),
        PRE_EVENT_WINDOW_DAYS,
    )
    return FeatureResult(
        frame=modeled,
        mean=mean,
        std=std,
        pca=pca,
        loadings=loadings,
        train_esp_ids=train_ids,
        test_esp_ids=test_ids,
    )


def plot_fig4_normalized_difference(frame: pd.DataFrame, path) -> None:
    """Figure 4 uses the modeling matrix: standardized levels and differences."""
    # Reuse the box-plot helper by calling it through a thin local wrapper.
    from src.preprocessing import _boxplot

    _boxplot(
        frame,
        FEATURE_COLUMNS,
        path,
        "Figure 4. Standardized levels and one-day moving differences",
        "Standardized value",
    )


def plot_fig5_pca_geometry(path) -> None:
    """Schematic of the geometric PCA construction in the paper's Figure 5."""
    rng = np.random.default_rng(42)
    # Positive definite covariance so the schematic cloud is a real ellipse.
    covariance = np.array([[1.0, 0.65], [0.65, 0.70]])
    cloud = rng.multivariate_normal([0.0, 0.0], covariance, size=400)
    centered = cloud - cloud.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2]
    fig, axis = plt.subplots(figsize=(6.4, 5.6))
    axis.scatter(cloud[:, 0], cloud[:, 1], s=12, alpha=0.45, c="#4C78A8", linewidths=0)
    origin = cloud.mean(axis=0)
    colors = ["#E45756", "#54A24B"]
    names = ["PC1", "PC2"]
    for direction, color, name in zip(axes, colors, names, strict=True):
        axis.annotate(
            "",
            xy=origin + direction * 2.4,
            xytext=origin,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
        )
        axis.text(*(origin + direction * 2.55), name, color=color, fontsize=12)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("Variable 1")
    axis.set_ylabel("Variable 2")
    axis.set_title("Figure 5. Geometric meaning of the principal axes")
    axis.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fig6_pca_projection(frame: pd.DataFrame, path) -> None:
    """Figure 6: scores on PC1 and PC2, colored by the operating condition."""
    colors = {
        "normal": "#4C78A8",
        "preworkover": "#F58518",
        "workover": "#E45756",
    }
    fig, axis = plt.subplots(figsize=(7.6, 6.2))
    for condition, color in colors.items():
        part = frame.loc[frame["condition"].eq(condition)]
        if part.empty:
            continue
        axis.scatter(
            part["PC1"],
            part["PC2"],
            s=16 if condition == "normal" else 28,
            alpha=0.22 if condition == "normal" else 0.85,
            c=color,
            label=f"{condition} (n={len(part)})",
            linewidths=0,
        )
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title("Figure 6. ESP days projected onto the first two principal components")
    axis.legend(frameon=False, markerscale=1.6)
    axis.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_fig7_explained_variance(pca: PCA, path, n_model_components: int = N_COMPONENTS) -> None:
    """Figure 7: per-component and cumulative explained variance."""
    ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(ratio)
    index = np.arange(1, len(ratio) + 1)
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.bar(index, ratio, color="#4C78A8", label="Explained variance")
    axis.plot(index, cumulative, color="#E45756", marker="o", label="Cumulative")
    axis.axhline(0.90, color="#333333", linestyle="--", linewidth=1, label="90% variance")
    axis.axvline(n_model_components, color="#F58518", linestyle=":", linewidth=1.4)
    axis.set_xlabel("Principal component")
    axis.set_ylabel("Fraction of variance")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(index)
    axis.set_title("Figure 7. Explained variance of the sensor and difference features")
    axis.legend(frameon=False)
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_kdd_workflow(path) -> None:
    """Figure 1: the five KDD phases used in the paper."""
    steps = [
        "1  Data collection\nHackathon ESP telemetry",
        "2  Preprocessing\nLabels, zero oil, IQR fences",
        "3  Feature extraction\nStandardize, difference, PCA",
        "4  Classifier\nOptuna and 10-fold XGBoost",
        "5  Evaluation\nROC-AUC, PR curve, F1",
    ]
    fig, axis = plt.subplots(figsize=(12.5, 3.4))
    axis.set_xlim(0, len(steps))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, text in enumerate(steps):
        box = plt.Rectangle(
            (index + 0.08, 0.28),
            0.78,
            0.46,
            facecolor="#F4F7FB",
            edgecolor="#1F4E79",
            linewidth=1.4,
        )
        axis.add_patch(box)
        axis.text(index + 0.47, 0.51, text, ha="center", va="center", fontsize=9)
        if index < len(steps) - 1:
            axis.annotate(
                "",
                xy=(index + 1.08, 0.51),
                xytext=(index + 0.88, 0.51),
                arrowprops={"arrowstyle": "->", "color": "#1F4E79", "lw": 1.4},
            )
    axis.set_title("Figure 1. Knowledge discovery in databases workflow", pad=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


