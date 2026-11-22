"""Interwell connectivity through geometric considerations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.linalg as sl
from numpy import ndarray

from pywaterflood import _core

idx = pd.IndexSlice


def calc_gains_homogeneous(locations: pd.DataFrame, x_e: float, y_e: float) -> pd.DataFrame:
    r"""Calculate gains from injectors to producers using multiwell productivity index.

    The equation for the influence of injection on production is

    .. math::
        \mathbf{\Lambda} = \frac{\mathbf{A}_p^{-1}}{\sum \mathbf{A}_p^{-1}}
            \times \left(\mathbf{1} \times \mathbf{A}_p^{-1} \times \mathbf{A}_c^T - 1\right)
            - \left( \mathbf{A}_p^{-1} \times \mathbf{A}_c^T \right)

    Args
    ----------
    locations : pd.DataFrame
        columns include:
        `X` : x-location for well
        `Y` : y-location for well
        `Type`: ("Producer" or "Injector")
    x_e : float
        scaling in x-direction (unit size)
    y_e : float
        scaling in y-direction (unit size)

    Returns
    -------
    Lambda : pd.DataFrame
        contains the production-injection multiwell productivity index

    Notes
    -----
    This assumes a roughly rectangular unit with major axes at x and y. You might want
    to rotate your locations.

    Lambda is negative. Not sure why. Negate it to get positive injection leading to
    positive production. :func:`geometry_crm_gains` applies that sign flip and returns
    a ``(n_producers, n_injectors)`` array aligned with requested well order.

    References
    ----------
    Kaviani, D. and Valkó, P.P., 2010. Inferring interwell connectivity using \
    multiwell productivity index (MPI). Journal of Petroleum Science and Engineering, \
    73(1-2), p.48-58.
    """
    locations = locations.copy()
    locations[["X", "Y"]] /= x_e
    y_D = y_e / x_e
    A_prod = calc_influence_matrix(locations, y_D, "prod")
    A_conn = calc_influence_matrix(locations, y_D, "conn")
    A_prod_inv = sl.inv(A_prod.to_numpy())
    term1 = A_prod_inv / np.sum(A_prod_inv)
    term2 = np.ones_like(A_prod_inv) @ A_prod_inv @ A_conn.to_numpy() - 1
    term3 = A_prod_inv @ A_conn.to_numpy()
    Lambda = term1 @ term2 - term3
    connectivity_df = pd.DataFrame(Lambda, index=A_prod.index, columns=A_conn.columns)
    return connectivity_df.rename_axis(index="Producers", columns="Injectors")


def geometry_crm_gains(
    locations: pd.DataFrame,
    x_e: float,
    y_e: float,
    producer_ids=None,
    injector_ids=None,
    n_prod: int | None = None,
    n_inj: int | None = None,
) -> ndarray:
    """Positive CRM gains from homogeneous multiwell productivity index.

    Computes :math:`\\Lambda` with :func:`calc_gains_homogeneous` and returns
    :math:`-\\Lambda` as a C-contiguous ``float64`` array of shape
    ``(n_producers, n_injectors)``. Well order follows ``producer_ids`` and
    ``injector_ids`` when given, otherwise the order producers and injectors
    appear in ``locations``.

    This mapping does not apply CRM sum-to-one or up-to-one normalization;
    constraint mapping belongs to the CRM parameter head so gains are not
    normalized twice.

    Args
    ----------
    locations : pd.DataFrame
        Well map with columns ``X``, ``Y``, ``Type`` (``"Producer"`` or
        ``"Injector"``). Index labels are the well IDs used for alignment.
    x_e, y_e : float
        Reservoir extents used by :func:`calc_gains_homogeneous`.
    producer_ids, injector_ids : sequence, optional
        Well IDs in the same order as CRM production and injection columns.
    n_prod, n_inj : int, optional
        Expected well counts. When given they must match the ID lists.

    Returns
    -------
    gains : ndarray
        Shape ``(n_producers, n_injectors)``, non-negative. Non-finite MPI
        entries are replaced with 0.
    """
    if locations is None:
        msg = (
            "locations is required for geometry-based CRM gains. Pass a DataFrame "
            "with columns X, Y, Type ('Producer' or 'Injector')."
        )
        raise ValueError(msg)
    if not isinstance(locations, pd.DataFrame):
        msg = "locations must be a pandas DataFrame with columns X, Y, Type"
        raise TypeError(msg)
    missing_cols = {"X", "Y", "Type"} - set(locations.columns)
    if missing_cols:
        msg = f"locations is missing required columns: {sorted(missing_cols)}"
        raise ValueError(msg)
    if x_e is None or y_e is None:
        msg = "x_e and y_e are required reservoir extents for geometry-based CRM gains"
        raise ValueError(msg)
    x_e_f = float(x_e)
    y_e_f = float(y_e)
    if not np.isfinite(x_e_f) or not np.isfinite(y_e_f) or x_e_f <= 0.0 or y_e_f <= 0.0:
        msg = "x_e and y_e must be finite and positive"
        raise ValueError(msg)

    type_values = {str(v) for v in locations["Type"].unique()}
    unknown = type_values - {"Producer", "Injector"}
    if unknown:
        msg = (
            "locations['Type'] must be 'Producer' or 'Injector'; "
            f"got extra values {sorted(unknown)}"
        )
        raise ValueError(msg)

    producer_order = _well_ids_for_type(locations, "Producer")
    injector_order = _well_ids_for_type(locations, "Injector")
    producer_ids = producer_order if producer_ids is None else list(producer_ids)
    injector_ids = injector_order if injector_ids is None else list(injector_ids)

    if len(producer_ids) != len(set(producer_ids)):
        msg = "producer_ids contains duplicate well labels"
        raise ValueError(msg)
    if len(injector_ids) != len(set(injector_ids)):
        msg = "injector_ids contains duplicate well labels"
        raise ValueError(msg)
    if n_prod is not None and len(producer_ids) != int(n_prod):
        msg = (
            f"producer_ids length {len(producer_ids)} does not match n_producers={n_prod} "
            "(production column count). Pass producer_ids in production-column order."
        )
        raise ValueError(msg)
    if n_inj is not None and len(injector_ids) != int(n_inj):
        msg = (
            f"injector_ids length {len(injector_ids)} does not match n_injectors={n_inj} "
            "(injection column count). Pass injector_ids in injection-column order."
        )
        raise ValueError(msg)

    lambda_df = calc_gains_homogeneous(locations, x_e_f, y_e_f)
    missing_prod = [wid for wid in producer_ids if wid not in lambda_df.index]
    missing_inj = [wid for wid in injector_ids if wid not in lambda_df.columns]
    if missing_prod or missing_inj:
        msg = (
            "Geometry well IDs do not match the MPI influence matrix. "
            f"Missing producers: {missing_prod}; missing injectors: {missing_inj}. "
            "Use the same labels as locations.index, in historical-data column order."
        )
        raise ValueError(msg)

    # MPI Lambda is negative; CRM gains f_ij must be non-negative.
    ordered = (-lambda_df).reindex(index=producer_ids, columns=injector_ids)
    return _sanitize_positive_gains(ordered.to_numpy(dtype=np.float64, copy=True))


def _well_ids_for_type(locations: pd.DataFrame, well_type: str) -> list:
    """Well IDs of one type, in the order they appear in ``locations``."""
    mask = locations["Type"] == well_type
    if not bool(mask.any()):
        msg = f"locations has no wells with Type={well_type!r}"
        raise ValueError(msg)
    return locations.index[mask].tolist()


def _sanitize_positive_gains(gains: ndarray) -> ndarray:
    """Replace non-finite entries, clip negatives, keep ``(n_prod, n_inj)``."""
    values = np.ascontiguousarray(np.asarray(gains, dtype=np.float64))
    if values.ndim != 2:
        msg = f"geometry gains must be 2-d (n_producers, n_injectors), got shape {values.shape}"
        raise ValueError(msg)
    values = np.where(np.isfinite(values), values, 0.0)
    np.clip(values, 0.0, None, out=values)
    return values


def translate_locations(
    locations: pd.DataFrame, x_col: str, y_col: str, type_col: str
) -> pd.DataFrame:
    """Translate locations  to prepare for building connectivity matrix.

    Moves the lower left edge of the reservoir to (0, 0), and sets up the matrix columns
    to work with `calc_gains_homogeneous`

    Args
    ----------
    locations : pd.DataFrame
        Has the x,y locations and type assignment for all wells
    x_col : str
        Column in `locations` for the x-location
    y_col : str
        Column in `locations` for the y-location
    type_col : str
        Column in `locations` for the type (producer or injector)

    Returns
    -------
    locations_out: pd.DataFrame
        columns are `X` (x-location), `Y` (y-location), `Type` (producer or injector)
    """
    locations_out = pd.DataFrame(index=locations.index, columns=["X", "Y", "Type"])
    locations_out["X"] = locations[x_col] - locations[x_col].min()
    locations_out["Y"] = locations[y_col] - locations[y_col].min()
    locations_out["Type"] = locations[type_col]
    return locations_out


def calc_influence_matrix(
    locations: pd.DataFrame, y_D: float, matrix_type: str = "conn", m_max: int = 100
) -> pd.DataFrame:
    """Calculate influence matrix A.

    Args
    ----------
    locations : pd.DataFrame
        Has the x,y locations and type assignment for all wells
        columns are `X` (x-location), `Y` (y-location), `Type` (producer or injector)
    y_D : float
        dimensionless scaling for y-direction
    matrix_type : str, choice of `conn` or `prod`
        injector-producer matrix or producer-producer matrix
    m_max : int > 0
        number of terms in the series to calculate. 100 is a good default.

    Returns
    -------
    influence_matrix : pd.DataFrame
        a matrix with the influences between wells
    """
    if matrix_type not in ["conn", "prod"]:
        msg = "matrix_type must be either `conn` or `prod`"
        raise ValueError(msg)
    XA = locations[locations.Type == "Producer"]
    XB = XA.copy() if matrix_type == "prod" else locations[locations.Type == "Injector"]
    influence_matrix = pd.DataFrame(
        index=pd.MultiIndex.from_product([XA.index, XB.index]), columns=["A"]
    )
    m = np.arange(1, m_max + 1, dtype="uint64")  # elements of sum
    for i, j in influence_matrix.index:
        x_i, y_i = XA.loc[i, ["X", "Y"]]
        x_j, y_j = XB.loc[j, ["X", "Y"]] + 1e-6
        influence_matrix.loc[idx[i, j], "A"] = calc_A_ij(x_i, y_i, x_j, y_j, y_D, m)
    return pd.pivot_table(influence_matrix["A"].reset_index(), "A", "level_0", "level_1").astype(
        "float64"
    )


def calc_A_ij(x_i: float, y_i: float, x_j: float, y_j: float, y_D: float, m: ndarray) -> float:
    r"""Calculate element in the influence matrix.

    .. math::
        A_{ij} = 2 \pi y_D (\frac13 - \frac{y_i}{y_D} +
            \frac{y_i^2 + y_j^2}{2 y_D^2})
            + \sum_{m=1}^\infty \frac{t_m}m \cos(m\pi \tilde x_i)
            \cos(m \pi \tilde x_j)
    where

    .. math::
        t_m = \frac{\cosh\left(m\pi (y_D - |\tilde y_i - \tilde y_j|)\right)
        + \cosh\left(m\pi (y_D - \tilde y_i - \tilde y_j\right)}
        {\sinh\left(m\pi y_D \right)}

    Args
    ----
    x_i : float
        x-location of i'th well
    y_i : float
        y-location of i'th well
    x_j : float
        x-location of j'th well
    y_j : float
        y-location of j'th well
    y_D : float
        dimensionless parameter for y-direction
    m : ndarray
        series terms, from 1 to m_max

    Returns
    -------
    A_ij : float

    References
    ----------
    Kaviani, D. and Valkó, P.P., 2010. Inferring interwell connectivity using \
    multiwell productivity index (MPI). Journal of Petroleum Science and Engineering, \
    73(1-2), p.48-58. https://doi.org/10.1016/j.petrol.2010.05.006
    """
    # Symmetry properties, see https://doi.org/10.1016/j.petrol.2010.05.006, A5-A6
    y_eD = y_D
    x_D = max([x_i, x_j])
    y_D = max([y_i, y_j])
    x_wD = min([x_i, x_j])
    y_wD = min([y_i, y_j])
    if not ((x_D - x_wD) > (y_D - y_wD)):
        y_eD = 1.0 / y_eD
    return _core.calc_A_ij(x_D, y_D, x_wD, y_wD, y_eD, m)
