"""Analyze waterfloods with capacitance-resistance models.

This is the central module in ``pywaterflood``, based around the :code:`CRM`
class. History matching uses a Physics-Informed Neural Network (Deep-CRM /
PINN) after Yewgat et al. (ECMOR 2020). Forecasting with :meth:`CRM.predict`
uses the identified physical parameters (connectivities, time constants, and
productivity indices) in the standard CRM superposition solution, so the
public API is unchanged.

For most cases, the best performance comes from selecting
:code:`CRM(primary=True, tau_selection="per-pair", constraints="up-to one")`.
In the literature, this is referred to as CRM-IP (injector producer).

If the data is too sparse, then change ``tau_selection`` to "per-producer".
This reduces the number of variables to fit by nearly half by using only one
time constant for all well connections influencing a producer. This is referred
to as CRM-P in the literature.

If the data is still too sparse, you can sum all the injectors, all the producers,
or both. This greatly decreases the utility of the model and is not recommended. In
the literature, it is known as CRM-T.

The base class assumes constant bottomhole pressures for the producing wells.
If you know the pressures for these wells or at least the trend, consider using
``CrmCompensated``.

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from pywaterflood import _core
from pywaterflood.multiwellproductivity import geometry_crm_gains

_DTYPE = torch.float64
_TAU_FLOOR = 1e-10
_VALID_CONSTRAINTS = (
    "positive",
    "up-to one",
    "sum-to-one",
    "sum-to-one injector",
)
_SCIPY_FIT_KEYS = {
    "method",
    "jac",
    "hess",
    "hessp",
    "bounds",
    "constraints",
    "tol",
    "callback",
}
_VALID_INITIAL_GUESS_MODES = ("uniform", "random", "geometry")
_GEOMETRY_FIT_KEYS = ("locations", "x_e", "y_e", "producer_ids", "injector_ids")


def q_primary(
    production: NDArray, time: NDArray, gain_producer: float, tau_producer: float
) -> NDArray:
    r"""Calculate primary production contribution.

    Uses Arps equation with :math:`b=0`

    .. math::
        q_{p}(t) = q_i e^{-bt}

    Args
    ----------
    production : NDArray
        Production, size: Number of time steps
    time : NDArray
        Producing times to forecast, size: Number of time steps
    gain_producer : float
        Arps :math:`q_i` factor
    tau_producer : float
        Arps time constant

    Returns
    ----------
    q_hat : NDArray
        Calculated production, :math:`\hat q`, size: Number of time steps
    """
    return _core.q_primary(
        _as_f64_1d(production),
        _as_f64_1d(time),
        float(gain_producer),
        float(tau_producer),
    )


def q_CRM_perpair(injection: NDArray, time: NDArray, gains: NDArray, taus: NDArray) -> NDArray:
    """Calculate per injector-producer pair production.

    Runs for influences of each injector on one producer, assuming
    individual :code:`gain` and :code:`tau` for each pair

    Args
    ----------
    injection : NDArray
        Injected fluid, size: Number of time steps
    time : NDArray
        Producing times to forecast, size: Number of time steps
    gains : NDArray
        Connectivities between each injector and the producer,
        size: Number of injectors
    taus : NDArray
        Time constants between each injector and the producer,
        size: Number of injectors

    Returns
    ----------
    q_hat : NDArray
        Calculated production :math:`\\hat q`, size: Number of time steps
    """
    injection_2d = _as_f64_2d(injection)
    taus_1d = _as_f64_1d(taus)
    if taus_1d.size == 1 and injection_2d.shape[1] > 1:
        taus_1d = np.full(injection_2d.shape[1], float(taus_1d[0]), dtype=np.float64)
    return _core.q_crm_perpair(injection_2d, _as_f64_1d(time), _as_f64_1d(gains), taus_1d)


def q_CRM_perproducer(injection: NDArray, time: NDArray, gain: NDArray, tau: float) -> NDArray:
    """Calculate per injector-producer pair production (simplified tank).

    Uses simplified CRMP model that assumes a single tau for each producer

    Args
    ----------
    injection : NDArray
        injected fluid in reservoir volumes, size: Number of time steps
    time : NDArray
        Producing times to forecast, size: Number of time steps
    gains : NDArray
        Connectivities between each injector and the producer
        size: Number of injectors
    tau : float
        Time constants all injectors and the producer

    Returns
    ----------
    q_hat : NDArray
        Calculated production :math:`\\hat q`

        shape: Number of time steps
    """
    injection_2d = _as_f64_2d(injection)
    tau2 = np.full(
        injection_2d.shape[1],
        float(np.asarray(tau, dtype=np.float64).reshape(-1)[0]),
        dtype=np.float64,
    )
    return q_CRM_perpair(injection_2d, time, gain, tau2)


def q_bhp(pressure_local: NDArray, pressure: NDArray, v_matrix: NDArray) -> NDArray:
    r"""Calculate the production effect from bottom-hole pressure variation.

    This looks like

    .. math::
        q_{BHP,j}(t_i) = \sum_{k} v_{kj}\left[ p_j(t_{i-1}) - p_k(t_i) \right]

    Args
    ----
    pressure_local : NDArray
        pressure for the well in question, shape: n_time
    pressure : NDArray
        bottomhole pressure, shape: n_time, n_producers
    v_matrix : NDArray
        connectivity between one producer and all producers, shape: n_producers

    Returns
    -------
    q : NDArray
        production from changing BHP, shape: n_time
    """
    return _core.q_bhp(
        _as_f64_1d(pressure_local),
        _as_f64_2d(pressure),
        _as_f64_1d(v_matrix),
    )


def random_weights(n_prod: int, n_inj: int, axis: int = 0, seed: int | None = None) -> NDArray:
    """Generate random weights for producer-injector gains.

    Args
    ----
    n_prod : int
        Number of producing wells
    n_inj : int
        Number of injecting wells
    axis : int, default is 0
        0 corresponds to normalizing among producers, 1 to normalizing among injectors
    seed : int, default is None
        state for random number generator

    Returns
    -------
    gains_guess: NDArray
        shape: n_prod, n_inj
    """
    rng = np.random.default_rng(seed)
    limit = 10 * (n_prod if axis == 0 else n_inj)
    vec = rng.integers(0, limit, (n_prod, n_inj))
    axis_sum = vec.sum(axis, keepdims=True)
    return vec / axis_sum


def _as_f64_1d(values) -> NDArray:
    """C-contiguous 1-d float64 vector for the PyO3 ``_core`` bindings."""
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))


def _as_f64_2d(values) -> NDArray:
    """C-contiguous 2-d float64 array for the PyO3 ``_core`` bindings."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if array.ndim == 1:
        array = np.ascontiguousarray(array.reshape(-1, 1))
    if array.ndim != 2:
        msg = "expected a 1-d or 2-d array"
        raise ValueError(msg)
    return array


def _tensor_to_numpy(tensor: torch.Tensor) -> NDArray:
    """Copy a PINN tensor to host float64 for CRM predict / ``_core``."""
    return np.ascontiguousarray(tensor.detach().cpu().to(dtype=_DTYPE).numpy())


def _resolve_torch_device(spec: Any) -> torch.device:
    """Select a device that can run float64 CRM physics.

    CUDA is used when requested and available. MPS is mapped to CPU because
    float64 autograd is not supported there. Unknown or unavailable devices
    fall back to CPU.
    """
    device = spec if isinstance(spec, torch.device) else torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps":
        return torch.device("cpu")
    return device


def _inv_softplus(values: NDArray | float) -> NDArray:
    """Numerically stable inverse of ``softplus`` for parameter initialization."""
    array = np.asarray(values, dtype=np.float64)
    array = np.clip(array, 1e-8, None)
    return np.where(array > 20.0, array, np.log(np.expm1(array)))


def _as_1d_time(time: NDArray) -> NDArray:
    return _as_f64_1d(time)


def _pop_geometry_fit_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pull MPI geometry arguments out of ``fit`` kwargs before PINN options."""
    return {key: kwargs.pop(key, None) for key in _GEOMETRY_FIT_KEYS}


def _initial_guess_mode(initial_guess, random: bool) -> str | None:
    """Map ``fit`` arguments to ``uniform`` / ``random`` / ``geometry``.

    Returns ``None`` when ``initial_guess`` is already a numeric ``x0``.
    """
    if isinstance(initial_guess, str):
        mode = initial_guess.strip().lower().replace("_", "-")
        if mode not in _VALID_INITIAL_GUESS_MODES:
            msg = (
                "initial_guess string must be one of 'uniform', 'random', or "
                f"'geometry', not {initial_guess!r}"
            )
            raise ValueError(msg)
        return mode
    if initial_guess is None or (
        not isinstance(initial_guess, (np.ndarray, torch.Tensor))
        and hasattr(initial_guess, "__len__")
        and len(initial_guess) == 0
    ):
        return "random" if random else "uniform"
    if isinstance(initial_guess, (np.ndarray, torch.Tensor)) and initial_guess.size == 0:
        return "random" if random else "uniform"
    return None


def _interp_along_time(
    t_new: torch.Tensor, t_ref: torch.Tensor, y_ref: torch.Tensor
) -> torch.Tensor:
    """Linear interpolate ``y_ref(t_ref)`` onto ``t_new``.

    Parameters
    ----------
    t_new : tensor
        Query times, shape (T*, 1) or (T*,)
    t_ref : tensor
        Monotone source times, shape (T,)
    y_ref : tensor
        Source samples, shape (T, C)
    """
    t_query = t_new.reshape(-1)
    t_ref = t_ref.reshape(-1)
    n_ref = t_ref.numel()
    if n_ref == 1:
        return y_ref.expand(t_query.numel(), -1)

    idx = torch.searchsorted(t_ref, t_query).clamp(1, n_ref - 1)
    t0 = t_ref[idx - 1]
    t1 = t_ref[idx]
    weight = ((t_query - t0) / (t1 - t0).clamp_min(1e-12)).unsqueeze(-1)
    return y_ref[idx - 1] * (1.0 - weight) + y_ref[idx] * weight


def _autograd_time_derivative(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute d(values)/dt column-wise with ``create_graph=True``."""
    derivatives = []
    n_col = values.shape[1]
    for col in range(n_col):
        grad_col = torch.autograd.grad(
            values[:, col].sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0]
        derivatives.append(grad_col)
    return torch.cat(derivatives, dim=1)


class _TimeMLP(nn.Module):
    """Map scalar time to a scalar well response (rate or pressure)."""

    def __init__(
        self,
        hidden_layers: int = 4,
        hidden_units: int = 20,
        nonnegative: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 1
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units, dtype=_DTYPE))
            layers.append(nn.Tanh())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, 1, dtype=_DTYPE))
        self.net = nn.Sequential(*layers)
        self.nonnegative = nonnegative
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        output = self.net(t_norm)
        if self.nonnegative:
            return F.softplus(output)
        return output


class _PhysicalParameterHead(nn.Module):
    """Constrained physical CRM parameters with tau-selection-aware shapes.

    Inter-well connectivity :math:`f_{ij}` lives in a ``(n_prod, n_inj)`` tensor.
    Time constants are ``(n_prod, 1)`` for ``per-producer`` (CRM-P) and
    ``(n_prod, n_inj)`` for ``per-pair`` (CRM-IP). Productivity indices :math:`J_j`
    are one non-negative value per producer.
    """

    def __init__(
        self,
        n_inj: int,
        n_prod: int,
        tau_selection: str,
        constraints: str,
        primary: bool,
        with_pressure: bool,
    ):
        super().__init__()
        if tau_selection not in ("per-pair", "per-producer"):
            msg = f'tau_selection must be one of ("per-pair","per-producer"), not {tau_selection}'
            raise ValueError(msg)
        self.n_inj = n_inj
        self.n_prod = n_prod
        self.tau_selection = tau_selection
        self.constraints = constraints
        self.primary = primary
        self.with_pressure = with_pressure

        self.raw_gains = nn.Parameter(torch.zeros(n_prod, n_inj, dtype=_DTYPE))
        self.raw_gain_slack = nn.Parameter(torch.zeros(n_inj, dtype=_DTYPE))

        tau_shape = (n_prod, n_inj) if tau_selection == "per-pair" else (n_prod, 1)
        self.raw_tau = nn.Parameter(torch.zeros(tau_shape, dtype=_DTYPE))

        self.raw_gain_producer = nn.Parameter(torch.zeros(n_prod, dtype=_DTYPE))
        self.raw_tau_producer = nn.Parameter(torch.zeros(n_prod, dtype=_DTYPE))
        self.raw_J = nn.Parameter(torch.zeros(n_prod, dtype=_DTYPE))
        self.raw_gain_pressure = nn.Parameter(torch.zeros(n_prod, n_prod, dtype=_DTYPE))

    def initialize_from_numpy(
        self,
        gains: NDArray,
        tau: NDArray,
        gains_producer: NDArray | None = None,
        tau_producer: NDArray | None = None,
        productivity: NDArray | None = None,
        gain_pressure: NDArray | None = None,
    ) -> None:
        """Set unconstrained raw tensors so activations match the given values."""
        gains = np.ascontiguousarray(gains, dtype=np.float64)
        gains = np.where(np.isfinite(gains), gains, 0.0)
        tau = np.ascontiguousarray(tau, dtype=np.float64)
        if tau.ndim == 1:
            tau = tau.reshape(-1, 1)

        with torch.no_grad():
            if self.constraints == "up-to one":
                # Softmax over producers+slack per injector; do not row-normalize first.
                column_sum = np.clip(gains.sum(axis=0), 0.0, 1.0 - 1e-6)
                slack = np.clip(1.0 - column_sum, 1e-6, None)
                stacked = np.vstack([np.clip(gains, 1e-8, None), slack.reshape(1, -1)])
                logits = np.log(stacked)
                logits -= logits.mean(axis=0, keepdims=True)
                self.raw_gains.copy_(torch.from_numpy(logits[:-1]))
                self.raw_gain_slack.copy_(torch.from_numpy(logits[-1]))
            elif self.constraints == "sum-to-one":
                row = np.clip(gains, 1e-8, None)
                row_total = row.sum(axis=1, keepdims=True)
                row_total = np.where(np.isfinite(row_total) & (row_total > 0.0), row_total, 1.0)
                row = row / row_total
                logits = np.log(row)
                logits -= logits.mean(axis=1, keepdims=True)
                self.raw_gains.copy_(torch.from_numpy(logits))
                self.raw_gain_slack.zero_()
            else:
                self.raw_gains.copy_(torch.from_numpy(_inv_softplus(np.clip(gains, 0.0, None))))
                self.raw_gain_slack.zero_()

            self.raw_tau.copy_(torch.from_numpy(_inv_softplus(np.clip(tau, _TAU_FLOOR, None))))

            if gains_producer is None:
                gains_producer = np.zeros(self.n_prod)
            if tau_producer is None:
                tau_producer = np.ones(self.n_prod)
            self.raw_gain_producer.copy_(
                torch.from_numpy(_inv_softplus(np.clip(gains_producer, 1e-8, None)))
            )
            self.raw_tau_producer.copy_(
                torch.from_numpy(_inv_softplus(np.clip(tau_producer, _TAU_FLOOR, None)))
            )

            if productivity is None:
                productivity = np.ones(self.n_prod)
            self.raw_J.copy_(torch.from_numpy(_inv_softplus(np.clip(productivity, 1e-8, None))))

            if gain_pressure is None:
                gain_pressure = np.eye(self.n_prod)
            pressure_gains = np.ascontiguousarray(gain_pressure, dtype=np.float64)
            self.raw_gain_pressure.copy_(torch.from_numpy(pressure_gains))

    def forward(self) -> dict[str, torch.Tensor]:
        if self.constraints == "up-to one":
            logits = torch.cat([self.raw_gains, self.raw_gain_slack.unsqueeze(0)], dim=0)
            gains = torch.softmax(logits, dim=0)[:-1]
        elif self.constraints == "sum-to-one":
            gains = torch.softmax(self.raw_gains, dim=1)
        else:
            gains = F.softplus(self.raw_gains)

        tau = F.softplus(self.raw_tau) + _TAU_FLOOR
        if self.primary:
            gains_producer = F.softplus(self.raw_gain_producer)
            tau_producer = F.softplus(self.raw_tau_producer) + _TAU_FLOOR
        else:
            gains_producer = torch.zeros(self.n_prod, dtype=_DTYPE, device=gains.device)
            tau_producer = torch.ones(self.n_prod, dtype=_DTYPE, device=gains.device)

        productivity = F.softplus(self.raw_J)
        gain_pressure = self.raw_gain_pressure
        return {
            "gains": gains,
            "tau": tau,
            "gains_producer": gains_producer,
            "tau_producer": tau_producer,
            "J": productivity,
            "gain_pressure": gain_pressure,
        }

    def constraint_penalty(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Soft penalties that reinforce activation mappings."""
        gains = params["gains"]
        tau = params["tau"]
        penalty = torch.relu(-gains).sum() + torch.relu(-tau).sum()
        penalty = penalty + torch.relu(-params["gains_producer"]).sum()
        penalty = penalty + torch.relu(-params["tau_producer"]).sum()
        penalty = penalty + torch.relu(-params["J"]).sum()
        if self.constraints == "up-to one":
            penalty = penalty + torch.relu(gains.sum(dim=0) - 1.0).sum()
            penalty = penalty + torch.relu(gains - 1.0).sum()
        elif self.constraints == "sum-to-one":
            penalty = penalty + (gains.sum(dim=1) - 1.0).pow(2).sum()
        return penalty


class _DeepCRM(nn.Module):
    """Multi-producer Deep-CRM PINN (Yewgat et al., ECMOR 2020).

    Each producer :math:`j` has a rate network :math:`t \\mapsto \\hat q_j(t)`.
    When BHP is observed, a parallel pressure network
    :math:`t \\mapsto \\hat p_{wf,j}(t)` supplies a smooth
    :math:`dp_{wf,j}/dt` through automatic differentiation.
    """

    def __init__(
        self,
        n_inj: int,
        n_prod: int,
        tau_selection: str,
        constraints: str,
        primary: bool,
        with_pressure: bool,
        hidden_layers: int = 4,
        hidden_units: int = 20,
        t_min: float = 0.0,
        t_span: float = 1.0,
        q_scale: float = 1.0,
        p_scale: float = 1.0,
        q0: NDArray | None = None,
    ):
        super().__init__()
        self.n_inj = n_inj
        self.n_prod = n_prod
        self.primary = primary
        self.with_pressure = with_pressure
        self.parameter_head = _PhysicalParameterHead(
            n_inj=n_inj,
            n_prod=n_prod,
            tau_selection=tau_selection,
            constraints=constraints,
            primary=primary,
            with_pressure=with_pressure,
        )
        self.q_nets = nn.ModuleList(
            [_TimeMLP(hidden_layers, hidden_units, nonnegative=True) for _ in range(n_prod)]
        )
        if with_pressure:
            self.p_nets: nn.ModuleList | None = nn.ModuleList(
                [_TimeMLP(hidden_layers, hidden_units, nonnegative=True) for _ in range(n_prod)]
            )
        else:
            self.p_nets = None

        self.register_buffer("t_min", torch.tensor(float(t_min), dtype=_DTYPE))
        self.register_buffer("t_span", torch.tensor(max(float(t_span), 1e-12), dtype=_DTYPE))
        self.register_buffer("q_scale", torch.tensor(max(float(q_scale), 1e-12), dtype=_DTYPE))
        self.register_buffer("p_scale", torch.tensor(max(float(p_scale), 1e-12), dtype=_DTYPE))
        if q0 is None:
            q0 = np.zeros(n_prod, dtype=np.float64)
        self.register_buffer("q0", torch.as_tensor(q0, dtype=_DTYPE))

    def normalize_time(self, t_phys: torch.Tensor) -> torch.Tensor:
        return (t_phys - self.t_min) / self.t_span

    def _stack_nets(self, nets: nn.ModuleList, t_norm: torch.Tensor) -> torch.Tensor:
        return torch.cat([net(t_norm) for net in nets], dim=1)

    def forward(self, t_phys: torch.Tensor) -> dict[str, torch.Tensor]:
        t_phys = t_phys.reshape(-1, 1)
        t_norm = self.normalize_time(t_phys)
        q_scaled = self._stack_nets(self.q_nets, t_norm)
        q_phys = q_scaled * self.q_scale
        dq_dt = _autograd_time_derivative(q_phys, t_phys)

        result: dict[str, torch.Tensor] = {"q": q_phys, "dq_dt": dq_dt, "t": t_phys}
        if self.p_nets is not None:
            p_scaled = self._stack_nets(self.p_nets, t_norm)
            p_phys = p_scaled * self.p_scale
            dp_dt = _autograd_time_derivative(p_phys, t_phys)
            result["p"] = p_phys
            result["dp_dt"] = dp_dt
        return result

    def effective_tau(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Producer-level time constant, reducing per-pair :math:`\\tau_{ij}` if needed."""
        tau = params["tau"]
        gains = params["gains"]
        if tau.shape[1] == 1:
            return tau.reshape(self.n_prod)
        weights = gains / gains.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return (weights * tau).sum(dim=1)

    def physics_residual(
        self,
        outputs: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        injection: torch.Tensor,
    ) -> torch.Tensor:
        r"""CRM-P residual used as :math:`\mathcal{L}_{physics}`.

        .. math::
            \tau_j \dot q_j + q_j - \sum_i f_{ij} I_i + \tau_j J_j \dot p_{wf,j}

        Primary production is removed before the material-balance residual is
        formed, matching the split used by the classical CRM.
        """
        q = outputs["q"]
        dq_dt = outputs["dq_dt"]
        t = outputs["t"].reshape(-1)
        tau = self.effective_tau(params).reshape(1, -1)
        inj_support = injection @ params["gains"].T

        q_sec = q
        dq_sec = dq_dt
        if self.primary:
            gain_p = params["gains_producer"].reshape(1, -1)
            tau_p = params["tau_producer"].reshape(1, -1)
            q_primary_t = self.q0.reshape(1, -1) * gain_p * torch.exp(-t.reshape(-1, 1) / tau_p)
            dq_primary_t = -q_primary_t / tau_p
            q_sec = q - q_primary_t
            dq_sec = dq_dt - dq_primary_t

        residual = tau * dq_sec + q_sec - inj_support
        if self.with_pressure and "dp_dt" in outputs:
            productivity = params["J"].reshape(1, -1)
            residual = residual + tau * productivity * outputs["dp_dt"]
        return residual

    def data_losses(
        self,
        outputs: dict[str, torch.Tensor],
        q_obs: torch.Tensor,
        p_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rate_loss = F.mse_loss(outputs["q"], q_obs)
        if p_obs is None or "p" not in outputs:
            pressure_loss = q_obs.new_zeros(())
        else:
            pressure_loss = F.mse_loss(outputs["p"], p_obs)
        return rate_loss, pressure_loss


def _extract_pinn_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate historical ``scipy.optimize`` kwargs plus PINN hyperparameters."""
    options = dict(kwargs.pop("options", None) or {})
    maxiter = options.pop("maxiter", kwargs.pop("maxiter", None))
    epochs = kwargs.pop("epochs", maxiter)
    if epochs is None:
        epochs = 2000
    lr = kwargs.pop("lr", options.pop("lr", 1e-3))
    for key in list(_SCIPY_FIT_KEYS):
        kwargs.pop(key, None)
        options.pop(key, None)
    options.pop("disp", None)
    options.pop("ftol", None)
    options.pop("gtol", None)

    pinn_opts = {
        "epochs": int(epochs),
        "lr": float(lr),
        "hidden_layers": int(kwargs.pop("hidden_layers", 4)),
        "hidden_units": int(kwargs.pop("hidden_units", 20)),
        "data_weight": float(kwargs.pop("data_weight", 1.0)),
        "physics_weight": float(kwargs.pop("physics_weight", 1.0)),
        "pressure_weight": float(kwargs.pop("pressure_weight", 1.0)),
        "constraint_weight": float(kwargs.pop("constraint_weight", 1.0)),
        "collocation_multiplier": float(kwargs.pop("collocation_multiplier", 1.0)),
        "optimizer": str(kwargs.pop("optimizer", "rmsprop")).lower(),
        "device": kwargs.pop("device", "cpu"),
        "seed": kwargs.pop("seed", None),
    }
    kwargs.pop("options", None)
    return pinn_opts


class CRM:
    """A Capacitance Resistance Model history matcher.

    CRM uses a physics-inspired mass balance approach to explain production for
    waterfloods. It treats each injector-producer well pair as a system
    with mass input, output, and pressure related to the mass balance.
    Several versions exist and can be selected from the arguments.

    Fitting is performed with a Deep-CRM physics-informed neural network
    (Yewgat et al., ECMOR 2020). The identified connectivities and time
    constants are then used by :meth:`predict`.

    The default arguments give the best results for most scenarios, but they
    can be sub-optimal if there is insufficient data, and they run slower than
    models with more simplifying assumptions.

    Args
    ----------
    primary : bool
        Whether to model primary production (True is strongly recommended)
    tau_selection : str
        How many tau values to select
            - If 'per-pair', fit tau for each producer-injector pair
            - If 'per-producer', fit tau for each producer (CRMP model)
    constraints : str
        How to constrain the gains
            * If 'up-to one' (default), let gains vary from 0 (no connection) to 1 \
           (all injection goes to producer) and enforce :math:`\\sum_j f_{ij} \\le 1`
            * If 'positive', require each gain to be positive \
                (It is unlikely to go negative in real life)
            * If 'sum-to-one', require the gains for each injector to sum to one \
                            (all production accounted for)
            * If 'sum-to-one injector' (not implemented), require each injector's \
                gains to sum to one (all injection accounted for)

    Examples
    ----------
    >>> crm = CRM(True, "per-pair", "up-to one")

    References
    ----------
    "A State-of-the-Art Literature Review on Capacitance Resistance Models for
    Reservoir Characterization and Performance Forecasting" - Wanderley de Holanda
    et al., 2018. https://www.mdpi.com/1996-1073/11/12/3368

    "Deep-CRM: A New Deep Learning Approach For Capacitance Resistive Models"
    - Yewgat, Busby, Chevalier, Lapeyre, Teste, ECMOR XVII, 2020.
    """

    def __init__(
        self,
        primary: bool = True,
        tau_selection: str = "per-pair",
        constraints: str = "positive",
    ):
        """Initialize CRM with appropriate settings."""
        if not isinstance(primary, bool):
            msg = "primary must be a boolean"
            raise TypeError(msg)
        self.primary = primary
        if constraints not in _VALID_CONSTRAINTS:
            msg = "Invalid constraints"
            raise ValueError(msg)
        self.constraints = constraints
        self.tau_selection = tau_selection
        if tau_selection == "per-pair":
            self.q_CRM = q_CRM_perpair
        elif tau_selection == "per-producer":
            self.q_CRM = q_CRM_perproducer
        else:
            msg = f'tau_selection must be one of ("per-pair","per-producer"), not {tau_selection}'
            raise ValueError(msg)

    def fit(
        self,
        production: NDArray,
        injection: NDArray,
        time: NDArray,
        initial_guess: NDArray | str | None = None,
        num_cores: int = 1,
        random: bool = False,
        **kwargs,
    ):
        """Build a CRM model from the production and injection data.

        History matching uses a Deep-CRM PINN. Keyword arguments accept both
        the historical ``scipy.optimize`` names (``options={"maxiter": ...}``)
        and PINN hyperparameters (``epochs``, ``lr``, ``hidden_layers``,
        ``physics_weight``, ...).

        Args
        ----------
        production : NDArray
            production rates for each time period,
            shape: (n_time, n_producers)
        injection : NDArray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        time : NDArray
            relative time for each rate measurement, starting from 0,
            shape: (n_time)
        initial_guess : NDArray or str, optional
            Per-producer ``x0`` vectors, or one of ``'uniform'``, ``'random'``,
            ``'geometry'``. ``'geometry'`` seeds connectivities from
            :func:`pywaterflood.multiwellproductivity.geometry_crm_gains`
            (``-\\Lambda``) and requires ``locations``, ``x_e``, and ``y_e``.
        num_cores : int
            number of CPU threads for PyTorch, defaults to 1
        random : bool
            whether to randomly initialize the gains when ``initial_guess``
            is omitted
        **kwargs:
            PINN training options, including ``epochs`` / ``options['maxiter']``,
            ``lr``, ``hidden_layers``, ``hidden_units``, ``physics_weight``,
            ``data_weight``, ``constraint_weight``, ``optimizer``, ``device``.
            Geometry init also accepts ``locations``, ``x_e``, ``y_e``,
            ``producer_ids``, and ``injector_ids``.

        Returns
        ----------
        self: trained model

        Example
        -------
        >>> gh_url = (
        ...     "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
        ... )
        >>> prod = pd.read_csv(gh_url + "production.csv", header=None).values
        >>> inj = pd.read_csv(gh_url + "injection.csv", header=None).values
        >>> time = pd.read_csv(gh_url + "time.csv", header=None).values[:, 0]
        >>> crm = CRM(True, "per-pair", "up-to one")
        >>> crm.fit(prod, inj, time)
        """
        _validate_inputs(production, injection, time)
        self.production = production
        self.injection = injection
        self.time = time

        self._get_bounds()

        geometry = _pop_geometry_fit_kwargs(kwargs)
        initial_guess = self._coerce_initial_guess(initial_guess, random, geometry)
        self._fit_pinn(
            pressure=None,
            initial_guess=initial_guess,
            num_cores=num_cores,
            random=random,
            **kwargs,
        )
        return self

    def predict(self, injection=None, time=None, connections=None, production=None):
        """Predict production for a trained model.

        If the injection and time are not provided, this will use the training values

        Args
        ----------
        injection : Optional NDArray
            The injection rates to input to the system, shape (n_time, n_inj)
        time : Optional NDArray
            The timesteps to predict
        connections : Optional dict
            if present, the gains, tau, gains_producer, tau_producer matrices
        production : Optional NDArray
            The production (only takes first row to use for primary production decline)

        Returns
        ----------
        q_hat :NDArray
            The predicted values, shape (n_time, n_producers)

        Example
        -------
        Using the synthetic reservoir:

        >>> gh_url = (
        ...     "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
        ... )
        >>> prod = pd.read_csv(gh_url + "production.csv", header=None).values
        >>> inj = pd.read_csv(gh_url + "injection.csv", header=None).values
        >>> time = pd.read_csv(gh_url + "time.csv", header=None).values[:, 0]
        >>> crm = CRM(True, "per-producer", "up-to one")
        >>> crm.fit(prod, inj, time)
        >>> crm.predict()

        Starting from a known model:

        >>> injection = np.ones((100, 2))
        >>> production = np.ones((1, 1)) * 2
        >>> time = np.arange(100, dtype=float)
        >>> connections = {
        ...     "gains": np.ones((2, 1)) * 0.95,
        ...     "tau": np.ones((2, 1)) * 3,
        ...     "gains_producer": np.zeros(1),
        ...     "tau_producer": np.ones(1),
        ... }
        >>> crm = CRM(False, "per-pair")
        >>> crm.predict(injection, time, connections=connections, production=production)
        """
        if production is None:
            production = self.production
        n_producers = production.shape[1]

        if connections is not None:
            gains = connections.get("gains")
            if gains is None:
                gains = self.gains
            tau = connections.get("tau")
            if tau is None:
                tau = self.tau
            gains_producer = connections.get("gains_producer")
            if gains_producer is None:
                gains_producer = self.gains_producer if self.primary else np.zeros(n_producers)
            tau_producer = connections.get("tau_producer")
            if tau_producer is None:
                tau_producer = self.tau_producer if self.primary else np.ones(n_producers)
        else:
            gains = self.gains
            tau = self.tau
            gains_producer = self.gains_producer
            tau_producer = self.tau_producer

        if int(injection is None) + int(time is None) == 1:
            msg = "Either both or neither of injection or time must be specified"
            raise TypeError(msg)
        if injection is None:
            injection = self.injection
        if time is None:
            time = self.time
        if time.shape[0] != injection.shape[0]:
            msg = "injection and time need same number of steps"
            raise ValueError(msg)

        q_hat = np.zeros((len(time), n_producers))
        for i in range(n_producers):
            if self.primary:
                q_hat[:, i] += q_primary(
                    production[:, i], time, gains_producer[i], tau_producer[i]
                )
            q_hat[:, i] += self.q_CRM(injection, time, gains[i, :], tau[i])
        return q_hat

    def set_rates(self, production=None, injection=None, time=None):
        """Set production and injection rates and time array.

        Args
        -----
        production : NDArray
            production rates with shape (n_time, n_producers)
        injection : NDArray
            injection rates with shape (n_time, n_injectors)
        time : NDArray
            timesteps with shape n_time

        Example
        -------
        >>> injection = np.ones((100,2))
        >>> production = np.full((100,1), 2.0)
        >>> time = np.arange(100, dtype=float)
        >>> crm = CRM()
        >>> crm.set_rates(production, injection, time)
        """
        _validate_inputs(production, injection, time)
        if production is not None:
            self.production = production
        if injection is not None:
            self.injection = injection
        if time is not None:
            self.time = time

    def set_connections(self, gains=None, tau=None, gains_producer=None, tau_producer=None):
        """Set waterflood properties.

        Args
        -----
        gains : NDArray
            connectivity between injector and producer
            shape: n_gains, n_producers
        tau : NDArray
            time-constant for injection to be felt by production
            shape: either n_producers or (n_gains, n_producers)
        gains_producer : NDArray
            gain on primary production, shape: n_producers
        tau_producer : NDArray
            Arps time constant for primary production, shape: n_producers

        Example
        -------
        >>> crm = CRM(False, "per-pair")
        >>> gains = np.full((2, 1),0.95)
        >>> tau = np.full((2, 1), 3.0)
        >>> crm.set_connections(gains, tau)
        """
        if gains is not None:
            self.gains = gains
        if tau is not None:
            self.tau = tau
        if gains_producer is not None:
            self.gains_producer = gains_producer
        if tau_producer is not None:
            self.tau_producer = tau_producer

    def residual(self, production=None, injection=None, time=None):
        """Calculate the production minus the predicted production for a trained model.

        If the production, injection, and time are not provided, this will use the
        training values

        Args
        ----------
        production : NDArray
            The production rates observed, shape: (n_timesteps, n_producers)
        injection : NDArray
            The injection rates to input to the system,
            shape: (n_timesteps, n_injectors)
        time : NDArray
            The timesteps to predict

        Returns
        ----------
        residual : NDArray
            The true production data minus the predictions, shape (n_time, n_producers)
        """
        q_hat = self.predict(injection, time)
        if production is None:
            production = self.production
        return production - q_hat

    def to_excel(self, fname: str):
        """Write trained model to an Excel file.

        Args
        ----
        fname : str
            Excel file to write out

        """
        for x in ("gains", "tau", "gains_producer", "tau_producer"):
            if x not in self.__dict__:
                msg = "Model has not been trained"
                raise ValueError(msg)
        with pd.ExcelWriter(fname) as f:
            pd.DataFrame(self.gains).to_excel(f, sheet_name="Gains")
            pd.DataFrame(self.tau).to_excel(f, sheet_name="Taus")
            pd.DataFrame(
                {
                    "Producer gains": self.gains_producer,
                    "Producer taus": self.tau_producer,
                }
            ).to_excel(f, sheet_name="Primary production")

    def to_pickle(self, fname: str):
        """Write trained model to a pickle file.

        Args
        -----
        fname : str
            pickle file to write out
        """
        with Path(fname).open("wb") as f:
            pickle.dump(self, f)

    def _gains_from_geometry(
        self,
        locations=None,
        x_e=None,
        y_e=None,
        producer_ids=None,
        injector_ids=None,
    ) -> NDArray:
        """Build ``(n_producers, n_injectors)`` CRM gains from well geometry."""
        missing = [
            name
            for name, value in (("locations", locations), ("x_e", x_e), ("y_e", y_e))
            if value is None
        ]
        if missing:
            msg = (
                "Geometry-based initialization requires "
                + ", ".join(missing)
                + ". Pass a well map as locations (DataFrame with columns X, Y, Type "
                "where Type is 'Producer' or 'Injector') plus reservoir extents x_e "
                "and y_e. Optional producer_ids and injector_ids must match production "
                "and injection column order."
            )
            raise ValueError(msg)
        n_prod = self.production.shape[1]
        n_inj = self.injection.shape[1]
        return geometry_crm_gains(
            locations,
            x_e,
            y_e,
            producer_ids=producer_ids,
            injector_ids=injector_ids,
            n_prod=n_prod,
            n_inj=n_inj,
        )

    def _coerce_initial_guess(self, initial_guess, random: bool, geometry: dict[str, Any] | None):
        """Resolve ``fit`` guess arguments into per-producer ``x0`` vectors."""
        geometry = geometry or {}
        mode = _initial_guess_mode(initial_guess, random)
        if mode is None:
            return initial_guess
        if mode == "geometry":
            gains = self._gains_from_geometry(**geometry)
            return self._get_initial_guess(random=False, gains=gains)
        return self._get_initial_guess(random=(mode == "random"))

    def _get_initial_guess(self, tau_selection: str | None = None, random=False, gains=None):
        """Create initial guesses for the CRM model parameters.

        :meta private:

        Args
        ----------
        tau_selection : str, one of 'per-pair' or 'per-producer'
            sets whether to use CRM (per-pair) or CRMp model
        random : bool
            whether initial gains are randomly (true) or proportionally assigned
        gains : NDArray, optional
            Connectivity matrix of shape ``(n_producers, n_injectors)``. When
            given it is used as-is (no extra row/column normalization).
        Returns
        ----------
        x0 : NDArray
            Initial primary production gain, time constant and waterflood gains
            and time constants, as one long 1-d array
        """
        if tau_selection is not None:
            self.tau_selection = tau_selection

        n_inj = self.injection.shape[1]
        n_prod = self.production.shape[1]
        d_t = self.time[1] - self.time[0]

        axis = 1 if (self.constraints == "sum-to-one injector") else 0
        if gains is not None:
            gains_guess1 = np.ascontiguousarray(np.asarray(gains, dtype=np.float64))
            if gains_guess1.shape != (n_prod, n_inj):
                msg = (
                    "geometry/connectivity gains must have shape "
                    f"(n_producers, n_injectors)=({n_prod}, {n_inj}), "
                    f"got {gains_guess1.shape}"
                )
                raise ValueError(msg)
            gains_producer_guess1 = np.ones(n_prod)
        elif random:
            rng = np.random.default_rng()
            gains_producer_guess1 = rng.random(n_prod)
            gains_guess1 = random_weights(n_prod, n_inj, axis)
        else:
            gains_unnormed = np.ones((n_prod, n_inj))
            gains_guess1 = gains_unnormed / np.sum(gains_unnormed, axis, keepdims=True)
            gains_producer_guess1 = np.ones(n_prod)
        tau_producer_guess1 = d_t * np.ones(n_prod)
        if self.tau_selection == "per-pair":
            tau_guess1 = d_t * np.ones((n_prod, n_inj))
        else:  # 'per-producer'
            tau_guess1 = d_t * np.ones((n_prod, 1))

        if self.primary:
            x0 = [
                np.concatenate(
                    [
                        gains_guess1[i, :],
                        tau_guess1[i, :],
                        gains_producer_guess1[[i]],
                        tau_producer_guess1[[i]],
                    ]
                )
                for i in range(n_prod)
            ]
        else:
            x0 = [np.concatenate([gains_guess1[i, :], tau_guess1[i, :]]) for i in range(n_prod)]
        return x0

    def _opt_numbers(self) -> tuple[int, int, int]:
        """Return the number of gains, taus, and primary production parameters to fit."""
        n_gains = self.injection.shape[1]
        n_tau = n_gains if self.tau_selection == "per-pair" else 1
        n_primary = 2 if self.primary else 0
        return n_gains, n_tau, n_primary

    def _get_bounds(self, constraints: str = "") -> tuple[tuple, tuple | dict]:
        """Create bounds for the model from initialized constraints."""
        if constraints:
            self.constraints = constraints

        n_inj = self.injection.shape[1]
        n = sum(self._opt_numbers())

        if self.constraints == "positive":
            bounds = ((0, np.inf),) * n
            constraints_optimizer = ()  # type: tuple | dict
        elif self.constraints == "sum-to-one":
            bounds = ((0, np.inf),) * n

            def constrain(x):
                x = x[:n_inj]
                return np.sum(x) - 1

            constraints_optimizer = {"type": "eq", "fun": constrain}
        elif self.constraints == "up-to one":
            lb = np.full(n, 0)
            ub = np.full(n, np.inf)
            ub[:n_inj] = 1
            bounds = tuple(zip(lb, ub))
            constraints_optimizer = ()
        elif self.constraints == "sum-to-one injector":
            msg = "sum-to-one injector is not implemented"
            raise NotImplementedError(msg)
        else:
            msg = (
                f"Constraint must be valid, not {self.constraints}.\n"
                "For least constrained, use 'positive'"
            )
            raise ValueError(msg)
        return bounds, constraints_optimizer

    def _calculate_qhat(
        self,
        x: NDArray,
        production: NDArray,
        injection: NDArray,
        time: NDArray,
    ):
        gains, tau, gain_producer, tau_producer = self._split_opts(x)
        q_hat = np.zeros(len(time))
        if self.primary:
            q_hat += q_primary(production, time, gain_producer, tau_producer)

        q_hat += self.q_CRM(injection, time, gains, tau)
        return q_hat

    def _split_opts(self, x: NDArray):
        n_inj = self.injection.shape[1]

        gains = x[:n_inj]
        tau = x[n_inj : n_inj * 2] if self.tau_selection == "per-pair" else x[n_inj]
        if self.primary:
            gain_producer = x[-2]
            tau_producer = x[-1]
        else:
            gain_producer = 0
            tau_producer = 1
        if self.tau_selection == "per-pair":
            tau[tau < 1e-10] = 1e-10
        elif tau < 1e-10:
            tau = 1e-10
        tau_producer = max(tau_producer, 1e-10)
        return gains, tau, gain_producer, tau_producer

    def _guess_matrices(self, initial_guess) -> dict[str, NDArray]:
        """Unpack per-well ``x0`` vectors into full physical-parameter arrays."""
        n_prod = self.production.shape[1]
        n_inj = self.injection.shape[1]
        gains = np.zeros((n_prod, n_inj), dtype=np.float64)
        if self.tau_selection == "per-pair":
            tau = np.zeros((n_prod, n_inj), dtype=np.float64)
        else:
            tau = np.zeros((n_prod, 1), dtype=np.float64)
        gains_producer = np.zeros(n_prod, dtype=np.float64)
        tau_producer = np.ones(n_prod, dtype=np.float64)
        n_pressure = self.production.shape[1]
        gain_pressure = np.eye(n_prod, dtype=np.float64)
        productivity = np.ones(n_prod, dtype=np.float64)

        def _first_scalar(value, default: float = 0.0) -> float:
            array = np.asarray(value, dtype=np.float64).reshape(-1)
            if array.size == 0:
                return default
            return float(array[0])

        for well, x0 in enumerate(initial_guess):
            split = self._split_opts(np.asarray(x0, dtype=np.float64))
            well_gains = np.asarray(split[0], dtype=np.float64).reshape(-1)
            well_tau = np.asarray(split[1], dtype=np.float64).reshape(-1)
            gains[well] = well_gains
            if self.tau_selection == "per-pair":
                tau[well] = well_tau
            else:
                tau[well, 0] = _first_scalar(well_tau, _TAU_FLOOR)
            gains_producer[well] = _first_scalar(split[2], 0.0)
            tau_producer[well] = _first_scalar(split[3], 1.0)
            if len(split) > 4:
                well_pressure = np.asarray(split[4], dtype=np.float64).reshape(-1)
                if well_pressure.size == n_pressure:
                    gain_pressure[well] = well_pressure
                    productivity[well] = float(np.mean(np.abs(well_pressure)))

        return {
            "gains": gains,
            "tau": tau,
            "gains_producer": gains_producer,
            "tau_producer": tau_producer,
            "gain_pressure": gain_pressure,
            "productivity": productivity,
        }

    def _fit_pinn(
        self,
        pressure: NDArray | None,
        initial_guess,
        num_cores: int,
        random: bool,
        **kwargs,
    ) -> None:
        """Train the Deep-CRM PINN and write identified parameters onto ``self``."""
        pinn_opts = _extract_pinn_options(kwargs)
        production = _as_f64_2d(self.production)
        injection = _as_f64_2d(self.injection)
        time = _as_1d_time(self.time)
        n_time, n_prod = production.shape
        n_inj = injection.shape[1]
        with_pressure = pressure is not None
        pressure_arr = _as_f64_2d(pressure) if with_pressure else None

        seed = pinn_opts["seed"]
        if seed is None and not random:
            seed = 0
        if seed is not None:
            torch.manual_seed(int(seed))

        t_min = float(time.min())
        t_span = float(time.max() - time.min()) or 1.0
        q_scale = float(np.max(np.abs(production))) or 1.0
        p_scale = 1.0
        if pressure_arr is not None:
            p_scale = float(np.max(np.abs(pressure_arr))) or 1.0

        device = _resolve_torch_device(pinn_opts["device"])
        model = _DeepCRM(
            n_inj=n_inj,
            n_prod=n_prod,
            tau_selection=self.tau_selection,
            constraints=self.constraints,
            primary=self.primary,
            with_pressure=with_pressure,
            hidden_layers=pinn_opts["hidden_layers"],
            hidden_units=pinn_opts["hidden_units"],
            t_min=t_min,
            t_span=t_span,
            q_scale=q_scale,
            p_scale=p_scale,
            q0=production[0],
        )
        guess = self._guess_matrices(initial_guess)
        model.parameter_head.initialize_from_numpy(
            gains=guess["gains"],
            tau=guess["tau"],
            gains_producer=guess["gains_producer"],
            tau_producer=guess["tau_producer"],
            productivity=guess["productivity"],
            gain_pressure=guess["gain_pressure"],
        )
        model.to(device)

        t_obs = torch.as_tensor(time, dtype=_DTYPE, device=device)
        q_obs = torch.as_tensor(production, dtype=_DTYPE, device=device)
        i_obs = torch.as_tensor(injection, dtype=_DTYPE, device=device)
        p_obs = (
            torch.as_tensor(pressure_arr, dtype=_DTYPE, device=device)
            if pressure_arr is not None
            else None
        )

        n_col = max(n_time, round(n_time * pinn_opts["collocation_multiplier"]))

        nn_params = [param for net in model.q_nets for param in net.parameters()]
        if model.p_nets is not None:
            nn_params.extend(param for net in model.p_nets for param in net.parameters())
        param_groups = [
            {"params": nn_params, "lr": pinn_opts["lr"]},
            {
                "params": list(model.parameter_head.parameters()),
                "lr": pinn_opts["lr"] * 5.0,
            },
        ]
        if pinn_opts["optimizer"] in ("adam",):
            optimizer: torch.optim.Optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.RMSprop(param_groups)

        prev_threads = torch.get_num_threads()
        torch.set_num_threads(max(1, int(num_cores)))
        history: list[dict[str, float]] = []
        try:
            model.train()
            for _epoch in range(pinn_opts["epochs"]):
                optimizer.zero_grad()
                t_data = t_obs.reshape(-1, 1).detach().requires_grad_(True)
                outputs_data = model(t_data)
                params = model.parameter_head()
                rate_loss, pressure_loss = model.data_losses(outputs_data, q_obs, p_obs)
                residual = model.physics_residual(outputs_data, params, i_obs)

                if n_col > n_time:
                    extra = torch.linspace(
                        float(time.min()),
                        float(time.max()),
                        n_col - n_time,
                        dtype=_DTYPE,
                        device=device,
                    )
                    t_col = extra.reshape(-1, 1).detach().requires_grad_(True)
                    outputs_col = model(t_col)
                    injection_col = _interp_along_time(t_col, t_obs, i_obs)
                    residual = torch.cat(
                        [residual, model.physics_residual(outputs_col, params, injection_col)],
                        dim=0,
                    )

                physics_loss = residual.pow(2).mean()
                constraint_loss = model.parameter_head.constraint_penalty(params)

                loss = (
                    pinn_opts["data_weight"] * rate_loss
                    + pinn_opts["physics_weight"] * physics_loss / (q_scale**2)
                    + pinn_opts["pressure_weight"] * pressure_loss
                    + pinn_opts["constraint_weight"] * constraint_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                history.append(
                    {
                        "loss": float(loss.detach()),
                        "data": float(rate_loss.detach()),
                        "physics": float(physics_loss.detach()),
                        "pressure": float(pressure_loss.detach()),
                        "constraints": float(constraint_loss.detach()),
                    }
                )
        finally:
            torch.set_num_threads(prev_threads)

        model.eval()
        with torch.no_grad():
            fitted = model.parameter_head()
        self.gains = _tensor_to_numpy(fitted["gains"])
        self.tau = _tensor_to_numpy(fitted["tau"])
        self.gains_producer = _tensor_to_numpy(fitted["gains_producer"])
        self.tau_producer = _tensor_to_numpy(fitted["tau_producer"])
        self.productivity_index = _tensor_to_numpy(fitted["J"])
        self.pinn_model_ = model
        self.loss_history_ = history
        if with_pressure:
            self.gain_pressure = _tensor_to_numpy(fitted["gain_pressure"])
            self.pressure = pressure_arr


class CrmCompensated(CRM):
    """Bottom-hole pressure compensated CRM."""

    def fit(
        self,
        production: NDArray,
        pressure: NDArray,
        injection: NDArray,
        time: NDArray,
        initial_guess: NDArray | str | None = None,
        num_cores: int = 1,
        random: bool = False,
        **kwargs,
    ):
        """Fit a CRM model from the production, pressure, and injection data.

        Args
        ----------
        production : NDArray
            production rates for each time period,
            shape: (n_time, n_producers)
        pressure : NDArray
            average pressure for each producer for each time period,
            shape: (n_time, n_producers)
        injection : NDArray
            injection rates for each time period,
            shape: (n_time, n_injectors)
        time : NDArray
            relative time for each rate measurement, starting from 0,
            shape: (n_time)
        initial_guess : NDArray or str, optional
            Per-producer ``x0`` vectors, or one of ``'uniform'``, ``'random'``,
            ``'geometry'``. Geometry init requires ``locations``, ``x_e``, ``y_e``.
        num_cores : int
            number of CPU threads for PyTorch, defaults to 1
        random : bool
            whether to randomly initialize the gains when ``initial_guess``
            is omitted
        **kwargs:
            PINN training options; ``options={'maxiter': N}`` is treated as
            epoch count for compatibility with the previous optimizer API.
            Geometry init also accepts ``locations``, ``x_e``, ``y_e``,
            ``producer_ids``, and ``injector_ids``.

        Returns
        ----------
        self: trained model
        """
        _validate_inputs(production, injection, time, pressure)
        self.production = production
        self.injection = injection
        self.time = time
        self.pressure = pressure

        self._get_bounds()

        geometry = _pop_geometry_fit_kwargs(kwargs)
        initial_guess = self._coerce_initial_guess(initial_guess, random, geometry)
        self._fit_pinn(
            pressure=pressure,
            initial_guess=initial_guess,
            num_cores=num_cores,
            random=random,
            **kwargs,
        )
        return self

    def predict(
        self,
        injection=None,
        time=None,
        connections=None,
        production=None,
        pressure=None,
    ):
        """Predict production for a trained model.

        If the injection and time are not provided, this will use the training values

        Args
        ----------
        injection : Optional NDArray
            The injection rates to input to the system, shape (n_time, n_inj)
        time : Optional NDArray
            The timesteps to predict
        connections : Optional dict
            if present, the gains, tau, gains_producer, tau_producer matrices
        production : Optional NDArray
            The production (only takes first row to use for primary production decline)

        Returns
        ----------
        q_hat :NDArray
            The predicted values, shape (n_time, n_producers)

        Example
        -------
        Using the synthetic reservoir:

        >>> gh_url = (
        ...     "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
        ... )
        >>> prod = pd.read_csv(gh_url + "production.csv", header=None).values
        >>> inj = pd.read_csv(gh_url + "injection.csv", header=None).values
        >>> time = pd.read_csv(gh_url + "time.csv", header=None).values[:, 0]
        >>> pressure = 1000 - prod * 0.1
        >>> crm = CrmCompensated(True, "per-producer", "up-to one")
        >>> crm.fit(prod, pressure, inj, time)
        >>> crm.predict()

        Starting from a known model:

        >>> injection = np.ones((100, 2))
        >>> production = np.ones((1, 1)) * 2
        >>> pressure = 1000 - production * 0.1
        >>> time = np.arange(100, dtype=float)
        >>> connections = {
        ...     "gains": np.ones((2, 1)) * 0.95,
        ...     "tau": np.ones((2, 1)) * 3,
        ...     "gains_producer": np.zeros(1),
        ...     "tau_producer": np.ones(1),
        ... }
        >>> crm = CRM(False, "per-pair")
        >>> crm.predict(injection, time, connections=connections, production=production)
        """
        if int(injection is None) + int(time is None) == 1:
            msg = "Either both or neither of injection or time must be specified"
            raise TypeError(msg)

        injection = self.injection if injection is None else injection
        time = self.time if time is None else time
        production = self.production if production is None else production
        pressure = self.pressure if pressure is None else pressure
        n_producers = production.shape[1]
        if time.shape[0] != injection.shape[0]:
            msg = "injection and time need same number of steps"
            raise ValueError(msg)

        if connections is not None:
            gains = connections.get("gains")
            if gains is None:
                gains = self.gains
            tau = connections.get("tau")
            if tau is None:
                tau = self.tau
            gains_producer = connections.get("gains_producer")
            if gains_producer is None:
                gains_producer = self.gains_producer if self.primary else np.zeros(n_producers)
            tau_producer = connections.get("tau_producer")
            if tau_producer is None:
                tau_producer = self.tau_producer if self.primary else np.ones(n_producers)
            gains_pressure = connections.get("gains_pressure")
            if gains_pressure is None:
                gains_pressure = self.gain_pressure
        else:
            gains = self.gains
            tau = self.tau
            gains_producer = self.gains_producer
            tau_producer = self.tau_producer
            gains_pressure = self.gain_pressure

        q_hat = np.zeros((len(time), n_producers))
        for i in range(n_producers):
            if self.primary:
                q_hat[:, i] += q_primary(
                    production[:, i], time, gains_producer[i], tau_producer[i]
                )
            q_hat[:, i] += self.q_CRM(injection, time, gains[i, :], tau[i])
            q_hat[:, i] += q_bhp(pressure[:, i], pressure, gains_pressure[i, :])
        return q_hat

    def _calculate_qhat(
        self,
        x: NDArray,
        production: NDArray,
        injection: NDArray,
        time: NDArray,
        pressure_local: NDArray,
        pressure: NDArray,
    ):
        gains, tau, gain_producer, tau_producer, gain_pressure = self._split_opts(x)
        if self.primary:
            q_hat = q_primary(production, time, gain_producer, tau_producer)
        else:
            q_hat = np.zeros(len(time))

        q_hat += self.q_CRM(injection, time, gains, tau)
        q_hat += q_bhp(pressure_local, pressure, gain_pressure)
        return q_hat

    def _opt_numbers(self) -> tuple[int, int, int, int]:
        n_gain, n_tau, n_primary = super()._opt_numbers()
        return n_gain, n_tau, n_primary, self.production.shape[1]

    def _split_opts(self, x: NDArray) -> tuple[NDArray, NDArray, Any, Any, NDArray]:
        n_gains, n_tau, n_primary = self._opt_numbers()[:3]
        n_connectivity = n_gains + n_tau

        gains = x[:n_gains]
        tau = np.asarray(x[n_gains:n_connectivity], dtype=np.float64)
        if self.primary:
            gain_producer = float(np.asarray(x[n_connectivity:][0], dtype=np.float64))
            tau_producer = float(np.asarray(x[n_connectivity:][1], dtype=np.float64))
        else:
            gain_producer = 0.0
            tau_producer = 1.0
        gain_pressure = np.asarray(x[n_connectivity + n_primary :], dtype=np.float64)

        if self.tau_selection == "per-pair":
            tau[tau < 1e-10] = 1e-10
        else:
            tau = max(float(np.ravel(tau)[0]), 1e-10)
        tau_producer = max(float(tau_producer), 1e-10)
        return gains, tau, gain_producer, tau_producer, gain_pressure

    def _get_initial_guess(self, tau_selection: str | None = None, random=False, gains=None):
        """Make the initial guesses for the CRM model parameters.

        :meta private:

        Args
        ----------
        tau_selection : str, one of 'per-pair' or 'per-producer'
            sets whether to use CRM (per-pair) or CRMp model
        random : bool
            whether initial gains are randomly assigned
        gains : NDArray, optional
            Connectivity matrix of shape ``(n_producers, n_injectors)``

        Returns
        ----------
        x0 : NDArray
            Initial primary production gain, time constant and waterflood gains
            and time constants, as one long 1-d array
        """
        guess = super()._get_initial_guess(
            tau_selection=tau_selection, random=random, gains=gains
        )
        _, _, _, n_pressure = self._opt_numbers()
        pressure_guess = np.ones(n_pressure)
        return [np.concatenate([guess[i], pressure_guess]) for i in range(len(guess))]


def _validate_inputs(
    production: NDArray | None = None,
    injection: NDArray | None = None,
    time: NDArray | None = None,
    pressure: NDArray | None = None,
) -> None:
    """Validate shapes and values of inputs.

    Args
    ----
    production : NDArray, optional
    injection : NDArray, optional
    time : NDArray, optional
    pressure : NDArray, optional

    Raises
    ------
    ValueError if timesteps don't match or production and pressure don't match
    """
    inputs = {
        "production": production,
        "injection": injection,
        "time": time,
        "pressure": pressure,
    }
    inputs = {key: val for key, val in inputs.items() if val is not None}
    # Shapes
    if time is not None:
        for timeseries in inputs:
            if inputs[timeseries].shape[0] != time.shape[0]:
                msg = f"{timeseries} and time do not have the same number of timesteps"
                raise ValueError(msg)
    if production is not None:
        if (injection is not None) and (production.shape[0] != injection.shape[0]):
            msg = "production and injection do not have the same number of timesteps"
            raise ValueError(msg)
        if (pressure is not None) and (production.shape != pressure.shape):
            msg = "production and pressure are not of the same shape"
            raise ValueError(msg)
    if (
        (injection is not None)
        and (pressure is not None)
        and (injection.shape[0] != pressure.shape[0])
    ):
        msg = "injection and pressure do not have the same number of timesteps"
        raise ValueError(msg)
    # Values
    for timeseries in inputs:
        if np.any(np.isnan(inputs[timeseries])):
            msg = f"{timeseries} cannot have NaNs"
            raise ValueError(msg)
        if np.any(inputs[timeseries] < 0.0):
            msg = f"{timeseries} cannot be negative"
            raise ValueError(msg)
