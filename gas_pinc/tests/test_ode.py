import numpy as np
import pytest
import torch

from data.friction import cubic_lambda
from data.param import get_well, nominal_input, nominal_state
from data.pinn_well import physics_residual
from data.well_ode import InfeasibleState, ode_numpy, ode_torch


def _relative(a, b):
    denom = np.maximum(np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def test_cubic_lambda_endpoints():
    well = get_well(1)
    low, ok_low = cubic_lambda(13000.0, well.cubic)
    high, ok_high = cubic_lambda(115000.0, well.cubic)
    assert ok_low and ok_high
    assert abs(low - 0.028) < 1e-3
    assert abs(high - 0.018) < 1e-3


def test_feasible_point_is_finite_for_both_closures():
    well = get_well(1)
    y = nominal_state(1)
    u = nominal_input()
    for friction in ("haaland", "cubic"):
        dy = ode_numpy(y, u, well, friction=friction)
        assert dy.shape == (3,)
        assert np.isfinite(dy).all()


def test_torch_matches_numpy():
    well = get_well(1)
    y = nominal_state(1)
    u = nominal_input()
    for friction in ("haaland", "cubic"):
        dy_np = ode_numpy(y, u, well, friction=friction)
        dy_th = ode_torch(y, u, well, friction=friction).detach().cpu().numpy()
        assert _relative(dy_th, dy_np) < 1e-5


def test_infeasible_liquid_mass_is_rejected_and_loss_is_not_nan():
    well = get_well(1)
    y = nominal_state(1).copy()
    y[2] = well.liquid_cap + 1000.0
    u = nominal_input()
    with pytest.raises(InfeasibleState):
        ode_numpy(y, u, well, friction="haaland")
    with pytest.raises(InfeasibleState):
        ode_torch(y, u, well, friction="cubic")

    class ConstantShift(torch.nn.Module):
        def forward(self, features):
            # Depends on the inputs so autograd is defined, and lands far outside the box.
            return 40.0 + 0.0 * features[:, :3]

    y0 = nominal_state(1)
    y_min = y0 - np.array([100.0, 20.0, 200.0])
    y_max = y0 + np.array([100.0, 20.0, 200.0])
    t_s = torch.zeros(2, 1, dtype=torch.float64, requires_grad=True)
    y0_s = torch.full((2, 3), 0.5, dtype=torch.float64)
    u_t = torch.full((2, 2), 0.5, dtype=torch.float64)
    with pytest.raises(InfeasibleState):
        physics_residual(
            ConstantShift(),
            t_s,
            y0_s,
            u_t,
            y_min,
            y_max,
            60.0,
            well,
            mask_infeasible=False,
        )
    residual, _predicted, _ok = physics_residual(
        ConstantShift(),
        t_s,
        y0_s,
        u_t,
        y_min,
        y_max,
        60.0,
        well,
        mask_infeasible=True,
    )
    loss = residual.pow(2).mean()
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    loss.backward()
