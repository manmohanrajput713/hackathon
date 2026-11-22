import numpy as np
import torch

from data.algebraic_nn import AlgebraicNet
from data.forecast import self_loop
from data.mpc import PINCMPC, casadi_skip_function
from data.param import get_well, nominal_input, nominal_state
from data.pinn_well import physics_residual
from data.skip_pinc import SkipPINC
from data.well_ode import algebraics_numpy


def _box():
    y0 = nominal_state(1)
    delta = np.array([150.0, 30.0, 300.0])
    return y0, y0 - delta, y0 + delta


def _center_net(skip: bool, n_layers=2, n_neurons=8):
    torch.manual_seed(0)
    net = SkipPINC(n_layers=n_layers, n_neurons=n_neurons, skip=skip).double()
    with torch.no_grad():
        net.out.weight.mul_(0.05)
        net.out.bias[:] = 0.5
    return net


def test_skip_forward_and_time_derivative():
    torch.manual_seed(0)
    net = SkipPINC().double()
    batch = 4
    t_s = torch.rand(batch, 1, dtype=torch.float64, requires_grad=True)
    y0 = torch.rand(batch, 3, dtype=torch.float64)
    u = torch.rand(batch, 2, dtype=torch.float64)
    features = torch.cat([t_s, y0, u], dim=1)
    output = net(features)
    assert output.shape == (batch, 3)
    grad = torch.autograd.grad(output.sum(), t_s)[0]
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_collocation_residual_has_hidden_gradient():
    well = get_well(1)
    _y0, y_min, y_max = _box()
    net = _center_net(skip=True)
    batch = 4
    t_s = torch.rand(batch, 1, dtype=torch.float64)
    y0_s = torch.full((batch, 3), 0.5, dtype=torch.float64)
    u = torch.tensor(nominal_input(), dtype=torch.float64).expand(batch, 2).clone()
    residual, _predicted, _ok = physics_residual(
        net, t_s, y0_s, u, y_min, y_max, 60.0, well, mask_infeasible=False
    )
    assert torch.isfinite(residual).all()
    residual.pow(2).mean().backward()
    grad = net.hidden[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_last_layer_residual_gradient_with_and_without_skip():
    well = get_well(1)
    _y0, y_min, y_max = _box()
    batch = 4
    t_s = torch.full((batch, 1), 0.3, dtype=torch.float64)
    y0_s = torch.full((batch, 3), 0.5, dtype=torch.float64)
    u = torch.tensor(nominal_input(), dtype=torch.float64).expand(batch, 2).clone()
    for skip in (True, False):
        net = _center_net(skip=skip)
        residual, _predicted, _ok = physics_residual(
            net, t_s, y0_s, u, y_min, y_max, 60.0, well, mask_infeasible=False
        )
        residual.pow(2).mean().backward()
        grad = net.out.weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()


def test_self_loop_three_steps():
    _y0, y_min, y_max = _box()
    net = SkipPINC(n_layers=2, n_neurons=8).double()
    u_seq = np.array([[0.4, 0.5], [0.6, 0.4], [0.5, 0.7]], dtype=np.float64)
    trajectory = self_loop(net, nominal_state(1), u_seq, y_min, y_max)
    assert trajectory.shape == (4, 3)
    assert np.isfinite(trajectory).all()


def test_casadi_skip_matches_torch():
    torch.manual_seed(1)
    net = SkipPINC(n_layers=2, n_neurons=6).double().eval()
    function = casadi_skip_function(net)
    rng = np.random.default_rng(1)
    for _ in range(6):
        features = rng.normal(size=6)
        torch_out = net(torch.tensor(features, dtype=torch.float64).reshape(1, 6)).detach().numpy().reshape(3)
        casadi_out = np.array(function(features.reshape(6, 1)), dtype=np.float64).reshape(3)
        assert np.max(np.abs(torch_out - casadi_out)) < 1e-5


def test_ipopt_one_step_respects_valve_bounds():
    well = get_well(1)
    y0, y_min, y_max = _box()
    torch.manual_seed(0)
    pinc = SkipPINC(n_layers=2, n_neurons=8).double()
    algebraic = AlgebraicNet(n_layers=2, n_neurons=8).double()
    z_mean = algebraics_numpy(y0, nominal_input(), well)
    z_std = np.ones(4, dtype=np.float64)
    controller = PINCMPC(pinc, algebraic, y_min, y_max, z_mean, z_std, horizon=3, control_horizon=2)
    valve = controller.solve(y0, nominal_input(), np.full(3, 95.0), first=True)
    assert valve.shape == (2,)
    assert np.all(valve >= -1e-6)
    assert np.all(valve <= 1.0 + 1e-6)
