import numpy as np
import torch

from data.algebraic_nn import AlgebraicNet
from data.param import get_well, nominal_state
from data.pinn_well import train_adam, train_algebraic, train_lbfgs
from data.sample import make_training_batches
from data.utils import load_checkpoint
from data.well_ode import is_feasible


def _tiny_domain():
    well = get_well(1)
    y0 = nominal_state(1)
    rng = np.random.default_rng(0)
    states = []
    valves = []
    for _ in range(4000):
        state = y0 * (1.0 + rng.normal(0.0, 0.015, size=3))
        valve = rng.uniform(0.25, 0.9, size=2)
        if is_feasible(state, valve, well, friction="haaland"):
            states.append(state)
            valves.append(valve)
        if len(states) >= 80:
            break
    assert len(states) >= 40
    delta = np.array([200.0, 40.0, 400.0])
    return {
        "y_min": y0 - delta,
        "y_max": y0 + delta,
        "y_pool": np.vstack(states),
        "u_pool": np.vstack(valves),
    }


def test_adam_then_lbfgs_smoke(tmp_path):
    domain = _tiny_domain()
    well = get_well(1)
    batches = make_training_batches(domain, nt=32, nf=64, n_val=2, well=well, seed=0)
    adam_dir = tmp_path / "adam"
    _model, _history, state = train_adam(
        domain,
        batches,
        adam_dir,
        well_id=1,
        epochs=2,
        lr=7e-3,
        seed=0,
        n_layers=2,
        n_neurons=8,
        skip=True,
        val_every=1,
    )
    assert np.isfinite(state.loss_train)
    checkpoint = load_checkpoint(adam_dir / "model.pt")
    assert "state_dict" in checkpoint
    assert checkpoint["friction"] == "cubic"
    restored = train_lbfgs(
        domain,
        batches,
        tmp_path / "lbfgs",
        init_path=adam_dir / "model.pt",
        well_id=1,
        iterations=2,
        seed=0,
    )
    assert np.isfinite(restored[2].loss_train)
    assert (tmp_path / "lbfgs" / "model.pt").is_file()


def test_algebraic_forward_and_one_lbfgs_step(tmp_path):
    torch.manual_seed(0)
    net = AlgebraicNet().double()
    output = net(torch.randn(5, 5, dtype=torch.float64))
    assert output.shape == (5, 4)
    assert torch.isfinite(output).all()
    domain = _tiny_domain()
    _model, payload = train_algebraic(
        domain,
        tmp_path / "algebraic",
        well_id=1,
        n_train=32,
        n_val=8,
        iterations=1,
        seed=0,
        n_layers=2,
        n_neurons=8,
    )
    assert np.isfinite(payload["z_mean"]).all()
    assert (tmp_path / "algebraic" / "model.pt").is_file()
