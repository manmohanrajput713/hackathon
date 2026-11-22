import os

import numpy as np
import torch

from data.BuildDataset import BuildingDataset, OrderedBatchLoader, split_sequences
from data.Logger import Logger
from data.forecast import XI_PHYS, UI_PHYS, free_run
from data.parameters import Parameters
from data.pinn_BCS import PINN_MODES, EncoderDecoderLSTM, RHO_SCALE, PI_SCALE, ed_bcs, pinn_vfm
from data.utils import Struct, dydt


def _safe_exogenous(n, steps=1):
    u = torch.zeros(n, steps, 4, dtype=torch.float32)
    u[:, :, 0] = 0.8
    u[:, :, 1] = 0.5
    u[:, :, 2] = 0.4
    u[:, :, 3] = 0.5
    return u


def _ordered_rows(n=16):
    x = torch.zeros(n, 20, 6, dtype=torch.float32)
    y = torch.zeros(n, 1, 2, dtype=torch.float32)
    u = _safe_exogenous(n, 1)
    for i in range(n):
        x[i] = 0.25 + 0.01 * i
        x[i, :, 4] = 0.40
        x[i, :, 5] = 0.35
        y[i, 0, 0] = 0.40
        y[i, 0, 1] = 0.35
    return x, y, u


def test_parameters_roundtrip_known_row():
    p = Parameters()
    u = np.array([[60.0, 100.0, 1.2e6, 1.2e7]], dtype=float)
    un = p.normalizar_u(u)
    np.testing.assert_allclose(un, np.array([[1.0, 1.0, 0.0, 0.0]]))
    u_back = un * np.asarray(p.uc, dtype=float) + np.asarray(p.u0, dtype=float)
    np.testing.assert_allclose(u_back, u)
    x = np.array([[2.0e6, 1.0e6, 0.01]], dtype=float)
    xn = p.normalizar_x(x)
    np.testing.assert_allclose(xn * p.xc + p.x0, x)
    x2 = np.array([[2.0e6, 1.0e6]], dtype=float)
    xn2 = p.normalizar_x(x2)
    np.testing.assert_allclose(xn2 * p.xc[:2] + p.x0[:2], x2)


def test_window_shapes_and_sequence_index(tmp_path):
    n = 100
    t = np.arange(n)
    x1 = np.linspace(2e6, 4e6, n).reshape(n, 1)
    x2 = np.linspace(1e6, 2e6, n).reshape(n, 1)
    x3 = np.linspace(0.008, 0.012, n).reshape(n, 1)
    U = np.column_stack([
        np.linspace(40, 60, n),
        np.linspace(30, 80, n),
        np.linspace(1.3e6, 1.8e6, n),
        np.linspace(1.22e7, 1.35e7, n),
    ])
    path = tmp_path / "tiny.npz"
    np.savez(path, t=t, x1=x1, x2=x2, x3=x3, U=U)
    raw = np.load(path)
    series = np.arange(30 * 7, dtype=float).reshape(30, 7)
    X, y, u = split_sequences(series, 20, 1)
    np.testing.assert_allclose(X[0], series[0:20])
    np.testing.assert_allclose(y[0, 0], series[20])
    np.testing.assert_allclose(u[0, 0], series[19])

    split_point = int(0.7 * len(raw["x1"]))
    ds = BuildingDataset(20, 1, raw, split_point, batch_size=500, seed=0)
    ds2 = BuildingDataset(20, 1, raw, split_point, batch_size=500, seed=0)
    assert ds.train_X.shape == (split_point, 20, 6)
    assert ds.train_y.shape == (split_point, 1, 2)
    assert ds.u_train.shape == (split_point, 1, 4)
    assert ds.test_y.ndim == 3 and ds.test_y.shape[1:] == (1, 3)
    assert ds.test_y.shape[0] > 0
    assert ds.train_X.dtype == np.float32
    assert np.isfinite(ds.train_X).all()
    assert np.isfinite(ds.train_y).all()
    assert np.isfinite(ds.u_train).all()
    assert np.isfinite(ds.test_y).all()
    np.testing.assert_array_equal(ds.train_X, ds2.train_X)
    np.testing.assert_array_equal(ds.train_y, ds2.train_y)


def test_forward_graph_and_grads():
    torch.manual_seed(0)
    model = EncoderDecoderLSTM(width=10, n_features=6, n_steps_in=20, n_steps_out=1, pinn_mode="on", var=(0.9, 0.9))
    assert model.width == 10
    assert model.encoder.input_size == 6
    assert model.encoder.hidden_size == 10
    assert model.decoder.input_size == 10
    assert model.decoder.hidden_size == 10
    model.eval()
    x = torch.randn(8, 20, 6)
    y = model(x)
    assert y.shape == (8, 1, 3)
    assert torch.isfinite(y).all()
    with torch.no_grad():
        _, (h_n, c_n) = model.encoder(x)
        repeated = h_n[-1].unsqueeze(1).repeat(1, 1, 1)
        dec_zero, _ = model.decoder(repeated)
        dec_state, _ = model.decoder(repeated, (h_n, c_n))
    delta = model.head(dec_zero)
    expected = torch.cat([x[:, -1:, 4:6] + delta[:, :, 0:2], delta[:, :, 2:3]], dim=2)
    assert torch.allclose(y, expected)
    assert not torch.allclose(dec_zero, dec_state)
    last = [module for module in model.head.modules() if isinstance(module, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        anchored = model(x)
    assert torch.allclose(anchored[:, :, 0], x[:, -1, 4].unsqueeze(1))
    assert torch.allclose(anchored[:, :, 1], x[:, -1, 5].unsqueeze(1))

    pinn = pinn_vfm(10, Logger(frequency=100), var=[0.9, 0.9], pinn_mode="on", inputs=6, device="cpu")
    pinn.u_model.eval()
    u = _safe_exogenous(8, 1)
    target = torch.zeros(8, 1, 2)
    target[:, :, 0] = 0.4
    target[:, :, 1] = 0.35
    parts = pinn.loss_on_batch(target, x, u)
    parts["total"].backward()
    lstm_hit = False
    for name, param in pinn.u_model.named_parameters():
        if param.grad is None:
            continue
        if "encoder" in name or "decoder" in name:
            if torch.isfinite(param.grad).all() and float(param.grad.abs().sum()) > 0:
                lstm_hit = True
    assert lstm_hit
    assert pinn.rho.grad is not None and torch.isfinite(pinn.rho.grad).all() and float(pinn.rho.grad.abs()) > 0
    assert pinn.PI.grad is not None and torch.isfinite(pinn.PI.grad).all() and float(pinn.PI.grad.abs()) > 0


def test_dropout_train_and_eval():
    torch.manual_seed(1)
    model = EncoderDecoderLSTM()
    x = torch.randn(8, 20, 6)
    model.train()
    outs = [model(x).detach().clone() for _ in range(6)]
    assert any(not torch.allclose(outs[0], other) for other in outs[1:])
    model.eval()
    a = model(x)
    b = model(x)
    assert torch.allclose(a, b)


def test_dydt_stencil_including_ends():
    values = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0], dtype=torch.float32)
    y = values.view(-1, 1, 1).expand(-1, 1, 3).contiguous()
    d = dydt(y, ts=1)
    expected = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=torch.float32)
    assert d.shape == (8, 3)
    assert torch.allclose(d[:, 0], expected)
    assert torch.allclose(d[:, 1], expected)
    assert torch.allclose(d[:, 2], expected)
    ramp = torch.arange(8, dtype=torch.float32).view(-1, 1, 1).expand(-1, 1, 3).contiguous()
    assert torch.allclose(dydt(ramp, ts=1), torch.ones(8, 3))
    try:
        dydt(ramp[:5], ts=1)
    except Exception as exc:
        assert "at least 6" in str(exc)
    else:
        raise AssertionError("short batch should fail")


def _ed_bcs_numpy(x, u, rho_hat, pi_hat, parameters, ts=1.0):
    from data.param import (
        g, Cc, A1, A2, D1, D2, hw, L1, L2, V1, V2,
        f0, b1, b2, M, mu, CH, Cq,
    )
    x = np.asarray(x, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    y = x[:, 0, :]
    n = y.shape[0]
    ddy = np.empty_like(y)
    ddy[0] = (np.array([-3.0, 4.0, -1.0], dtype=np.float32) @ y[0:3]) / (2.0 * ts)
    ddy[-1] = (np.array([1.0, -4.0, 3.0], dtype=np.float32) @ y[-3:]) / (2.0 * ts)
    ddy[1:-1] = (y[2:n] - y[0:n - 2]) / (2.0 * ts)
    p = parameters
    pbh = x[:, 0, 0:1] * np.float32(p.pbc) + np.float32(p.pbmin)
    pwh = x[:, 0, 1:2] * np.float32(p.pwc) + np.float32(p.pwmin)
    q = x[:, 0, 2:] * np.float32(p.qc) + np.float32(p.qmin)
    fq = u[:, 0, :1] * np.float32(60.0)
    zc = u[:, 0, 1:2] * np.float32(100.0)
    pm = u[:, 0, 2:3] * np.float32(p.pm_c) + np.float32(p.pm0)
    pr = u[:, 0, 3:] * np.float32(p.prc) + np.float32(p.pr0)
    q0 = q / np.float32(Cq) * (np.float32(f0) / fq)
    H0 = np.float32(-1.2454e6) * q0 ** 2 + np.float32(7.4959e3) * q0 + np.float32(9.5970e2)
    H = np.float32(CH) * H0 * (fq / np.float32(f0)) ** 2
    rho_phys = np.float32(rho_hat) * np.float32(RHO_SCALE)
    F1 = np.float32(0.158) * ((rho_phys * np.float32(L1) * q ** 2) / (np.float32(D1) * np.float32(A1) ** 2)) * (
        np.float32(mu) / (rho_phys * np.float32(D1) * q)
    ) ** np.float32(0.25)
    F2 = np.float32(0.158) * ((rho_phys * np.float32(L2) * q ** 2) / (np.float32(D2) * np.float32(A2) ** 2)) * (
        np.float32(mu) / (rho_phys * np.float32(D2) * q)
    ) ** np.float32(0.25)
    qr = np.float32(pi_hat) * np.float32(PI_SCALE) * (pr - pbh)
    qch = (zc / np.float32(100.0)) * np.float32(Cc) * np.sqrt(np.abs(pwh - pm))
    qch_n = (qch - np.float32(p.qch_lim[0])) / np.float32(p.qcc)
    F1_n = (F1 - np.float32(p.F1lim[0])) / np.float32(p.F1c)
    F2_n = (F2 - np.float32(p.F2lim[0])) / np.float32(p.F2c)
    H_n = (H - np.float32(p.H_lim[0])) / np.float32(p.Hc)
    dy1 = -(1.0 / np.float32(p.pbc)) * np.float32(b1) / np.float32(V1) * (qr - q)
    dy2 = -(1.0 / np.float32(p.pwc)) * np.float32(b2) / np.float32(V2) * (
        q - (np.float32(p.qcc) * qch_n + np.float32(p.qch_lim[0]))
    )
    dy3 = -(1.0 / (np.float32(p.qc) * np.float32(M))) * (
        pbh
        - pwh
        - rho_phys * np.float32(g) * np.float32(hw)
        - (np.float32(p.F1c) * F1_n + np.float32(p.F1lim[0]))
        - (np.float32(p.F2c) * F2_n + np.float32(p.F2lim[0]))
        + rho_phys * np.float32(g) * (H_n * np.float32(p.Hc) + np.float32(p.H_lim[0]))
    )
    r1 = np.mean(np.square(ddy[:, 0:1] + dy1))
    r2 = np.mean(np.square(ddy[:, 1:2] + dy2))
    r3 = np.mean(np.square(ddy[:, 2:] + dy3))
    return np.array([r1, r2, r3], dtype=np.float64)


def test_ed_bcs_matches_numpy_and_grads():
    torch.manual_seed(0)
    pinn = pinn_vfm(10, Logger(frequency=100), var=[0.9, 0.9], pinn_mode="on", inputs=6, device="cpu")
    pinn.u_model.eval()
    x = torch.zeros(8, 20, 6)
    x[:, :, 0] = 0.8
    x[:, :, 1] = 0.5
    x[:, :, 2] = 0.4
    x[:, :, 3] = 0.5
    x[:, :, 4] = 0.45
    x[:, :, 5] = 0.35
    with torch.no_grad():
        last = [module for module in pinn.u_model.head.modules() if isinstance(module, torch.nn.Linear)][-1]
        last.weight.mul_(0.05)
        last.bias.copy_(torch.tensor([0.0, 0.0, 0.55]))
    u = _safe_exogenous(8, 1)
    y_pred = pinn.u_model(x)
    r1, r2, r3 = ed_bcs(y_pred, u, pinn.rho, pinn.PI, pinn.ode_parameters, ts=1.0)
    ref = _ed_bcs_numpy(
        y_pred.detach().cpu().numpy(),
        u.numpy(),
        float(pinn.rho.detach()),
        float(pinn.PI.detach()),
        pinn.ode_parameters,
    )
    got = torch.stack([r1, r2, r3]).detach().cpu().numpy()
    np.testing.assert_allclose(got, ref, atol=1e-5)
    (r1 + r2 + r3).backward()
    assert torch.isfinite(pinn.rho.grad) and float(pinn.rho.grad.abs()) > 0
    assert torch.isfinite(pinn.PI.grad) and float(pinn.PI.grad.abs()) > 0
    assert any(
        param.grad is not None and torch.isfinite(param.grad).all() and float(param.grad.abs().sum()) > 0
        for name, param in pinn.u_model.named_parameters()
        if "encoder" in name or "decoder" in name
    )


def test_pinn_mode_off_and_data_loss_ignores_q():
    torch.manual_seed(0)
    off = pinn_vfm(10, Logger(frequency=100), var=[0.9, 0.9], pinn_mode="off", inputs=6, device="cpu")
    assert off.pinn_mode == PINN_MODES["off"]
    assert float(off.rho) == 1.0
    assert float(off.PI) == 1.0
    assert not off.rho.requires_grad
    assert not off.PI.requires_grad
    opt_ids = {id(p) for group in off.optimizer.param_groups for p in group["params"]}
    assert id(off.rho) not in opt_ids
    assert id(off.PI) not in opt_ids
    x, y, u = _ordered_rows(8)
    rho_before = float(off.rho)
    pi_before = float(off.PI)
    before = {name: param.detach().clone() for name, param in off.u_model.named_parameters()}
    y_pred = off.u_model(x)
    loss_obs, r1, r2, r3, _ = off.GetLoss(y, y_pred, u)
    assert float(r1) == 0.0 and float(r2) == 0.0 and float(r3) == 0.0
    loss_obs.backward()
    off.optimizer.step()
    assert float(off.rho) == rho_before
    assert float(off.PI) == pi_before
    assert any(not torch.allclose(before[name], param.detach()) for name, param in off.u_model.named_parameters())

    on = pinn_vfm(10, Logger(frequency=100), var=[0.9, 0.9], pinn_mode="on", inputs=6, device="cpu")
    on.u_model.eval()
    y_pred = torch.randn(8, 1, 3)
    u = _safe_exogenous(8, 1)
    target = torch.randn(8, 1, 3)
    changed = target.clone()
    changed[:, :, 2] = changed[:, :, 2] + 5
    a, *_ = on.GetLoss(target, y_pred, u)
    b, *_ = on.GetLoss(changed, y_pred, u)
    assert torch.allclose(a, b)


def test_adam_one_epoch_keeps_order():
    torch.manual_seed(0)
    x, y, u = _ordered_rows(16)
    x_before = x.clone()
    loader = OrderedBatchLoader(x, y, u, batch_size=8)
    batches = list(loader)
    assert len(batches) == 2
    assert torch.equal(batches[0][0], x[:8])
    assert torch.equal(batches[1][0], x[8:16])
    pinn = pinn_vfm(10, Logger(frequency=1), var=[0.9, 0.9], pinn_mode="on", inputs=6, device="cpu", lr=0.01)
    before = {name: param.detach().clone() for name, param in pinn.u_model.named_parameters()}
    test_x = x[:8].clone()
    test_y = torch.zeros(8, 1, 3)
    test_y[:, :, 0:2] = y[:8]
    loss_history, trainstate, _var = pinn.fit([loader, test_x, test_y], tf_epochs=1)
    assert np.isfinite(trainstate.loss_train)
    assert np.isfinite(trainstate.loss_test)
    assert any(not torch.allclose(before[name], param.detach()) for name, param in pinn.u_model.named_parameters())
    assert torch.equal(x, x_before)
    again = list(loader)
    assert torch.equal(again[0][0], x_before[:8])
    assert torch.equal(again[1][0], x_before[8:16])
    assert loss_history.steps[0] == 0


def test_lbfgs_two_iterations():
    torch.manual_seed(0)
    x, y, u = _ordered_rows(16)
    pinn = pinn_vfm(10, Logger(frequency=10), var=[0.9, 0.9], pinn_mode="on", inputs=6, device="cpu")
    before = {name: param.detach().clone() for name, param in pinn.u_model.named_parameters()}
    test_y = torch.zeros(8, 1, 3)
    test_y[:, :, 0:2] = y[:8]
    nt = Struct()
    nt.maxIter = 2
    nt.nCorrection = 50
    nt.tol = 1e-5
    nt.tolfun = 1e-5
    _loss, trainstate, _var = pinn.fit_LBFGS([x, y, u, x[:8], test_y], nt)
    assert np.isfinite(trainstate.loss_train)
    assert np.isfinite(trainstate.loss_test)
    assert np.isfinite(trainstate.loss_train_x1)
    assert np.isfinite(trainstate.loss_train_x2)
    assert np.isfinite(trainstate.loss_train_x3)
    assert any(not torch.allclose(before[name], param.detach()) for name, param in pinn.u_model.named_parameters())


def test_save_load_restores_state(tmp_path):
    torch.manual_seed(0)
    pinn = pinn_vfm(10, Logger(frequency=100), var=[0.9, 0.9], pinn_mode="on", inputs=6, device="cpu")
    with torch.no_grad():
        pinn.rho.fill_(0.77)
        pinn.PI.fill_(0.66)
        last = [module for module in pinn.u_model.head.modules() if isinstance(module, torch.nn.Linear)][-1]
        last.bias.add_(0.1)
    path = tmp_path / "model.pt"
    pinn.lamb_l1 = 2.0
    pinn.save_checkpoint(str(path))
    other = pinn_vfm(10, Logger(frequency=100), var=[0.1, 0.1], pinn_mode="on", inputs=6, device="cpu")
    ckpt = other.load_checkpoint(str(path))
    for key, value in pinn.u_model.state_dict().items():
        assert torch.equal(value.cpu(), other.u_model.state_dict()[key].cpu())
    assert torch.equal(pinn.rho.detach().cpu(), other.rho.detach().cpu())
    assert torch.equal(pinn.PI.detach().cpu(), other.PI.detach().cpu())
    assert other.rho.detach().cpu().item() == ckpt["rho"]
    assert other.PI.detach().cpu().item() == ckpt["PI"]
    assert ckpt["width"] == 10
    assert ckpt["pinn_mode"] == "on"


def test_free_run_does_not_feed_q_back():
    torch.manual_seed(0)
    model = EncoderDecoderLSTM(pinn_mode="on", var=(0.9, 0.9))
    model.eval()
    un = torch.zeros(40, 4)
    for i in range(40):
        un[i, 0] = 0.01 * i
        un[i, 1] = 0.5
        un[i, 2] = 0.2
        un[i, 3] = 0.4
    preds, trace = free_run(model, un, steps=5, n_steps_in=20, trace=True)
    assert preds.shape == (6, 1, 3)
    assert torch.isfinite(preds).all()
    assert len(trace) == 5
    for window in trace:
        prev = window["prev_y"][0, 0]
        assert window["Xi"].shape == (1, 20, 2)
        assert torch.allclose(window["Xi"][0, -1, 0], prev[0])
        assert torch.allclose(window["Xi"][0, -1, 1], prev[1])
    assert torch.allclose(trace[0]["Ui"][0, -1], un[20])
    assert torch.allclose(trace[1]["Ui"][0, -1], un[21])
    assert torch.allclose(trace[4]["Ui"][0, -1], un[24])
    # The supplied initial condition is what the first window repeats.
    parameters = Parameters()
    ui = torch.as_tensor(parameters.normalizar_u(UI_PHYS), dtype=torch.float32)
    assert torch.allclose(trace[0]["Ui"][0, 0], ui[0])
    xi = torch.as_tensor(parameters.normalizar_x(XI_PHYS), dtype=torch.float32)
    assert torch.allclose(trace[0]["Xi"][0, 0], xi[0])
    jit_preds = free_run(model, un, steps=5, n_steps_in=20, jit_trace=True)
    assert jit_preds.shape == preds.shape
    assert torch.allclose(jit_preds, preds, atol=1e-5, rtol=1e-5)
