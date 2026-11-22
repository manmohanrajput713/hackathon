"""PINC trainer: scaled skip network, Adam, then L-BFGS.

The physics residual is in scaled state units. Time is the only input that
the autograd derivative differentiates, with create_graph=True. y_min, y_max,
and T are constants. A network prediction that leaves the explicit model is
not sent through the ODE (that call would be undefined). Those rows
contribute (y_s - 0.5)^2 instead, which stays finite. Once every row is
feasible the loss is exactly λ_y MSE_y + λ_F MSE_F.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.Logger import Logger
from data.LossHistory import LossHistory
from data.TrainState import TrainState
from data.param import T_HORIZON, get_well
from data.skip_pinc import SkipPINC
from data.utils import (
    build_pinc,
    load_checkpoint,
    plot_loss,
    save_checkpoint,
    scale_state,
    set_seed,
    store_pickle,
)
from data.well_ode import InfeasibleState, feasible_mask, ode_torch

LAMBDA_Y = 1.0
LAMBDA_F = 1.0


def _const(value, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=like.dtype, device=like.device)


def pack_features(t_s: torch.Tensor, y0_s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    t_col = t_s.reshape(-1, 1)
    return torch.cat([t_col, y0_s, u], dim=1)


def time_derivative(y_s: torch.Tensor, t_s: torch.Tensor) -> torch.Tensor:
    """dy_s/dt_s for each state. t_s is (B, 1) and must require grad."""
    columns = []
    ones = torch.ones(y_s.shape[0], dtype=y_s.dtype, device=y_s.device)
    for i in range(y_s.shape[1]):
        grad = torch.autograd.grad(
            y_s[:, i],
            t_s,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        columns.append(grad)
    return torch.cat(columns, dim=1)


def physics_residual(
    net: SkipPINC,
    t_s: torch.Tensor,
    y0_s: torch.Tensor,
    u: torch.Tensor,
    y_min,
    y_max,
    T: float,
    well,
    *,
    mask_infeasible: bool = False,
):
    """Scaled residual F, shape (B, 3).

    With mask_infeasible=False an undefined state raises InfeasibleState.
    That is the path the unit test uses. Training sets the flag so a bad
    prediction cannot turn the loss into NaN.
    """
    if t_s.ndim == 1:
        t_s = t_s.reshape(-1, 1)
    if not t_s.requires_grad:
        t_s = t_s.detach().requires_grad_(True)
    y_s = net(pack_features(t_s, y0_s, u))
    dy_s_dt_s = time_derivative(y_s, t_s)
    span = _const(np.asarray(y_max) - np.asarray(y_min), y_s).reshape(1, 3)
    lo = _const(y_min, y_s).reshape(1, 3)
    y_phys = y_s * span + lo
    scale = float(T) / span
    if not mask_infeasible:
        f_phys = ode_torch(y_phys, u, well, friction="cubic")
        return dy_s_dt_s - f_phys * scale, y_s, torch.ones(y_s.shape[0], dtype=torch.bool)
    with torch.no_grad():
        ok_np = feasible_mask(
            y_phys.detach().cpu().numpy(),
            u.detach().cpu().numpy(),
            well,
            friction="cubic",
        )
    ok = torch.as_tensor(ok_np, device=y_s.device)
    penalty = y_s - 0.5
    if not bool(ok.any()):
        return penalty, y_s, ok
    if bool(ok.all()):
        try:
            f_phys = ode_torch(y_phys, u, well, friction="cubic")
        except InfeasibleState:
            return penalty, y_s, torch.zeros_like(ok)
        return dy_s_dt_s - f_phys * scale, y_s, ok
    try:
        f_ok = ode_torch(y_phys[ok], u[ok], well, friction="cubic")
    except InfeasibleState:
        return penalty, y_s, torch.zeros_like(ok)
    index = ok.nonzero(as_tuple=False).expand(-1, y_s.shape[1])
    f_all = torch.zeros_like(y_s).scatter(0, index, f_ok)
    mask = ok.to(dtype=y_s.dtype).reshape(-1, 1)
    residual = mask * (dy_s_dt_s - f_all * scale) + (1.0 - mask) * penalty
    return residual, y_s, ok


def batch_losses(
    net: SkipPINC,
    y_ic_s,
    u_ic,
    t_f,
    y_f_s,
    u_f,
    y_min,
    y_max,
    T: float,
    well,
    lambda_y: float = LAMBDA_Y,
    lambda_f: float = LAMBDA_F,
    mask_infeasible: bool = True,
):
    t0 = torch.zeros(y_ic_s.shape[0], 1, dtype=y_ic_s.dtype, device=y_ic_s.device)
    y_hat0 = net(pack_features(t0, y_ic_s, u_ic))
    mse_y = (y_hat0 - y_ic_s).pow(2).mean()
    residual, _y_s, ok = physics_residual(
        net, t_f, y_f_s, u_f, y_min, y_max, T, well, mask_infeasible=mask_infeasible
    )
    mse_f = residual.pow(2).mean()
    loss = float(lambda_y) * mse_y + float(lambda_f) * mse_f
    fraction = float(ok.detach().to(dtype=torch.float64).mean()) if ok.numel() else 1.0
    return loss, mse_y, mse_f, fraction


def validation_mse(net, y0_s, u, y_rk_s) -> float:
    net.eval()
    with torch.no_grad():
        t = torch.ones(y0_s.shape[0], 1, dtype=y0_s.dtype, device=y0_s.device)
        y_hat = net(pack_features(t, y0_s, u))
        mse = (y_hat - y_rk_s).pow(2).mean()
    net.train()
    return float(mse)


def _tensors(batches, y_min, y_max, device):
    def s(y):
        return torch.as_tensor(scale_state(y, y_min, y_max), dtype=torch.float64, device=device)

    def t(value):
        return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64, device=device)

    return {
        "y_ic_s": s(batches["y_ic"]),
        "u_ic": t(batches["u_ic"]),
        "t_f": t(batches["t_f"]),
        "y_f_s": s(batches["y_f"]),
        "u_f": t(batches["u_f"]),
        "y_val_s": s(batches["y_val"]),
        "u_val": t(batches["u_val"]),
        "y_rk_s": s(batches["y_rk"]),
    }


def _checkpoint(model, y_min, y_max, well_id, lambda_y, lambda_f, extra=None) -> dict:
    payload = {
        "state_dict": model.state_dict(),
        "y_min": np.asarray(y_min, dtype=np.float64),
        "y_max": np.asarray(y_max, dtype=np.float64),
        "T": float(T_HORIZON),
        "well": int(well_id),
        "architecture": model.architecture(),
        "lambda_y": float(lambda_y),
        "lambda_F": float(lambda_f),
        "friction": "cubic",
        "kind": "pinc",
    }
    if extra:
        payload.update(extra)
    return payload


def _prepare_model(well_id, n_layers, n_neurons, skip, resume_path):
    if resume_path:
        ckpt = load_checkpoint(resume_path)
        model = build_pinc(ckpt["architecture"])
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt
    model = SkipPINC(n_layers=n_layers, n_neurons=n_neurons, skip=skip).double()
    return model, None


def train_adam(
    domain,
    batches,
    out_dir,
    *,
    well_id: int = 1,
    epochs: int = 1000,
    lr: float = 7e-3,
    seed: int = 0,
    n_layers: int = 6,
    n_neurons: int = 30,
    skip: bool = True,
    lambda_y: float = LAMBDA_Y,
    lambda_f: float = LAMBDA_F,
    val_every: int = 50,
    resume_path=None,
    log_every: int | None = None,
):
    set_seed(seed)
    device = torch.device("cpu")
    model, ckpt = _prepare_model(well_id, n_layers, n_neurons, skip, resume_path)
    model.to(device)
    y_min = np.asarray(domain["y_min"], dtype=np.float64)
    y_max = np.asarray(domain["y_max"], dtype=np.float64)
    if ckpt is not None:
        y_min = np.asarray(ckpt["y_min"], dtype=np.float64)
        y_max = np.asarray(ckpt["y_max"], dtype=np.float64)
        lambda_y = float(ckpt.get("lambda_y", lambda_y))
        lambda_f = float(ckpt.get("lambda_F", lambda_f))
    well = get_well(well_id)
    data = _tensors(batches, y_min, y_max, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    history = LossHistory()
    state = TrainState()
    logger = Logger(frequency=1 if log_every is None and epochs <= 20 else (log_every or 50))
    logger.log_start(model, lr, "Adam")
    model.train()
    for epoch in range(1, int(epochs) + 1):
        opt.zero_grad()
        loss, mse_y, mse_f, fraction = batch_losses(
            model,
            data["y_ic_s"],
            data["u_ic"],
            data["t_f"],
            data["y_f_s"],
            data["u_f"],
            y_min,
            y_max,
            T_HORIZON,
            well,
            lambda_y,
            lambda_f,
            mask_infeasible=True,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite Adam loss at epoch {epoch}")
        loss.backward()
        opt.step()
        do_val = epoch % int(val_every) == 0 or epoch == int(epochs)
        mse_val = None
        if do_val:
            mse_val = validation_mse(model, data["y_val_s"], data["u_val"], data["y_rk_s"])
        history.append(epoch, loss.detach(), mse_y.detach(), mse_f.detach(), mse_val, fraction)
        state.update(epoch, loss.detach(), mse_y.detach(), mse_f.detach(), mse_val, fraction)
        logger.log_epoch(
            epoch,
            float(loss.detach()),
            float(mse_y.detach()),
            float(mse_f.detach()),
            mse_val,
            fraction,
            force=do_val,
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint(
        model,
        y_min,
        y_max,
        well_id,
        lambda_y,
        lambda_f,
        extra={"seed": int(seed), "optimizer": "adam", "lr": float(lr)},
    )
    save_checkpoint(out / "model.pt", payload)
    store_pickle(out / "Loss.pk", history)
    store_pickle(out / "trainstate.pk", state)
    plot_loss(history, out / "loss.png")
    logger.log_end(epochs, state.loss_train)
    _report_balance(state.loss_y, state.loss_f, lambda_y, lambda_f)
    return model, history, state


def train_lbfgs(
    domain,
    batches,
    out_dir,
    *,
    init_path,
    well_id: int = 1,
    iterations: int = 1000,
    seed: int = 0,
    lambda_y: float | None = None,
    lambda_f: float | None = None,
    log_every: int = 1,
):
    set_seed(seed)
    device = torch.device("cpu")
    ckpt = load_checkpoint(init_path)
    model = build_pinc(ckpt["architecture"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    y_min = np.asarray(ckpt["y_min"], dtype=np.float64)
    y_max = np.asarray(ckpt["y_max"], dtype=np.float64)
    lambda_y = float(ckpt.get("lambda_y", LAMBDA_Y) if lambda_y is None else lambda_y)
    lambda_f = float(ckpt.get("lambda_F", LAMBDA_F) if lambda_f is None else lambda_f)
    well = get_well(int(ckpt.get("well", well_id)))
    data = _tensors(batches, y_min, y_max, device)
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    history = LossHistory()
    state = TrainState()
    logger = Logger(frequency=max(1, int(log_every)))
    logger.log_start(model, 1.0, "L-BFGS")
    model.train()
    ran = 0

    def closure():
        opt.zero_grad()
        loss, mse_y, mse_f, fraction = batch_losses(
            model,
            data["y_ic_s"],
            data["u_ic"],
            data["t_f"],
            data["y_f_s"],
            data["u_f"],
            y_min,
            y_max,
            T_HORIZON,
            well,
            lambda_y,
            lambda_f,
            mask_infeasible=True,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite L-BFGS loss")
        loss.backward()
        closure.last = (loss.detach(), mse_y.detach(), mse_f.detach(), fraction)
        return loss

    closure.last = None
    for step in range(1, int(iterations) + 1):
        try:
            opt.step(closure)
        except RuntimeError as exc:
            print(f"L-BFGS stopped at iteration {step}: {exc}")
            break
        ran = step
        loss_v, mse_y, mse_f, fraction = closure.last
        mse_val = validation_mse(model, data["y_val_s"], data["u_val"], data["y_rk_s"])
        history.append(step, loss_v, mse_y, mse_f, mse_val, fraction)
        state.update(step, loss_v, mse_y, mse_f, mse_val, fraction)
        logger.log_epoch(step, float(loss_v), float(mse_y), float(mse_f), mse_val, fraction, force=True)
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().pow(2).sum())
        if grad_norm ** 0.5 <= 1e-9:
            print(f"L-BFGS gradient tolerance met at iteration {step}")
            break
    if state.loss_train is None:
        # The line search rejected every step. Keep a finite record of the
        # objective at the loaded weights so the checkpoint is still usable.
        with torch.no_grad():
            # batch_losses builds a graph; evaluate it once more for the log.
            pass
        loss, mse_y, mse_f, fraction = batch_losses(
            model,
            data["y_ic_s"],
            data["u_ic"],
            data["t_f"],
            data["y_f_s"],
            data["u_f"],
            y_min,
            y_max,
            T_HORIZON,
            well,
            lambda_y,
            lambda_f,
        )
        loss_value = float(loss.detach())
        history.append(0, loss_value, float(mse_y), float(mse_f), None, fraction)
        state.update(0, loss_value, float(mse_y), float(mse_f), None, fraction)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = _checkpoint(
        model,
        y_min,
        y_max,
        well.well_id,
        lambda_y,
        lambda_f,
        extra={"seed": int(seed), "optimizer": "lbfgs", "lbfgs_iterations_ran": int(ran)},
    )
    save_checkpoint(out / "model.pt", payload)
    store_pickle(out / "Loss.pk", history)
    store_pickle(out / "trainstate.pk", state)
    plot_loss(history, out / "loss.png")
    logger.log_end(ran, state.loss_train)
    _report_balance(state.loss_y, state.loss_f, lambda_y, lambda_f)
    return model, history, state


def _report_balance(mse_y, mse_f, lambda_y, lambda_f):
    eps = 1e-30
    ratio = max(mse_y, eps) / max(mse_f, eps)
    print(
        f"loss balance mse_y={mse_y:.3e} mse_F={mse_f:.3e} "
        f"ratio={ratio:.3e} lambda_y={lambda_y:g} lambda_F={lambda_f:g}"
    )
    if ratio > 100.0 or ratio < 0.01:
        print(
            "one loss term is at least 100x the other. "
            "lambda_y and lambda_F were left at the values stored in the checkpoint."
        )


def train_algebraic(
    domain,
    out_dir,
    *,
    well_id: int = 1,
    n_train: int = 50000,
    n_val: int = 10000,
    iterations: int = 5000,
    seed: int = 0,
    n_layers: int = 4,
    n_neurons: int = 30,
):
    """L-BFGS on standardized algebraic targets. No physics loss."""
    from data.algebraic_nn import AlgebraicNet, pack_algebraic_input
    from data.utils import build_algebraic
    from data.well_ode import algebraics_numpy

    set_seed(seed)
    well = get_well(well_id)
    rng = np.random.default_rng(seed)
    pool_y = np.asarray(domain["y_pool"], dtype=np.float64)
    pool_u = np.asarray(domain["u_pool"], dtype=np.float64)
    n_total = int(n_train) + int(n_val)
    index = rng.integers(0, pool_y.shape[0], size=n_total)
    y = pool_y[index]
    u = pool_u[index]
    targets = algebraics_numpy(y, u, well, friction="haaland")
    y_min = np.asarray(domain["y_min"], dtype=np.float64)
    y_max = np.asarray(domain["y_max"], dtype=np.float64)
    inputs = pack_algebraic_input(y, u, y_min, y_max)
    z_train = targets[: int(n_train)]
    z_mean = z_train.mean(axis=0)
    z_std = np.maximum(z_train.std(axis=0), 1e-8)
    z_s = (targets - z_mean) / z_std
    x_t = torch.as_tensor(inputs[: int(n_train)], dtype=torch.float64)
    z_t = torch.as_tensor(z_s[: int(n_train)], dtype=torch.float64)
    x_v = torch.as_tensor(inputs[int(n_train) :], dtype=torch.float64)
    z_v = torch.as_tensor(z_s[int(n_train) :], dtype=torch.float64)
    model = AlgebraicNet(n_layers=n_layers, n_neurons=n_neurons).double()
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    logger = Logger(frequency=max(1, min(50, int(iterations))))
    logger.log_start(model, 1.0, "L-BFGS algebraic")
    model.train()
    ran = 0

    def closure():
        opt.zero_grad()
        pred = model(x_t)
        loss = (pred - z_t).pow(2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite algebraic loss")
        loss.backward()
        closure.last = loss.detach()
        return loss

    closure.last = None
    for step in range(1, int(iterations) + 1):
        try:
            opt.step(closure)
        except RuntimeError as exc:
            print(f"algebraic L-BFGS stopped at iteration {step}: {exc}")
            break
        ran = step
        if step == 1 or step % logger.frequency == 0 or step == int(iterations):
            with torch.no_grad():
                val = float((model(x_v) - z_v).pow(2).mean()) if x_v.shape[0] else float("nan")
            logger.log_epoch(step, float(closure.last), float(closure.last), 0.0, val, 1.0, force=True)
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += float(param.grad.detach().pow(2).sum())
        if grad_norm ** 0.5 <= 1e-9:
            print(f"algebraic L-BFGS gradient tolerance met at iteration {step}")
            break
    if closure.last is None:
        with torch.no_grad():
            closure.last = (model(x_t) - z_t).pow(2).mean().detach()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "y_min": y_min,
        "y_max": y_max,
        "T": float(T_HORIZON),
        "well": int(well_id),
        "architecture": model.architecture(),
        "lambda_y": 1.0,
        "lambda_F": 0.0,
        "friction": "haaland",
        "kind": "algebraic",
        "z_mean": z_mean,
        "z_std": z_std,
        "seed": int(seed),
        "lbfgs_iterations_ran": int(ran),
    }
    save_checkpoint(out / "model.pt", payload)
    logger.log_end(ran, float(closure.last))
    return model, payload
