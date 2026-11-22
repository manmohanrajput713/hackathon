import numpy as np
import torch

from data.parameters import Parameters
from data.utils import plot_states_BCS

# Initial measured state and exogenous input used by the free-run demo.
XI_PHYS = np.array([[7.92750470e6, 2.90972936e6]], dtype=np.float64)
UI_PHYS = np.array([[49.5914975, 50.0190605, 1.99149787e6, 1.18634314e7]], dtype=np.float64)


def _model_device(model):
    return next(model.parameters()).device


def _pack_input(buf, Ui, Xi):
    n_u = Ui.shape[-1]
    buf[:, :, :n_u].copy_(Ui)
    buf[:, :, n_u:].copy_(Xi)
    return buf


@torch.inference_mode()
def _free_run_loop(predict_fn, un_t, Xi, Ui, steps, n_steps_in, capture):
    """Roll Xi and Ui with ping-pong buffers. No torch.cat in the loop."""
    Xi_buf = Xi.clone()
    Ui_buf = Ui.clone()
    Xi_spare = torch.empty_like(Xi_buf)
    Ui_spare = torch.empty_like(Ui_buf)
    packed = torch.empty(
        Xi_buf.shape[0],
        Xi_buf.shape[1],
        Ui_buf.shape[-1] + Xi_buf.shape[-1],
        device=Xi_buf.device,
        dtype=Xi_buf.dtype,
    )
    y0 = predict_fn(_pack_input(packed, Ui_buf, Xi_buf))
    predict = torch.empty((int(steps) + 1,) + tuple(y0.shape[1:]), device=y0.device, dtype=y0.dtype)
    predict[0:1].copy_(y0)
    windows = []
    for i in range(int(steps)):
        prev = y0
        Xi_spare[:, :-1, :].copy_(Xi_buf[:, 1:, :])
        Xi_spare[:, -1:, :].copy_(y0[:, -1:, :2])
        Ui_spare[:, :-1, :].copy_(Ui_buf[:, 1:, :])
        Ui_spare[:, -1:, :].copy_(un_t[n_steps_in + i].reshape(1, 1, -1))
        Xi_buf, Xi_spare = Xi_spare, Xi_buf
        Ui_buf, Ui_spare = Ui_spare, Ui_buf
        if capture:
            windows.append({
                "Xi": Xi_buf.clone(),
                "Ui": Ui_buf.clone(),
                "prev_y": prev.clone(),
            })
        y0 = predict_fn(_pack_input(packed, Ui_buf, Xi_buf))
        predict[i + 1:i + 2].copy_(y0)
    return predict, windows


def free_run(model, un, steps=800, n_steps_in=20, parameters=None, xi=None, ui=None, trace=False, jit_trace=False):
    """Roll the network forward.

    Only predicted pbh and pwh are written back into the state window.
    q is not fed back. Future exogenous rows come from ``un``, starting
    at index ``n_steps_in``.

    ``trace`` returns the state and input windows after each shift.
    ``jit_trace`` compiles one forward with ``torch.jit.trace`` and
    reuses it for the rollout.
    """
    parameters = parameters if parameters is not None else Parameters()
    if xi is None:
        xi = XI_PHYS
    if ui is None:
        ui = UI_PHYS
    xi_n = np.asarray(parameters.normalizar_x(np.asarray(xi, dtype=float)), dtype=np.float32)
    ui_n = np.asarray(parameters.normalizar_u(np.asarray(ui, dtype=float)), dtype=np.float32)
    if not torch.is_tensor(un):
        un_t = torch.as_tensor(np.asarray(un), dtype=torch.float32)
    else:
        un_t = un.to(dtype=torch.float32)
    device = _model_device(model)
    un_t = un_t.to(device)
    if n_steps_in + int(steps) > int(un_t.shape[0]):
        raise ValueError(
            f"Need {n_steps_in + int(steps)} exogenous rows and un has {int(un_t.shape[0])}."
        )
    Xi = torch.as_tensor(np.repeat(xi_n[:, None, :], n_steps_in, axis=1), dtype=torch.float32, device=device)
    Ui = torch.as_tensor(np.repeat(ui_n[:, None, :], n_steps_in, axis=1), dtype=torch.float32, device=device)
    training = model.training
    model.eval()
    try:
        predict_fn = model
        if jit_trace:
            example = torch.empty(
                Xi.shape[0],
                Xi.shape[1],
                Ui.shape[-1] + Xi.shape[-1],
                device=device,
                dtype=torch.float32,
            )
            predict_fn = torch.jit.trace(model, _pack_input(example, Ui, Xi))
        predict, windows = _free_run_loop(predict_fn, un_t, Xi, Ui, steps, n_steps_in, trace)
    finally:
        model.train(training)
    if trace:
        return predict, windows
    return predict


def save_forecast_plot(pred, path, parameters=None):
    parameters = parameters if parameters is not None else Parameters()
    yp = np.asarray(pred.detach().cpu() if isinstance(pred, torch.Tensor) else pred)
    if yp.ndim == 3:
        yp = yp[:, 0, :]
    phys = yp * np.asarray(parameters.xc, dtype=float) + np.asarray(parameters.x0, dtype=float)
    fig = plot_states_BCS(phys, np.arange(len(phys)))
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path
