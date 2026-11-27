import os
import time

import numpy as np
import torch
from torch import nn

from data.LossHistory import LossHistory
from data.VarHistory import VarHistory
from data.TrainState import TrainState
from data.utils import dydt
from data.parameters import Parameters
from data.param import (
    g, Cc, A1, A2, D1, D2, hw, L1, L2, V1, V2,
    f0, b1, b2, M, mu, CH, Cq,
)

# Physical scales for the normalized trainable parameters.
RHO_SCALE = 950.0
PI_SCALE = 2.32e-9

PINN_MODES = {
    "off": 0,
    "on": 1,
    "all": 2,
    "loss2": 3,
}


def _init_lstm(lstm):
    """Glorot input weights, orthogonal recurrent weights, forget-gate bias 1."""
    hidden = lstm.hidden_size
    for name, param in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
            if name.startswith("bias_ih"):
                param.data[hidden:2 * hidden].fill_(1.0)


def _init_encoder_decoder(encoder, decoder, head):
    _init_lstm(encoder)
    _init_lstm(decoder)
    linears = [module for module in head.modules() if isinstance(module, nn.Linear)]
    for module in linears:
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)
    if not linears:
        return
    # Pressure channels are deltas on the measured state. A large initial
    # delta drives q negative and makes the friction residual non-finite.
    # The last weight stays nonzero so the LSTM gets a gradient on step one.
    nn.init.xavier_uniform_(linears[-1].weight, gain=0.1)
    linears[-1].bias.data[2] = 0.5


def _mode_name(value):
    if isinstance(value, str):
        if value not in PINN_MODES:
            raise ValueError("Invalid arguments for pinn_mode")
        return value
    for name, number in PINN_MODES.items():
        if int(value) == number:
            return name
    raise ValueError("Invalid arguments for pinn_mode")


def ed_bcs(x, u, rho_hat, pi_hat, parameters, ts=1.0):
    """Squared BCS residuals. The batch axis is the time axis inside dydt."""
    ddy = dydt(x, ts)
    p = parameters
    pbh = x[:, 0, 0:1] * p.pbc + p.pbmin
    pwh = x[:, 0, 1:2] * p.pwc + p.pwmin
    q = x[:, 0, 2:] * p.qc + p.qmin
    fq = u[:, 0, :1] * 60.0
    zc = u[:, 0, 1:2] * 100.0
    pm = u[:, 0, 2:3] * p.pm_c + p.pm0
    pr = u[:, 0, 3:] * p.prc + p.pr0

    q0 = q / Cq * (f0 / fq)
    H0 = -1.2454e6 * q0 ** 2 + 7.4959e3 * q0 + 9.5970e2
    H = CH * H0 * (fq / f0) ** 2
    rho_phys = rho_hat * RHO_SCALE
    F1 = 0.158 * ((rho_phys * L1 * q ** 2) / (D1 * A1 ** 2)) * (mu / (rho_phys * D1 * q)) ** 0.25
    F2 = 0.158 * ((rho_phys * L2 * q ** 2) / (D2 * A2 ** 2)) * (mu / (rho_phys * D2 * q)) ** 0.25
    qr = pi_hat * PI_SCALE * (pr - pbh)
    qch = (zc / 100.0) * Cc * torch.sqrt(torch.abs(pwh - pm))

    qch_n = (qch - p.qch_lim[0]) / p.qcc
    F1_n = (F1 - p.F1lim[0]) / p.F1c
    F2_n = (F2 - p.F2lim[0]) / p.F2c
    H_n = (H - p.H_lim[0]) / p.Hc

    dy1 = -(1.0 / p.pbc) * b1 / V1 * (qr - q)
    dy2 = -(1.0 / p.pwc) * b2 / V2 * (q - (p.qcc * qch_n + p.qch_lim[0]))
    dy3 = -(1.0 / (p.qc * M)) * (
        pbh
        - pwh
        - rho_phys * g * hw
        - (p.F1c * F1_n + p.F1lim[0])
        - (p.F2c * F2_n + p.F2lim[0])
        + rho_phys * g * (H_n * p.Hc + p.H_lim[0])
    )
    loss_x1 = torch.mean(torch.square(ddy[:, 0:1] + dy1))
    loss_x2 = torch.mean(torch.square(ddy[:, 1:2] + dy2))
    loss_x3 = torch.mean(torch.square(ddy[:, 2:] + dy3))
    return loss_x1, loss_x2, loss_x3


class EncoderDecoderLSTM(nn.Module):
    """Encoder-decoder LSTM with a dense residual head.

    The decoder input is the encoder's last hidden vector, repeated
    n_steps_out times. The decoder hidden and cell state start at zero.
    Pbh and Pwh are the last measured normalized pressures plus a delta.
    q is the head's third output and is not residual.
    """

    def __init__(self, width=10, n_features=6, n_steps_in=20, n_steps_out=1, pinn_mode="on", var=(0.9, 0.9)):
        super().__init__()
        self.width = int(width)
        self.n_features = int(n_features)
        self.n_steps_in = int(n_steps_in)
        self.n_steps_out = int(n_steps_out)
        self.pinn_mode_name = _mode_name(pinn_mode)
        self.pinn_mode = PINN_MODES[self.pinn_mode_name]
        self.encoder = nn.LSTM(self.n_features, self.width, batch_first=True)
        self.decoder = nn.LSTM(self.width, self.width, batch_first=True)
        self.drop_enc = nn.Dropout(0.2)
        self.drop_dec = nn.Dropout(0.2)
        self.head = nn.Sequential(
            nn.Linear(self.width, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        _init_encoder_decoder(self.encoder, self.decoder, self.head)
        if self.pinn_mode == 0:
            rho0, pi0 = 1.0, 1.0
        else:
            rho0, pi0 = float(var[0]), float(var[1])
        rho = torch.tensor(rho0, dtype=torch.float32)
        pi = torch.tensor(pi0, dtype=torch.float32)
        if self.pinn_mode == 0:
            self.register_buffer("rho", rho)
            self.register_buffer("PI", pi)
        else:
            self.rho = nn.Parameter(rho)
            self.PI = nn.Parameter(pi)

    def forward(self, x):
        _, (h_n, _c_n) = self.encoder(x)
        last = self.drop_enc(h_n[-1])
        repeated = last.unsqueeze(1).repeat(1, self.n_steps_out, 1)
        decoded, _state = self.decoder(repeated)
        decoded = self.drop_dec(decoded)
        delta = self.head(decoded)
        # Columns 4 and 5 are the latest normalized Pbh and Pwh.
        # x_{t+1} = x_t + Delta x keeps long rollouts from drifting.
        anchor = x[:, -1, 4:6].unsqueeze(1)
        pressures = anchor + delta[:, :, 0:2]
        return torch.cat([pressures, delta[:, :, 2:3]], dim=2)

    def predict(self, x):
        training = self.training
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            elif isinstance(x, torch.Tensor):
                tensor = x.to(dtype=torch.float32)
            else:
                tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            tensor = tensor.to(next(self.parameters()).device)
            out = self.forward(tensor)
        self.train(training)
        return out.detach().cpu().numpy()


def load_module(path, device=None):
    if os.path.isdir(path):
        path = os.path.join(path, "model.pt")
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = EncoderDecoderLSTM(
        width=ckpt["width"],
        n_features=ckpt["n_features"],
        n_steps_in=ckpt["n_steps_in"],
        n_steps_out=ckpt["n_steps_out"],
        pinn_mode=ckpt["pinn_mode"],
        var=(ckpt["rho"], ckpt["PI"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    return model, ckpt


class pinn_vfm(object):
    def __init__(self, neurons=10, logger=None, var=None, pinn_mode="on", inputs=6,
                 n_steps_in=20, n_steps_out=1, parameters=None, lr=0.01, device=None):
        self.logger = logger
        self.ode_parameters = parameters if parameters is not None else Parameters()
        self.inputs = inputs
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.dtype = torch.float32
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if var is None:
            var = [0.9, 0.9]
        self.var0 = [float(var[0]), float(var[1])]
        self.pinn_mode_name = _mode_name(pinn_mode)
        self.pinn_mode = PINN_MODES[self.pinn_mode_name]
        self.u_model = EncoderDecoderLSTM(
            width=neurons,
            n_features=inputs,
            n_steps_in=n_steps_in,
            n_steps_out=n_steps_out,
            pinn_mode=self.pinn_mode_name,
            var=self.var0,
        ).to(self.device)
        self.rho = self.u_model.rho
        self.PI = self.u_model.PI

        self.alfa = 0.8
        self.lamb_bc = 1.0
        self.lamb_l1 = 1.0
        self.lamb_l2 = 1.0
        self.lamb_l3 = 1.0
        self.lr = float(lr)
        self.optimizer = torch.optim.Adam(self.training_parameters(), lr=self.lr)

        self.Font = 14
        self.losshistory = LossHistory()
        self.varhistory = VarHistory()
        self.train_state = TrainState()
        self.nsess = 0
        self.epoch = 0
        self.test_X = torch.zeros(1, n_steps_in, inputs, dtype=torch.float32, device=self.device)
        self.test_y = torch.zeros(1, n_steps_out, 2, dtype=torch.float32, device=self.device)
        if self.logger is not None:
            self.logger.log_train_start(self)

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def training_parameters(self):
        return [p for p in self.u_model.parameters() if p.requires_grad]

    def pinn_mode_set(self, value):
        name = _mode_name(value)
        self.pinn_mode_name = name
        self.pinn_mode = PINN_MODES[name]
        self.u_model.pinn_mode_name = name
        self.u_model.pinn_mode = self.pinn_mode

    def get_lamb_weights(self):
        return f"[{self.lamb_bc:4.3f},{self.lamb_l1:4.3f},{self.lamb_l2:4.3f},{self.lamb_l3:4.3f}]"

    def summary_model(self):
        n_param = sum(p.numel() for p in self.u_model.parameters())
        return f"{self.u_model}\nTrainable parameters: {n_param}"

    def snapshot_weights(self):
        return {k: v.detach().cpu().numpy().copy() for k, v in self.u_model.state_dict().items()}

    def ED_BCS(self, x, u):
        return ed_bcs(x, u, self.rho, self.PI, self.ode_parameters, ts=1.0)

    def GetLoss(self, y, y_pred, u):
        # Data loss uses the measured pressures only. q is not in the target term.
        loss_obs = torch.mean(torch.square(y[:, :, 0:2] - y_pred[:, :, 0:2]))
        zero = y_pred.new_zeros(())
        if self.pinn_mode == 1 or self.pinn_mode == 2:
            r1, r2, r3 = self.ED_BCS(y_pred, u)
        else:
            r1, r2, r3 = zero, zero, zero
        if self.pinn_mode == 2 or self.pinn_mode == 3:
            mixed = torch.cat([y[:, :, 0:2], y_pred[:, :, 2:]], dim=2)
            # Same call as the original GetLoss. The result is not added to the scalar loss.
            self.ED_BCS(mixed, u)
        return loss_obs, r1, r2, r3, (r1 + r2 + r3)

    def loss_on_batch(self, y, x, u):
        y_pred = self.u_model(x)
        loss_obs, r1, r2, r3, _sum_r = self.GetLoss(y, y_pred, u)
        weighted_r1 = self.lamb_l1 * r1
        weighted_r2 = self.lamb_l2 * r2
        weighted_r3 = self.lamb_l3 * r3
        loss_f = weighted_r1 + weighted_r2 + weighted_r3
        loss_bc = self.lamb_bc * loss_obs
        total = loss_bc + loss_f
        return {
            "total": total,
            "loss_obs": loss_obs,
            "loss_bc": loss_bc,
            "loss_f": loss_f,
            "r1": weighted_r1,
            "r2": weighted_r2,
            "r3": weighted_r3,
            "y_pred": y_pred,
        }

    def erro(self):
        # Toggle dropout only. model.train()/eval() on the LSTM between
        # forward and backward drops the saved recurrent workspace.
        drops = tuple(module for module in self.u_model.modules() if isinstance(module, nn.Dropout))
        previous = [module.training for module in drops]
        for module in drops:
            module.eval()
        with torch.no_grad():
            y_pred = self.u_model(self.test_X)
            err = torch.mean(torch.square(self.test_y[:, :, 0:2] - y_pred[:, :, 0:2]))
            value = float(err.detach().cpu())
        for module, was_training in zip(drops, previous):
            module.train(was_training)
        return value

    def _as_float(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _record_state(self, parts, step):
        self.train_state.step = str(step)
        self.train_state.rho = self._as_float(self.rho) * RHO_SCALE
        self.train_state.PI = self._as_float(self.PI) * PI_SCALE
        self.train_state.loss_test = self.erro()
        self.train_state.loss_train = self._as_float(parts["total"])
        self.train_state.loss_train_bc = self._as_float(parts["loss_bc"])
        self.train_state.loss_train_f = self._as_float(parts["loss_f"])
        self.train_state.loss_train_x1 = self._as_float(parts["r1"])
        self.train_state.loss_train_x2 = self._as_float(parts["r2"])
        self.train_state.loss_train_x3 = self._as_float(parts["r3"])
        self.train_state.weights = self.snapshot_weights()
        self.train_state.update_best()

    def _append_history(self, step):
        self.losshistory.append(
            step,
            self.train_state.loss_train,
            self.train_state.loss_train_bc,
            self.train_state.loss_train_f,
            self.train_state.loss_train_x1,
            self.train_state.loss_train_x2,
            self.train_state.loss_train_x3,
            self.train_state.loss_test,
            None,
        )
        self.varhistory.append(step, self.train_state.rho, self.train_state.PI)

    def _custom_log(self, parts):
        r1 = self._as_float(parts["r1"])
        r2 = self._as_float(parts["r2"])
        r3 = self._as_float(parts["r3"])
        rho_phys = self._as_float(self.rho) * RHO_SCALE
        pi_phys = self._as_float(self.PI) * PI_SCALE
        return f"|{r1:.2e},{r2:.2e},{r3:.2e}|{rho_phys:.1f} {pi_phys:.2e}"

    def fit(self, train_data, tf_epochs=5000, adapt_w=False):
        del adapt_w
        train_dataset, self.test_X, self.test_y = train_data
        self.test_X = self.test_X.to(self.device)
        self.test_y = self.test_y.to(self.device)
        if self.logger is not None:
            self.logger.set_error_fn(self.erro)
            self.logger.log_train_opt("Adam", self.pinn_mode, self.get_lamb_weights())
            self.logger.start_time = time.time()
        self.u_model.train()
        try:
            for epoch in range(int(tf_epochs)):
                last = None
                for x_batch, y_batch, u_batch in train_dataset:
                    if x_batch.shape[0] < 6:
                        continue
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    u_batch = u_batch.to(self.device)
                    self.u_model.train()
                    self.optimizer.zero_grad()
                    parts = self.loss_on_batch(y_batch, x_batch, u_batch)
                    if not torch.isfinite(parts["total"]):
                        print(
                            "Nan values appear. Stopping training",
                            self._as_float(parts["r1"]),
                            self._as_float(parts["r2"]),
                            self._as_float(parts["r3"]),
                            self._as_float(parts["loss_obs"]),
                        )
                        if self.logger is not None:
                            self.logger.log_train_end(tf_epochs, self.train_state)
                        self.summary_train(self.train_state)
                        raise Exception("Loss with Nan values found")
                    parts["total"].backward()
                    self._record_state(parts, self.epoch)
                    self.optimizer.step()
                    last = parts
                if last is None:
                    raise Exception("No batch had at least 6 rows")
                if epoch % 20 == 0:
                    self._append_history(self.epoch)
                if self.logger is not None:
                    self.logger.log_train_epoch(
                        self.epoch,
                        self.train_state.loss_train,
                        self.train_state.loss_train_f,
                        self.train_state.loss_train_bc,
                        custom=self._custom_log(last),
                    )
                self.epoch = self.epoch + 1
                self.varhistory.update_best(self.train_state)
            self.nsess = self.nsess + int(tf_epochs)
            if self.logger is not None:
                self.logger.log_train_end(tf_epochs, self.train_state)
            self.summary_train(self.train_state)
        except Exception as err:
            print(err.args)
            raise
        return self.losshistory, self.train_state, self.varhistory

    def fit_LBFGS(self, dataset, nt_config):
        train_x, train_y, u_train, self.test_X, self.test_y = dataset
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        u_train = u_train.to(self.device)
        self.test_X = self.test_X.to(self.device)
        self.test_y = self.test_y.to(self.device)
        if train_x.shape[0] < 6:
            raise Exception("Model output size must have at least 6 time points ")
        if self.logger is not None:
            self.logger.set_error_fn(self.erro)
            self.logger.log_train_opt("LBFGS", self.pinn_mode, self.get_lamb_weights())
            self.logger.start_time = time.time()
        # Dropout stays off. The quasi-Newton objective has to be the same
        # function at every trial step of the line search.
        self.u_model.eval()
        max_iter = int(nt_config.maxIter)
        history_size = int(nt_config.nCorrection) if nt_config.nCorrection else 50
        tol = float(nt_config.tol) if nt_config.tol else 1e-5
        tol_fun = float(nt_config.tolfun) if nt_config.tolfun else 1e-5
        # strong_wolfe evaluates the loss several times per iteration.
        max_eval = max(max_iter * 20, 20)
        optimizer = torch.optim.LBFGS(
            self.training_parameters(),
            lr=1.0,
            max_iter=max_iter,
            max_eval=max_eval,
            history_size=history_size,
            tolerance_grad=tol,
            tolerance_change=tol_fun,
            line_search_fn="strong_wolfe",
        )
        self._lbfgs_eval = int(self.epoch)

        def closure():
            optimizer.zero_grad()
            parts = self.loss_on_batch(train_y, train_x, u_train)
            loss = parts["total"]
            if not torch.isfinite(loss):
                raise Exception("Loss with Nan values found")
            loss.backward()
            self._record_state(parts, self._lbfgs_eval)
            self._lbfgs_eval += 1
            if self._lbfgs_eval % 10 == 0:
                rho_phys = self._as_float(self.rho) * RHO_SCALE
                pi_phys = self._as_float(self.PI) * PI_SCALE
                pct_rho = 100.0 * abs(RHO_SCALE - rho_phys) / RHO_SCALE
                pct_pi = 100.0 * abs(PI_SCALE - pi_phys) / PI_SCALE
                custom = (
                    f"[{self._as_float(parts['r1']):.2e},{self._as_float(parts['r2']):.2e},{self._as_float(parts['r3']):.2e}]"
                    f"{rho_phys:.1f} {pi_phys * 3.6e8:.4f} ({pct_rho:1.3f}%) ({pct_pi:1.3f}%)"
                )
                if self.logger is not None:
                    self.logger.log_train_epoch(
                        self._lbfgs_eval,
                        parts["total"],
                        parts["loss_f"],
                        parts["loss_obs"],
                        custom=custom,
                    )
                self._append_history(self._lbfgs_eval)
                self.varhistory.update_best(self.train_state)
            return loss

        optimizer.step(closure)
        self.epoch = self._lbfgs_eval
        self.rho = self.u_model.rho
        self.PI = self.u_model.PI
        if (not self.losshistory.steps) or self.losshistory.steps[-1] != self._lbfgs_eval:
            self._append_history(self._lbfgs_eval)
            self.varhistory.update_best(self.train_state)
        if self.logger is not None:
            self.logger.log_train_end(self.epoch, self.train_state)
        self.summary_train(self.train_state)
        return self.losshistory, self.train_state, self.varhistory

    def use_best_weights(self):
        weights = self.train_state.best_test_weights
        state = {k: torch.as_tensor(v) for k, v in weights.items()}
        self.u_model.load_state_dict(state)
        self.rho = self.u_model.rho
        self.PI = self.u_model.PI

    def disregard_best_weights(self):
        self.train_state.best_loss_train = np.inf

    def summary_train(self, train_state):
        rho_now = self._as_float(self.rho) * RHO_SCALE
        pi_now = self._as_float(self.PI) * PI_SCALE
        pct_rho = 100.0 * abs(RHO_SCALE - rho_now) / RHO_SCALE
        pct_pi = 100.0 * abs(PI_SCALE - pi_now) / PI_SCALE
        best_rho = 0.0 if train_state.best_test_rho is None else float(train_state.best_test_rho)
        best_pi = 0.0 if train_state.best_test_PI is None else float(train_state.best_test_PI)
        print(
            f"Best model at step:{train_state.best_test_step}, Best rho:{best_rho:.1f}, Best PI:{best_pi:.4e}"
            f" ({pct_rho:1.3f}%) ({pct_pi:1.3f}%)"
        )
        print("  train loss: {:.2e}".format(float(train_state.best_test_train)))
        print("  test loss: {:.2e}".format(float(train_state.best_test_loss)))
        self.is_header_print = False

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.u_model.state_dict().items()}
        payload = {
            "state_dict": state,
            "rho": self._as_float(self.rho),
            "PI": self._as_float(self.PI),
            "width": self.u_model.width,
            "n_steps_in": self.u_model.n_steps_in,
            "n_steps_out": self.u_model.n_steps_out,
            "n_features": self.u_model.n_features,
            "pinn_mode": self.pinn_mode_name,
            "lambdas": [float(self.lamb_bc), float(self.lamb_l1), float(self.lamb_l2), float(self.lamb_l3)],
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path):
        if os.path.isdir(path):
            path = os.path.join(path, "model.pt")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if int(ckpt["width"]) != int(self.u_model.width):
            raise ValueError(
                f"Checkpoint width {ckpt['width']} does not match the model width {self.u_model.width}."
            )
        self.u_model.load_state_dict(ckpt["state_dict"])
        self.rho = self.u_model.rho
        self.PI = self.u_model.PI
        if "lambdas" in ckpt:
            self.lamb_bc, self.lamb_l1, self.lamb_l2, self.lamb_l3 = [float(v) for v in ckpt["lambdas"]]
        if "pinn_mode" in ckpt:
            self.pinn_mode_name = ckpt["pinn_mode"]
            self.pinn_mode = PINN_MODES[self.pinn_mode_name]
        return ckpt
