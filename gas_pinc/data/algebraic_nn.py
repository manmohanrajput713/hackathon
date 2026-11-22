"""Feedforward net for algebraic well outputs.

Section 4.6 says this net sees only the three masses. Section 4.5 and the
CasADi listing pass the state and the control, and bottom-hole pressure
depends on the gas-lift choke through w_G_in and the tubing friction.
This net therefore takes five inputs:

    [m_G_an, m_G_tb, m_L_tb, u1, u2]

with the masses scaled by the PINC state bounds and the valves left in [0, 1].
The four outputs are standardized with the training mean and standard
deviation and inverted before they are reported:

    [P_bh (bar), w_G_in, w_G_res, w_L_res]
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from data.skip_pinc import _glorot


class AlgebraicNet(nn.Module):
    def __init__(self, in_dim: int = 5, out_dim: int = 4, n_layers: int = 4, n_neurons: int = 30):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_layers = int(n_layers)
        self.n_neurons = int(n_neurons)
        hidden = []
        width = self.in_dim
        for _ in range(self.n_layers):
            hidden.append(nn.Linear(width, self.n_neurons))
            width = self.n_neurons
        self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(self.n_neurons, self.out_dim)
        _glorot(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)

    def architecture(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation": "tanh",
        }

    def summary(self) -> str:
        n_param = sum(p.numel() for p in self.parameters())
        return f"AlgebraicNet layers={self.n_layers} width={self.n_neurons} params={n_param}"


def pack_algebraic_input(y, u, y_min, y_max) -> np.ndarray:
    """Scaled 5-D algebraic-net input. y is kg, u is in [0, 1]."""
    y_arr = np.asarray(y, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    single = y_arr.ndim == 1
    if single:
        y_arr = y_arr.reshape(1, 3)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(1, 2)
    if u_arr.shape[0] == 1 and y_arr.shape[0] != 1:
        u_arr = np.repeat(u_arr, y_arr.shape[0], axis=0)
    span = np.asarray(y_max, dtype=np.float64) - np.asarray(y_min, dtype=np.float64)
    y_s = (y_arr - np.asarray(y_min, dtype=np.float64)) / span
    packed = np.concatenate([y_s, u_arr], axis=1)
    return packed[0] if single else packed


def invert_algebraic(z_s, z_mean, z_std) -> np.ndarray:
    z_s = np.asarray(z_s, dtype=np.float64)
    mean = np.asarray(z_mean, dtype=np.float64)
    std = np.asarray(z_std, dtype=np.float64)
    return z_s * std + mean


def algebraic_predict(net: AlgebraicNet, y, u, y_min, y_max, z_mean, z_std) -> np.ndarray:
    """Physical algebraic outputs. P_bh is in bar."""
    packed = pack_algebraic_input(y, u, y_min, y_max)
    single = packed.ndim == 1
    if single:
        packed = packed.reshape(1, -1)
    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype
    xt = torch.as_tensor(packed, dtype=dtype, device=device)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        z_s = net(xt).detach().cpu().numpy()
    if was_training:
        net.train()
    z = invert_algebraic(z_s, z_mean, z_std)
    return z[0] if single else z
