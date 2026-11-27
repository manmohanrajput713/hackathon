"""Skip-connection PINC, Wang et al. (2021) as used by Kittelsen et al. (2024).

Input X is 6-D: [t/T, scaled initial state (3), valve openings (2)].
Output is the scaled state at that time, 3-D, with a linear head.

U and V are two tanh encoders of X. Each hidden layer builds
    Z = tanh(W z_in + b),   A = (1 - Z) * U + Z * V
with z_in = X on the first layer and A on the later layers.
`--no-skip` is the same depth and width as a plain tanh MLP.
"""

from __future__ import annotations

import torch
from torch import nn


def _glorot(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class SkipPINC(nn.Module):
    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 3,
        n_layers: int = 6,
        n_neurons: int = 30,
        skip: bool = True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if n_neurons < 1:
            raise ValueError("n_neurons must be at least 1")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_layers = int(n_layers)
        self.n_neurons = int(n_neurons)
        self.skip = bool(skip)
        if self.skip:
            self.enc_u = nn.Linear(self.in_dim, self.n_neurons)
            self.enc_v = nn.Linear(self.in_dim, self.n_neurons)
            hidden = [nn.Linear(self.in_dim, self.n_neurons)]
            for _ in range(self.n_layers - 1):
                hidden.append(nn.Linear(self.n_neurons, self.n_neurons))
            self.hidden = nn.ModuleList(hidden)
        else:
            self.enc_u = None
            self.enc_v = None
            hidden = []
            width = self.in_dim
            for _ in range(self.n_layers):
                hidden.append(nn.Linear(width, self.n_neurons))
                width = self.n_neurons
            self.hidden = nn.ModuleList(hidden)
        self.out = nn.Linear(self.n_neurons, self.out_dim)
        _glorot(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.skip:
            h = x
            for layer in self.hidden:
                h = torch.tanh(layer(h))
            return self.out(h)
        u = torch.tanh(self.enc_u(x))
        v = torch.tanh(self.enc_v(x))
        activated = x
        for k, layer in enumerate(self.hidden):
            z = torch.tanh(layer(x if k == 0 else activated))
            activated = (1.0 - z) * u + z * v
        return self.out(activated)

    def architecture(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "skip": self.skip,
            "activation": "tanh",
        }

    def summary(self) -> str:
        n_param = sum(p.numel() for p in self.parameters())
        kind = "skip" if self.skip else "mlp"
        return f"SkipPINC {kind} layers={self.n_layers} width={self.n_neurons} params={n_param}"
