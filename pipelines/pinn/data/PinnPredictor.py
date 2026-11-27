import os

import numpy as np
import torch

from data.parameters import Parameters
from data.pinn_BCS import load_module


class PinnPredictor(object):
    def __init__(self, path_model, device=None):
        self.parameters = Parameters()
        self.path_model = path_model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = self.get_model()

    def get_model(self):
        path = self.path_model
        if os.path.isdir(path):
            path = os.path.join(path, "model.pt")
        print(f"Loading model from {path}")
        model, _ckpt = load_module(path, device=self.device)
        return model

    def norm_u(self, u):
        aux = []
        u[2] = u[2] - self.parameters.u0[0]
        u[3] = u[3] - self.parameters.u0[1]
        for i, valor in enumerate(u):
            aux.append(valor / self.parameters.uc[i])
        return np.hstack(aux)

    def norma_x(self, x):
        xn = [(x[:, i] - self.parameters.x0[i]) / self.parameters.xc[i] for i in range(3)]
        return np.array(xn).T
