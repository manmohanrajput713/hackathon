"""Latest and best training metrics for one PINC run."""

from __future__ import annotations

import numpy as np


class TrainState(object):
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self.loss_train = None
        self.loss_y = None
        self.loss_f = None
        self.loss_val = None
        self.feasible_fraction = None
        self.best_epoch = 0
        self.best_loss_train = np.inf
        self.best_loss_val = np.inf

    def update(self, epoch, loss_train, loss_y, loss_f, loss_val, feasible_fraction):
        self.epoch = int(epoch)
        self.step = int(epoch)
        self.loss_train = float(loss_train)
        self.loss_y = float(loss_y)
        self.loss_f = float(loss_f)
        self.loss_val = None if loss_val is None else float(loss_val)
        self.feasible_fraction = float(feasible_fraction)
        if self.loss_train < self.best_loss_train:
            self.best_loss_train = self.loss_train
            self.best_epoch = self.epoch
        if self.loss_val is not None and self.loss_val < self.best_loss_val:
            self.best_loss_val = self.loss_val
