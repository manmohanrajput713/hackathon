"""Training-loss record for the gas-lift PINC."""

from __future__ import annotations


class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_y = []
        self.loss_f = []
        self.loss_val = []
        self.feasible_fraction = []

    def append(self, step, loss_train, loss_y, loss_f, loss_val, feasible_fraction):
        self.steps.append(int(step))
        self.loss_train.append(float(loss_train))
        self.loss_y.append(float(loss_y))
        self.loss_f.append(float(loss_f))
        if loss_val is None:
            loss_val = self.loss_val[-1] if self.loss_val else float("nan")
        self.loss_val.append(float(loss_val))
        self.feasible_fraction.append(float(feasible_fraction))
