"""Console logger. Training is pinned to CPU."""

from __future__ import annotations

import time

import numpy as np
import torch


class Logger(object):
    def __init__(self, frequency: int = 50):
        print(f"PyTorch version: {torch.__version__}")
        print("Training device: cpu")
        self.frequency = max(1, int(frequency))
        self.start_time = time.time()

    def elapsed(self) -> str:
        seconds = int(max(0, time.time() - self.start_time))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def log_start(self, model, lr, kind: str):
        print("\nTraining started")
        print("================")
        print(model.summary())
        print(f"optimizer={kind}  lr={lr:.4g}")
        print("epoch | time | loss | mse_y | mse_F | mse_val | feasible")

    def log_epoch(self, epoch, loss, loss_y, loss_f, loss_val, feasible, force: bool = False):
        if not force and epoch % self.frequency != 0:
            return
        val = " " if loss_val is None or not np.isfinite(loss_val) else f"{loss_val:.3e}"
        print(
            f"{epoch:5d} | {self.elapsed()} | {loss:.3e} | {loss_y:.3e} | "
            f"{loss_f:.3e} | {val:>10} | {feasible:.2f}"
        )

    def log_end(self, epoch, loss):
        print("==================")
        print(f"Training finished at epoch {epoch}  duration={self.elapsed()}  loss={loss:.4e}")
