"""Seed, checkpoints, scaling, and plots. Matplotlib uses the Agg backend."""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.algebraic_nn import AlgebraicNet
from data.skip_pinc import SkipPINC


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def scale_state(y, y_min, y_max) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float64)
    lo = np.asarray(y_min, dtype=np.float64)
    span = np.asarray(y_max, dtype=np.float64) - lo
    return (y_arr - lo) / span


def unscale_state(y_s, y_min, y_max) -> np.ndarray:
    y_arr = np.asarray(y_s, dtype=np.float64)
    lo = np.asarray(y_min, dtype=np.float64)
    span = np.asarray(y_max, dtype=np.float64) - lo
    return y_arr * span + lo


def save_checkpoint(path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_pinc(arch: dict) -> SkipPINC:
    model = SkipPINC(
        in_dim=int(arch.get("in_dim", 6)),
        out_dim=int(arch.get("out_dim", 3)),
        n_layers=int(arch["n_layers"]),
        n_neurons=int(arch["n_neurons"]),
        skip=bool(arch.get("skip", True)),
    )
    return model.double()


def build_algebraic(arch: dict) -> AlgebraicNet:
    model = AlgebraicNet(
        in_dim=int(arch.get("in_dim", 5)),
        out_dim=int(arch.get("out_dim", 4)),
        n_layers=int(arch["n_layers"]),
        n_neurons=int(arch["n_neurons"]),
    )
    return model.double()


def load_pinc(path) -> tuple[SkipPINC, dict]:
    ckpt = load_checkpoint(path)
    model = build_pinc(ckpt["architecture"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_algebraic(path) -> tuple[AlgebraicNet, dict]:
    ckpt = load_checkpoint(path)
    model = build_algebraic(ckpt["architecture"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def store_pickle(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def plot_loss(history, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = np.asarray(history.steps)
    ax.semilogy(steps, np.maximum(history.loss_train, 1e-16), label="train")
    if np.any(np.isfinite(history.loss_val)):
        ax.semilogy(steps, np.maximum(history.loss_val, 1e-16), label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forecast(t, y_pinc, y_rk, pbh_pinc, pbh_rk, u_seq, step, path) -> None:
    """t includes the initial knot. u_seq has one row per control interval."""
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    labels = [r"$m_{G,an}$ (kg)", r"$m_{G,tb}$ (kg)", r"$m_{L,tb}$ (kg)"]
    for i, label in enumerate(labels):
        axes[i].plot(t, y_rk[:, i], color="C0", label="RK45")
        axes[i].plot(t, y_pinc[:, i], color="C6", ls="--", label="PINC")
        axes[i].scatter(t, y_pinc[:, i], color="C6", s=18, zorder=3)
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[3].plot(t, pbh_rk, color="C0", label="RK map")
    axes[3].plot(t, pbh_pinc, color="C6", ls="--", label="algebraic net")
    axes[3].set_ylabel(r"$P_{bh}$ (bar)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")
    t_u = np.arange(u_seq.shape[0] + 1) * step
    axes[4].step(t_u, np.r_[u_seq[:, 0], u_seq[-1, 0]], where="post", label="u1")
    axes[4].step(t_u, np.r_[u_seq[:, 1], u_seq[-1, 1]], where="post", label="u2")
    axes[4].set_ylabel("valve")
    axes[4].set_xlabel("time (s)")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="best")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_mpc(records: dict, path) -> None:
    """records[name] = dict with t, pbh, ref, u, y."""
    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
    styles = {"pinc": ("C6", "-"), "slmpc": ("C0", "--")}
    for name, rec in records.items():
        color, ls = styles.get(name, ("k", "-"))
        axes[3].plot(rec["t"], rec["pbh"], color=color, ls=ls, label=name)
        axes[4].step(rec["t"], rec["u"][:, 0], where="post", color=color, ls=ls, label=f"{name} u1")
        axes[4].step(rec["t"], rec["u"][:, 1], where="post", color=color, ls=":", label=f"{name} u2")
        for i in range(3):
            axes[i].plot(rec["t"], rec["y"][:, i], color=color, ls=ls, label=name)
    ref_name = next(iter(records))
    axes[3].plot(records[ref_name]["t"], records[ref_name]["ref"], color="k", lw=1, label="reference")
    labels = [r"$m_{G,an}$ (kg)", r"$m_{G,tb}$ (kg)", r"$m_{L,tb}$ (kg)", r"$P_{bh}$ (bar)", "valve"]
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    axes[-1].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
