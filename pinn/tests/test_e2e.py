import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "dataset" / "BCS_data_train_limitado_f_zc_pm_pr_ident.npz"


def _run(cmd, env):
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr)
    if proc.returncode != 0:
        raise AssertionError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout + "\n" + proc.stderr


def _finite_state(path):
    with open(path, "rb") as handle:
        state = pickle.load(handle)
    for name in (
        "loss_train",
        "loss_train_bc",
        "loss_train_f",
        "loss_train_x1",
        "loss_train_x2",
        "loss_train_x3",
        "loss_test",
        "rho",
        "PI",
    ):
        value = getattr(state, name)
        assert value is not None, name
        assert np.isfinite(float(value)), name
    return state


def test_e2e_ident_npz(tmp_path):
    assert NPZ.is_file()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    py = sys.executable
    ds_path = tmp_path / "dataset01.pk"
    adam_dir = tmp_path / "model_adam_200"
    lbfgs_dir = tmp_path / "model_adam_lbfgs"
    fig = tmp_path / "forecast.png"
    commands = [
        [py, "main_build_dataset.py", "--out", str(ds_path), "--fig-dir", str(tmp_path), "--seed", "0"],
        [py, "main_train_adam.py", "--dataset", str(ds_path), "--epochs", "2", "--out", str(adam_dir), "--fig-dir", str(tmp_path / "adam_figs"), "--seed", "0"],
        [py, "main_train_lbfgs.py", "--dataset", str(ds_path), "--checkpoint", str(adam_dir / "model.pt"), "--iterations", "2", "--out", str(lbfgs_dir), "--fig-dir", str(tmp_path / "lbfgs_figs"), "--seed", "0"],
        [py, "main_forecast.py", "--dataset", str(ds_path), "--checkpoint", str(lbfgs_dir / "model.pt"), "--steps", "20", "--out", str(fig), "--npz", str(NPZ)],
    ]
    for cmd in commands:
        _run(cmd, env)

    adam_state = _finite_state(adam_dir / "trainstate.pk")
    lbfgs_state = _finite_state(lbfgs_dir / "trainstate.pk")
    with open(adam_dir / "Loss.pk", "rb") as handle:
        adam_loss = pickle.load(handle)
    with open(lbfgs_dir / "Loss.pk", "rb") as handle:
        lbfgs_loss = pickle.load(handle)
    for value in list(adam_loss.loss_train) + list(lbfgs_loss.loss_train):
        assert np.isfinite(float(value))

    ckpt = torch.load(lbfgs_dir / "model.pt", map_location="cpu", weights_only=False)
    from data.pinn_BCS import load_module
    model, loaded = load_module(str(lbfgs_dir / "model.pt"), device="cpu")
    assert model.rho.detach().cpu().item() == ckpt["rho"]
    assert model.PI.detach().cpu().item() == ckpt["PI"]
    assert loaded["rho"] == ckpt["rho"]
    assert loaded["PI"] == ckpt["PI"]
    assert ckpt["width"] == 10

    forecast = np.load(tmp_path / "forecast.npy")
    assert forecast.shape == (21, 1, 3)
    assert np.isfinite(forecast).all()
    assert fig.is_file() and fig.stat().st_size > 8
    assert fig.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.isfinite(adam_state.loss_train)
    assert np.isfinite(lbfgs_state.loss_train)
