import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.BuildDataset import load_dataset
from data.Logger import Logger
from data.TrainingReport import TrainingReport
from data.pinn_BCS import pinn_vfm
from data.utils import Struct, save_model_files


def build_argparser():
    parser = argparse.ArgumentParser(description="Adam training for the PINN virtual flow meter.")
    parser.add_argument("--dataset", default="dataset01.pk")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out", default="model_adam_200")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fig-dir", default=None)
    return parser


def print_losses(trainstate, pinn):
    print(
        f"loss_total={float(trainstate.loss_train):.6e} "
        f"loss_data={float(trainstate.loss_train_bc):.6e} "
        f"loss_physics={float(trainstate.loss_train_f):.6e} "
        f"loss_r1={float(trainstate.loss_train_x1):.6e} "
        f"loss_r2={float(trainstate.loss_train_x2):.6e} "
        f"loss_r3={float(trainstate.loss_train_x3):.6e} "
        f"loss_test={float(trainstate.loss_test):.6e} "
        f"rho={float(pinn.rho.detach().cpu()) * 950:.6f} "
        f"PI={float(pinn.PI.detach().cpu()) * 2.32e-9:.6e}"
    )


def save_report_figures(report, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    report.gen_plot_result().savefig(os.path.join(fig_dir, "result.png"), bbox_inches="tight")
    report.gen_plot_loss_res().savefig(os.path.join(fig_dir, "loss.png"), bbox_inches="tight")
    report.gen_var_plot().savefig(os.path.join(fig_dir, "var.png"), bbox_inches="tight")
    plt.close("all")


def main(argv=None):
    args = build_argparser().parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ds = load_dataset(args.dataset)

    nt_config = Struct()
    # parallel_iterations from the old quasi-Newton setup has no equivalent here.
    nt_config.parallel_iter = 2
    nt_config.maxIter = 400
    nt_config.nCorrection = 50
    nt_config.tolfun = 1e-5
    nt_config.tol = 1e-5
    nt_config.maxIter = 100

    logger = Logger(frequency=100)
    neurons = 15  # unused name; the LSTM width is Nc
    rho = 950
    PI = 2.32 * 1e-9
    start_rho = 0.9 * 950 / rho
    start_PI = 0.9 * 2.32 * 1e-9 / PI
    var = [start_rho, start_PI]
    n_features = 6
    Nc = 10
    pinn = pinn_vfm(
        Nc, logger,
        var=var, pinn_mode="on",
        inputs=n_features,
        n_steps_in=ds.n_steps_in,
        n_steps_out=ds.n_steps_out,
        parameters=ds.parameters,
        lr=args.lr,
    )
    pinn.lamb_l1 = 1.0
    pinn.lamb_l2 = 1.0
    pinn.lamb_l3 = 1.0
    loss_history, trainstate, var_history = pinn.fit(ds.adam_dataset(), tf_epochs=args.epochs)
    folder_string = args.out
    objects2save = {"Loss": loss_history, "trainstate": trainstate, "vartrain": var_history}
    save_model_files(folder_string, objects2save, pinn)
    training_report = TrainingReport(pinn, [loss_history, trainstate, var_history], ds)
    fig_dir = args.fig_dir or folder_string
    save_report_figures(training_report, fig_dir)
    print_losses(trainstate, pinn)
    return loss_history, trainstate, var_history


if __name__ == "__main__":
    main()
