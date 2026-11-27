import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

from data.BuildDataset import BuildingDataset


def build_argparser():
    parser = argparse.ArgumentParser(description="Build dataset_opera.pk from the operation npz.")
    parser.add_argument("--npz", default="./dataset/BCS_data_train_limitado_f_zc_pm_pr_opera.npz")
    parser.add_argument("--out", default="dataset_opera.pk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fig-dir", default=None)
    return parser


def save_object(obj, filename):
    with open(filename, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def main(argv=None):
    args = build_argparser().parse_args(argv)
    dados = np.load(args.npz)
    n_steps_in, n_steps_out = 20, 1
    split_point = int(0.98 * len(dados["x1"]))
    ds = BuildingDataset(n_steps_in, n_steps_out, dados, split_point, batch_size=500, seed=args.seed)
    save_object(ds, args.out)
    fig_dir = args.fig_dir or (os.path.dirname(args.out) or ".")
    os.makedirs(fig_dir, exist_ok=True)
    figs = ds.gen_fig()
    for name, fig in figs.items():
        fig.savefig(os.path.join(fig_dir, f"dataset_opera_{name}.png"), bbox_inches="tight")
    plt.close("all")
    print(f"Wrote {args.out} train_X={ds.train_X.shape} split_point={split_point} seed={args.seed}")
    return ds


if __name__ == "__main__":
    main()
