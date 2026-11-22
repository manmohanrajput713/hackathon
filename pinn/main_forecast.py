import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import pickle

import numpy as np

from data.BuildDataset import BuildingDataset, load_dataset
from data.PinnPredictor import PinnPredictor
from data.forecast import free_run, save_forecast_plot


def build_argparser():
    parser = argparse.ArgumentParser(description="Free-run forecast. Predicted q is not fed back.")
    parser.add_argument("--checkpoint", default="model_adam_lbfgs/model.pt")
    parser.add_argument("--dataset", default="dataset_opera.pk")
    parser.add_argument("--npz", default="./dataset/BCS_data_train_limitado_f_zc_pm_pr_opera.npz")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--out", default="forecast.png")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--split", type=float, default=None,
                        help="Raw-series fraction used when rebuilding. Default 0.98 for the opera file, else 0.7.")
    return parser


def load_or_build(path, npz, seed, rebuild, split):
    if not rebuild and os.path.exists(path):
        try:
            return load_dataset(path)
        except RuntimeError as exc:
            print(exc)
            print(f"Rebuilding {path} from {npz}")
    if not os.path.exists(npz):
        raise SystemExit(f"Dataset file not found: {path} or {npz}")
    dados = np.load(npz)
    if split is None:
        split = 0.98 if "opera" in os.path.basename(npz) else 0.7
    split_point = int(split * len(dados["x1"]))
    ds = BuildingDataset(20, 1, dados, split_point, batch_size=500, seed=seed)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as outp:
        pickle.dump(ds, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {path}")
    return ds


def main(argv=None):
    args = build_argparser().parse_args(argv)
    ds = load_or_build(args.dataset, args.npz, args.seed, args.rebuild, args.split)
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}. Train first; old .h5 files are not loaded.")
    predictor = PinnPredictor(args.checkpoint)
    predict = free_run(
        predictor.model,
        ds.un,
        steps=args.steps,
        n_steps_in=ds.n_steps_in,
        parameters=ds.parameters,
    )
    predict_np = predict.detach().cpu().numpy()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.splitext(args.out)[0] + ".npy"
    np.save(npy_path, predict_np)
    save_forecast_plot(predict_np, args.out, parameters=ds.parameters)
    print(f"forecast_shape={tuple(predict_np.shape)} finite={bool(np.isfinite(predict_np).all())}")
    print(f"Wrote {args.out} and {npy_path}")
    return predict_np


if __name__ == "__main__":
    main()
