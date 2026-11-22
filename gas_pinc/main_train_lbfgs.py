import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.param import get_well
from data.pinn_well import train_lbfgs
from data.sample import domain_path, load_batches, load_domain, make_training_batches, save_batches
from data.utils import set_seed


def build_argparser():
    parser = argparse.ArgumentParser(description="L-BFGS refinement of a gas-lift skip PINC.")
    parser.add_argument("--well", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--nf", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--init", default=None, help="model.pt to start from. Default: runs/well{id}/adam/model.pt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true", help="Continue runs/well{id}/lbfgs/model.pt")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    domain_file = Path(args.domain) if args.domain else domain_path(ROOT, args.well)
    adam_dir = ROOT / "runs" / f"well{args.well}" / "adam"
    lbfgs_dir = ROOT / "runs" / f"well{args.well}" / "lbfgs"
    out = Path(args.out) if args.out else lbfgs_dir
    if args.resume:
        init = out / "model.pt"
    elif args.init:
        init = Path(args.init)
    else:
        init = adam_dir / "model.pt"
    if not init.is_file():
        raise SystemExit(f"missing checkpoint {init}. Train with main_train_adam.py first.")
    domain = load_domain(domain_file)
    batch_file = init.parent / "batches.npz"
    if batch_file.is_file() and args.nt == 1000 and args.nf == 10000:
        batches = load_batches(batch_file)
        print(f"loaded collocation batches from {batch_file}")
    else:
        batches = make_training_batches(
            domain, args.nt, args.nf, args.n_val, get_well(args.well), args.seed
        )
        save_batches(out / "batches.npz", batches)
    train_lbfgs(
        domain,
        batches,
        out,
        init_path=init,
        well_id=args.well,
        iterations=args.iterations,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
