import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pinn_well import train_algebraic
from data.sample import domain_path, load_domain
from data.utils import set_seed


def build_argparser():
    parser = argparse.ArgumentParser(description="L-BFGS training of the algebraic output network.")
    parser.add_argument("--well", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--out", default=None)
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    domain_file = Path(args.domain) if args.domain else domain_path(ROOT, args.well)
    out = Path(args.out) if args.out else ROOT / "runs" / f"well{args.well}" / "algebraic"
    domain = load_domain(domain_file)
    train_algebraic(
        domain,
        out,
        well_id=args.well,
        n_train=args.n_train,
        n_val=args.n_val,
        iterations=args.iterations,
        seed=args.seed,
        n_layers=args.layers,
        n_neurons=args.neurons,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
