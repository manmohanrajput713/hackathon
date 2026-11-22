import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.param import get_well
from data.pinn_well import train_adam
from data.sample import domain_path, load_domain, make_training_batches, save_batches
from data.utils import set_seed


def build_argparser():
    parser = argparse.ArgumentParser(description="Adam training for the gas-lift skip PINC.")
    parser.add_argument("--well", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=7e-3)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--nf", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--neurons", type=int, default=30)
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    domain_file = Path(args.domain) if args.domain else domain_path(ROOT, args.well)
    out = Path(args.out) if args.out else ROOT / "runs" / f"well{args.well}" / "adam"
    domain = load_domain(domain_file)
    well = get_well(args.well)
    batches = make_training_batches(domain, args.nt, args.nf, args.n_val, well, args.seed)
    save_batches(out / "batches.npz", batches)
    resume = out / "model.pt" if args.resume else None
    train_adam(
        domain,
        batches,
        out,
        well_id=args.well,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        n_layers=args.layers,
        n_neurons=args.neurons,
        skip=not args.no_skip,
        val_every=args.val_every,
        resume_path=resume,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
