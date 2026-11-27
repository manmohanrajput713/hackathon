import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.sample import PBH_GATE, discover_domain, domain_path, gate_ok, save_domain


def build_argparser():
    parser = argparse.ArgumentParser(description="Discover the feasible state cube for one gas-lift well.")
    parser.add_argument("--well", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--pool", type=int, default=20000)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="npz path. Default: dataset/well{id}_domain.npz")
    return parser


def main(argv=None):
    args = build_argparser().parse_args(argv)
    payload = discover_domain(args.well, pool_size=args.pool, seed=args.seed, hours=args.hours)
    path = Path(args.out) if args.out else domain_path(ROOT, args.well)
    save_domain(path, payload)
    values = payload["settled_pbh_bar"]
    lo, hi = PBH_GATE
    print(f"steady-state gate: at least one settled P_bh in [{lo:.0f}, {hi:.0f}] bar")
    if not gate_ok(values):
        print(
            "Steady-state gate failed. No settled bottom-hole pressure is near "
            f"{lo:.0f}-{hi:.0f} bar. Revisit P_0, D_bh, and pressure units before training."
        )
        return 2
    print("steady-state gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
