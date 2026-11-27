import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.algebraic_nn import AlgebraicNet
from data.mpc import PINCMPC, SLMPC, closed_loop
from data.param import get_well, nominal_input, nominal_state
from data.sample import domain_path, load_domain
from data.skip_pinc import SkipPINC
from data.utils import load_algebraic, load_pinc, plot_mpc, set_seed
from data.well_ode import algebraics_numpy, is_feasible

PAPER_NOISELESS_IAE = 0.038
PAPER_NOISY_PINC = 0.317
PAPER_NOISY_SLMPC = 0.327


def build_argparser():
    parser = argparse.ArgumentParser(description="Bottom-hole pressure MPC. Default is a short smoke.")
    parser.add_argument("--well", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--noise", type=float, default=0.0, help="0 or 0.05 (std = noise * |x|)")
    parser.add_argument("--controller", choices=("pinc", "slmpc", "both"), default="pinc")
    parser.add_argument("--minutes", type=int, default=None, help="Plant steps of 60 s. Default 3, or 90 with --full.")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--nu", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="90 min, N=50, Nu=45")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--pinc", default=None)
    parser.add_argument("--algebraic", default=None)
    parser.add_argument("--out", default=None)
    return parser


def _load_models(args):
    pinc_path = Path(args.pinc) if args.pinc else ROOT / "runs" / f"well{args.well}" / "lbfgs" / "model.pt"
    if not pinc_path.is_file():
        alt = ROOT / "runs" / f"well{args.well}" / "adam" / "model.pt"
        pinc_path = alt if alt.is_file() else None
    alg_path = Path(args.algebraic) if args.algebraic else ROOT / "runs" / f"well{args.well}" / "algebraic" / "model.pt"
    if pinc_path is not None and pinc_path.is_file():
        pinc, ckpt = load_pinc(pinc_path)
        y_min = np.asarray(ckpt["y_min"], dtype=np.float64)
        y_max = np.asarray(ckpt["y_max"], dtype=np.float64)
        print(f"loaded PINC {pinc_path}")
    else:
        print("no PINC checkpoint; the smoke uses an untrained skip network")
        pinc = SkipPINC().double()
        y_min, y_max, _zmean, _zstd = _fallback_scale(args.well)
    if alg_path.is_file():
        algebraic, alg_ckpt = load_algebraic(alg_path)
        z_mean = np.asarray(alg_ckpt["z_mean"], dtype=np.float64)
        z_std = np.asarray(alg_ckpt["z_std"], dtype=np.float64)
        y_min = np.asarray(alg_ckpt["y_min"], dtype=np.float64)
        y_max = np.asarray(alg_ckpt["y_max"], dtype=np.float64)
        print(f"loaded algebraic net {alg_path}")
    else:
        print("no algebraic checkpoint; the smoke uses an untrained algebraic net")
        algebraic = AlgebraicNet().double()
        if pinc_path is None or not Path(pinc_path).is_file():
            y_min, y_max, z_mean, z_std = _fallback_scale(args.well)
        else:
            _y0 = nominal_state(args.well)
            z = algebraics_numpy(_y0, nominal_input(), get_well(args.well))
            z_mean = np.asarray(z, dtype=np.float64)
            z_std = np.ones(4, dtype=np.float64)
    return pinc, algebraic, y_min, y_max, z_mean, z_std


def _fallback_scale(well_id):
    y0 = nominal_state(well_id)
    span = np.maximum(0.2 * np.abs(y0), 50.0)
    return y0 - span, y0 + span, np.array([95.0, 1.0, 0.0, 16.0]), np.ones(4)


def _initial_state(domain, well):
    if domain is not None and np.asarray(domain["settled_pbh_bar"]).size:
        pbh = np.asarray(domain["settled_pbh_bar"], dtype=np.float64)
        index = int(np.argmin(np.abs(pbh - 95.0)))
        return (
            np.asarray(domain["settled_y"][index], dtype=np.float64),
            np.asarray(domain["settled_u"][index], dtype=np.float64),
        )
    y0 = nominal_state(well.well_id)
    u0 = nominal_input()
    if not is_feasible(y0, u0, well):
        raise SystemExit("nominal state is not feasible and no domain file was found")
    return y0, u0


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    if args.full:
        minutes = 90 if args.minutes is None else args.minutes
        horizon = 50 if args.horizon is None else args.horizon
        nu = 45 if args.nu is None else args.nu
    else:
        minutes = 3 if args.minutes is None else args.minutes
        horizon = 5 if args.horizon is None else args.horizon
        nu = 4 if args.nu is None else args.nu
    if nu > horizon:
        raise SystemExit("Nu cannot exceed N")
    domain_file = Path(args.domain) if args.domain else domain_path(ROOT, args.well)
    domain = load_domain(domain_file) if domain_file.is_file() else None
    if domain is None:
        print(f"no domain file at {domain_file}; starting from the nominal feasible state")
    well = get_well(args.well)
    pinc, algebraic, y_min, y_max, z_mean, z_std = _load_models(args)
    x0, u_prev = _initial_state(domain, well)
    print(f"x0={x0.tolist()} u_prev={u_prev.tolist()} N={horizon} Nu={nu} minutes={minutes} noise={args.noise}")
    names = ("pinc", "slmpc") if args.controller == "both" else (args.controller,)
    records = {}
    for name in names:
        if name == "pinc":
            controller = PINCMPC(pinc, algebraic, y_min, y_max, z_mean, z_std, horizon, nu)
        else:
            controller = SLMPC(horizon, nu)
        record = closed_loop(
            controller,
            kind=name,
            well=well,
            x0=x0,
            u_prev=u_prev,
            minutes=minutes,
            noise=args.noise,
            seed=args.seed,
        )
        records[name] = record
        paper = PAPER_NOISELESS_IAE if args.noise == 0.0 else (PAPER_NOISY_PINC if name == "pinc" else PAPER_NOISY_SLMPC)
        label = "noiseless" if args.noise == 0.0 else f"noise std {args.noise:.0%} of |x|"
        print(f"{name} IAE={record['iae']:.4f} bar ({label}; paper example {paper:.3f} bar)")
    out = Path(args.out) if args.out else ROOT / "runs" / f"well{args.well}"
    out.mkdir(parents=True, exist_ok=True)
    plot_mpc(records, out / "mpc.png")
    np.savez(
        out / "mpc.npz",
        **{f"{name}_{key}": value for name, rec in records.items() for key, value in rec.items()},
    )
    print(f"wrote {out / 'mpc.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
