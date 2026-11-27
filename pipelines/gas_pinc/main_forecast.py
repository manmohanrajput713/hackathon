import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.forecast import rollout_vs_rk
from data.param import T_HORIZON, get_well
from data.sample import domain_path, load_domain
from data.utils import load_algebraic, load_pinc, plot_forecast, set_seed
from data.well_ode import InfeasibleState, is_feasible


PAPER_STATE_IAE = 0.00994
PAPER_STATE_STD = 0.00621
PAPER_PBH_IAE = 0.3087
PAPER_ONE_STATE = 0.00959
PAPER_ONE_PBH = 0.2813


def build_argparser():
    parser = argparse.ArgumentParser(description="50-step self-loop forecast against RK45.")
    parser.add_argument("--well", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--pinc", default=None)
    parser.add_argument("--algebraic", default=None)
    parser.add_argument("--out", default=None, help="Directory for forecast.npz and forecast.png")
    return parser


def _draw_trial(domain, well, steps, rng):
    pool_y = np.asarray(domain["y_pool"], dtype=np.float64)
    pool_u = np.asarray(domain["u_pool"], dtype=np.float64)
    for _ in range(40):
        index = int(rng.integers(0, pool_y.shape[0]))
        y0 = pool_y[index]
        u_seq = rng.random((steps, 2))
        u_seq = np.clip(u_seq, 1e-3, 1.0)
        if not is_feasible(y0, u_seq[0], well, friction="haaland"):
            continue
        try:
            # Probe the first plant step so a bad draw fails before the plot.
            from data.sample import integrate_haaland

            probe = integrate_haaland(y0, u_seq[0], well, T_HORIZON)
            if not probe["reached"]:
                continue
        except InfeasibleState:
            continue
        return y0, u_seq
    # Fall back to a stored feasible pair and the pool inputs, repeated.
    y0 = pool_y[0]
    u_seq = np.resize(pool_u, (steps, 2)).copy()
    return y0, u_seq


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    domain = load_domain(Path(args.domain) if args.domain else domain_path(ROOT, args.well))
    pinc_path = Path(args.pinc) if args.pinc else ROOT / "runs" / f"well{args.well}" / "lbfgs" / "model.pt"
    if not pinc_path.is_file():
        pinc_path = ROOT / "runs" / f"well{args.well}" / "adam" / "model.pt"
    alg_path = Path(args.algebraic) if args.algebraic else ROOT / "runs" / f"well{args.well}" / "algebraic" / "model.pt"
    net, ckpt = load_pinc(pinc_path)
    algebraic = None
    alg_ckpt = None
    if alg_path.is_file():
        algebraic, alg_ckpt = load_algebraic(alg_path)
    else:
        print(f"no algebraic checkpoint at {alg_path}; P_bh IAE will be omitted")
    well = get_well(args.well)
    rng = np.random.default_rng(args.seed)
    y0, u_seq = _draw_trial(domain, well, args.steps, rng)
    result = rollout_vs_rk(
        net, algebraic, alg_ckpt, y0, u_seq, ckpt["y_min"], ckpt["y_max"], well
    )
    metrics = result["metrics"]
    print(
        f"state IAE={metrics['IAE_states']:.5f}  "
        f"(paper well-1 mean {PAPER_STATE_IAE:.5f}, std {PAPER_STATE_STD:.5f}, "
        f"one trial {PAPER_ONE_STATE:.5f})"
    )
    print(
        f"P_bh IAE={metrics['IAE_Pbh']:.4f} bar  "
        f"(paper well-1 mean {PAPER_PBH_IAE:.4f} bar, one trial {PAPER_ONE_PBH:.4f} bar)"
    )
    out = Path(args.out) if args.out else ROOT
    t = np.arange(result["y_pinc"].shape[0]) * T_HORIZON
    np.savez(
        out / "forecast.npz",
        t=t,
        y_pinc=result["y_pinc"],
        y_rk=result["y_rk"],
        pbh_pinc=result["pbh_pinc"],
        pbh_rk=result["pbh_rk"],
        u=result["u"],
        iae_states=metrics["IAE_states"],
        iae_pbh=metrics["IAE_Pbh"],
        y0=y0,
    )
    plot_forecast(
        t,
        result["y_pinc"],
        result["y_rk"],
        result["pbh_pinc"],
        result["pbh_rk"],
        result["u"],
        T_HORIZON,
        out / "forecast.png",
    )
    print(f"wrote {out / 'forecast.npz'} and {out / 'forecast.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
