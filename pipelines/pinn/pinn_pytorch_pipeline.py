"""Thin entry point.

The implementation lives in data/ and the root scripts
main_build_dataset.py, main_train_adam.py, main_train_lbfgs.py, and main_forecast.py.
"""

from data.BuildDataset import BuildingDataset
from data.forecast import free_run
from data.pinn_BCS import pinn_vfm


def main(argv=None):
    from main_forecast import main as forecast_main
    return forecast_main(argv)


if __name__ == "__main__":
    main()
