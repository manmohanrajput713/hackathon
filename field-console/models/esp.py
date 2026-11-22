import sys
import numpy as np
import torch
from common import ROOT

sys.path.insert(0, str(ROOT / "pinn"))
from data.PinnPredictor import PinnPredictor

def series():
    folder = ROOT / "pinn"
    predictor = PinnPredictor(str(folder / "model_adam_lbfgs/model.pt"))
    params = predictor.parameters
    with np.load(folder / "dataset/BCS_data_train_limitado_f_zc_pm_pr_opera.npz") as feed:
        pressures = np.column_stack((feed["x1"], feed["x2"]))
        inputs = np.column_stack((params.normalizar_u(feed["U"]), params.normalizar_x(pressures)))
        width = predictor.model.n_steps_in
        steps = np.arange(width, len(inputs), 5)
        windows = np.stack([inputs[i-width:i] for i in steps])
        torch.set_num_threads(2)
        predicted = predictor.model.predict(windows)[:, 0, 2]
        rate = (predicted * params.qc + params.qmin) * 3600
        reference = feed["x3"].reshape(-1)[steps] * 3600
        return dict(name="ESP test well", source="ESP simulation · saved pump checkpoint", unit="m³/h",
                    xLabel="Simulation time (s)", quantity="Total flow", model="Virtual flow meter",
                    points=np.column_stack([feed["t"][steps], rate, reference]).round(4).tolist())

