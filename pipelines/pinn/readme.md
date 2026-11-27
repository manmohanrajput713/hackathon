PINN VFM

Physics-informed virtual flow meter for an ESP-lifted oil well. Training runs in PyTorch. The LSTM width is 10. Multiplicative noise is seeded with NumPy and PyTorch seed 0, so a rebuild of the windows is reproducible.

Old TensorFlow `model.h5` files and SavedModel folders are not loaded. Retrain and load `model.pt`. After training, `model_adam_200/` and `model_adam_lbfgs/` hold that checkpoint and the pickled histories.

## Setup

```bash
pip install -r requirements.txt
export MPLBACKEND=Agg
```

## Procedure

Rebuild `dataset01.pk` before training. An older pickle of this file cannot be loaded.

1. Build the identification dataset from `dataset/BCS_data_train_limitado_f_zc_pm_pr_ident.npz`:

```bash
python main_build_dataset.py
```

`split_point = int(0.7 * len(x1))` indexes the windowed samples. Batch size is 500. Batches are not shuffled, because the time derivative is taken along the batch axis.

2. Adam for 200 epochs at learning rate 0.01, width 10, loss weights 1, 1, 1. This writes `model_adam_200/model.pt`:

```bash
python main_train_adam.py
```

A short run can pass `--epochs 2`. The default stays 200.

3. L-BFGS for 100 iterations from that checkpoint, with weights `lamb_l1 = 2.0`, `lamb_l2 = 1.9`, `lamb_l3 = 0.01`. This writes `model_adam_lbfgs/model.pt`:

```bash
python main_train_lbfgs.py
```

A short run can pass `--iterations 2`. The default stays 100. Watch the loss terms before changing the weights.

4. Free-run forecast. Only predicted bottom-hole pressure and wellhead pressure are fed back. Flow rate is not fed back. Future exogenous inputs come from the data. The default horizon is 800 steps:

```bash
python main_build_dataset_opera.py
python main_forecast.py --steps 800
```

`main_build_dataset_opera.py` reads `dataset/BCS_data_train_limitado_f_zc_pm_pr_opera.npz` with `split_point = int(0.98 * len(x1))`.

`PINN_VFM_Master_Guide.ipynb` is the training and forecast walkthrough. `main_forecast.py` is the free-run entry point.

```bash
pytest
```

## Citation

If this code helps you, you are encouraged to cite the following paper:

```
@article{FRANKLIN2022,
title = {A Physics-Informed Neural Networks (PINN) oriented approach to flow metering in oil wells: an ESP lifted oil well system as a case study},
journal = {Digital Chemical Engineering},
pages = {100056},
year = {2022},
issn = {2772-5081},
doi = {https://doi.org/10.1016/j.dche.2022.100056},
url = {https://www.sciencedirect.com/science/article/pii/S2772508122000461},
author = {Taniel S. Franklin and Leonardo S. Souza and Raony M. Fontes and Márcio A.F. Martins},
keywords = {soft sensor, Physics-Informed Neural Networks, electrical submersible pump, virtual flow meter, recurrent neural network},
}
```
