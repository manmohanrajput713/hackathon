# Getting Started

## Installing pywaterflood

```bash
pip install pywaterflood
```

Pywaterflood 0.3.4 history-matches CRM with PyTorch. A working install needs
`numpy`, `pandas`, `scipy`, and `torch` (pulled in automatically).

## Smallest useful script

```python
import pandas as pd
from pywaterflood import CRM

gh_url = (
    "https://raw.githubusercontent.com/frank1010111/pywaterflood/master/testing/data/"
)
prod = pd.read_csv(gh_url + "production.csv", header=None).to_numpy()
inj = pd.read_csv(gh_url + "injection.csv", header=None).to_numpy()
time = pd.read_csv(gh_url + "time.csv", header=None).to_numpy()[:, 0]

crm = CRM(primary=True, tau_selection="per-pair", constraints="up-to one")
crm.fit(prod, inj, time, epochs=400, seed=0)
q_hat = crm.predict()
residuals = crm.residual()

connectivity = pd.DataFrame(crm.gains).rename_axis(index="Producer", columns="Injector")
connectivity.style.highlight_max(axis=1).format("{:.2f}")
```

`prod` is `(n_time, n_producers)`, `inj` is `(n_time, n_injectors)`, `time` is
`(n_time,)`. `crm.gains` is `(n_producers, n_injectors)`.

`epochs=400` is enough to see structure on this sample. The default is 2000.

## What to read next

- {doc}`../user-guide/index` — map of CRM, geometry, Buckley-Leverett, aquifer
- {doc}`../user-guide/crm-in-practice` — data rules, PINN knobs, geometry seed,
  forecasting
- {doc}`../user-guide/7-minutes-to-pywaterflood` — plotted history match
