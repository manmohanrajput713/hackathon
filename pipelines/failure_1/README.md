# ESP predictive maintenance (Abdalla et al., 2022)

Reproduction of the Knowledge Discovery in Databases pipeline in Abdalla, Samara, Perozo, Carvajal, and Jaeger, "Machine Learning Approach for Predictive Maintenance of the Electrical Submersible Pumps (ESPs)", *ACS Omega* 2022, 7, 17641–17651. [doi:10.1021/acsomega.1c05881](https://doi.org/10.1021/acsomega.1c05881).

The open-access PDF is stored at `docs/acsomega.1c05881.pdf` (CC BY-NC-ND 4.0). The code here is an implementation of the published method on the Energy A.I. Hackathon 2023 telemetry from [PGEHackathon/data2023](https://github.com/PGEHackathon/data2023).

## What the pipeline does

1. **Collection.** Download the paper and the hackathon tables, then check their columns.
2. **Preprocessing.** Treat each `(Well_ID, AL_Key)` installation as one ESP. Label the failure day and the seven days before it. Drop non-ESP rows, incomplete sensor rows, non-positive oil, unphysical values, and Tukey outliers.
3. **Features.** Standardize the eight sensor levels on the training installations, take the one-day difference, and project the 16 columns onto eight principal components.
4. **Classifier.** Search the Table 5 ranges with Optuna. The objective is the mean ROC-AUC of a 10-fold split that keeps every installation inside one fold.
5. **Evaluation.** Figure 8 is that 10-fold ROC. Figure 9 and Table 6 are the holdout precision-recall curve and the classification report.

The positive class is the paper's "7 days or less pre-event" label: the workover day together with the preceding seven daily rows.

## Sensor map

The paper's vector is frequency, discharge pressure, intake pressure, wellhead pressure, wellhead temperature, motor temperature, casing-head pressure, and drive current. This feed contains five of those measurements directly. The other three paper channels are not in the files, so the eight-column vector uses the measured pressure, temperature, and electrical tags that fill the same roles.

| Code | Hackathon column | Role in the paper vector |
| --- | --- | --- |
| FRQ | `ESP Data - Output Frequency` | Pump frequency (Hz) |
| PDP | `Discharge_Pressure` | Discharge pressure (psi) |
| PIP | `ESP Data - Intake Pressure` | Intake pressure (psi) |
| PAVG | `Pump_Average_Pressure` | Pressure level used in the wellhead-pressure slot (psi) |
| MT | `ESP Data - Motor Winding Temperature` | Motor temperature (F) |
| ESPT | `ESP_Temperature` | Temperature used in the wellhead-temperature slot (F) |
| CURRENT | `ESP Data - Drive Current` | Drive current (A) |
| VOLT | `ESP Data - Drive Voltage` | Drive voltage used in the casing-pressure slot (V) |

Oil rate is the zero-production filter. It is not a PCA input. Static pump-design fields are merged onto the modeling table and are not PCA inputs either.

## Labels

`dailyData.csv` has no calendar column. Row order within an installation is the daily index. The published `wellData.csv` also has no `Status` or `Failure_Type` field. The hackathon README states that installations absent from `solution.csv` have failed, so the last daily row of those installations is the workover day.

| Cohort | Rule | Supervised label |
| --- | --- | --- |
| `failed` | Absent from `solution.csv` | 1 when `0 <= days_to_event <= 7`, otherwise 0 |
| `survivor` | `Fail in 30 days` is 0 | 0 on every observed day |
| `prospective` | `Fail in 30 days` is 1 | Excluded. The failure has not occurred inside the observed series, so a 7-day label is not identified. |

The holdout is 30% of the labeled installations, stratified by `failed` and `survivor`. The same installation is never in both sides. The scaler, the PCA, and XGBoost are fit on the training installations. Cleaning fences are estimated on the whole corpus, which is the paper's order: outlier removal is finished before the split.

## Formulas

Tukey fences, with `IQR = Q3 - Q1` and `k = 1.5`:

```text
lower = Q1 - k IQR
upper = Q3 + k IQR
```

Standardization uses the population standard deviation so the columns have unit variance. The paper's sentence says "dividing by the variance" and, in the same sentence, requires unit variance. Unit variance is the division by `sigma`.

```text
z = (x - mean) / sigma
```

Moving difference inside an installation, kept only when the daily index increases by 1:

```text
diff(z_t) = z_t - z_{t-1}
```

Covariance, principal-component scores, and correlation loadings:

```text
Sigma = (X_c' X_c) / (n - 1)
Y = X W
L = W diag(sqrt(lambda))
```

`L` stores the correlations between the standardized variables and the component scores. XGBoost minimizes the regularized second-order objective of Chen and Guestrin (2016). With gradient `g`, Hessian `h`, L2 penalty `lambda`, and split penalty `gamma`:

```text
similarity = (sum g)^2 / (sum h + lambda)
leaf weight = -(sum g) / (sum h + lambda)
gain = 0.5 (similarity_left + similarity_right - similarity_parent) - gamma
```

Precision, recall, true positive rate, and false positive rate:

```text
precision = TP / (TP + FP)
recall = TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
F1 = 2 precision recall / (precision + recall)
```

## Imbalance and the decision threshold

Table 3 of the paper is heavily imbalanced, and the authors leave SMOTE as future work. Training uses XGBoost `scale_pos_weight = n_negative / n_positive` on each training fold. The reported operating point is the threshold that maximizes the positive-class F1 on the out-of-fold training scores. That threshold is frozen before the holdout is scored.

Table 5 prints `max_depth` in `{2, ..., 10}`, `min_child_weight` log-uniform on `[1e-10, 1e10]`, `gamma` in `{0, ..., 5}`, `colsample_bytree` uniform on `[0, 1]`, and `learning_rate` uniform on `[0, 0.1]`. A child-weight above the sum of the Hessians cannot split a node, and a zero column fraction or learning rate does not define an update. The study therefore samples the interior of that table: `min_child_weight` log-uniform on `[1e-2, 20]`, `colsample_bytree` uniform on `[0.5, 1]`, and `learning_rate` uniform on `[0.02, 0.1]`. `n_estimators` is fixed at 200. Both the published box and the sampled interior are stored in `results/table5_best_hyperparameters.json`.

## Layout

```text
failure_1/
├── docs/acsomega.1c05881.pdf
├── dataset/dailyData.csv
├── dataset/wellData.csv
├── dataset/solution.csv
├── dataset/processed_esp_data.parquet
├── src/
├── results/
├── tests/test_pipeline.py
├── main.py
├── requirements.txt
└── README.md
```

`solution.csv` is part of the hackathon repository. The failure rule cannot be applied without it. `results/processing_audit.json` records row counts, IQR fences, and the PCA variance actually obtained on this feed. `results/fig1_kdd_workflow.png` and `results/fig5_pca_geometry.png` are drawn schematics of the paper's Figure 1 and Figure 5.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
pytest -q
```

`python main.py --skip-download` reuses the files already in `dataset/` and `docs/`. `python main.py --n-trials 20` sets the Optuna budget. The default budget is 20 trials of 10 group folds and takes several minutes.

The published mean ROC-AUC of 0.95 and the test F1 of 0.71 were measured on the authors' polymer-flood archive, where the leading principal components already separated pre-workover days. On this hackathon feed the mean shift between ordinary days and the 7-day window is small on every retained channel. The eight principal components still hold about 98% of the variance, and the pipelined classifier is scored with the same ROC, precision-recall, and F1 definitions. Read those values from `results/table5_best_hyperparameters.json` and `results/table6_classification_report.csv` after `python main.py`.

## Tests

`tests/test_pipeline.py` checks the fence algebra, the covariance and loading identities, the one-day difference, the hourly median resample, the 7-day label window, the removal of zero oil and unphysical values, the disjoint group holdout, and that PCA loadings do not change when the holdout installations are corrupted.
