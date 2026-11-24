# SuddWatch: technical evaluation guide

This page maps each of the five technical rubrics to the evidence in this repository and the live deployment.
Every number below comes from a script you can run; the file that holds it is linked next to it.

| | |
|---|---|
| Live website | https://suddwatch-api.wittypond-9ed3ebe4.southafricanorth.azurecontainerapps.io/ |
| Live API (Swagger) | https://suddwatch-api.wittypond-9ed3ebe4.southafricanorth.azurecontainerapps.io/docs |
| Task | Will a South Sudan county's surface water exceed its own normal high level (training-period P75) in the window 13–24 days after the issue date? |
| Unit | one county × one 12-day Sentinel-1 window; 79 counties × 295 windows = 23,305 rows (2017-01 to 2026-08) |

## Verify in five minutes

```powershell
python -m venv .venv; .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest tests -q              # 13 tests: leakage, schema validation, API, web page
.venv\Scripts\python -m src.missingness_tests        # Little's MCAR, Cramér's V, MNAR check      (~1 min)
.venv\Scripts\python -m src.robustness_audit         # CIs, slices, ablation, stress, hold-out    (~1 min)
.venv\Scripts\python -m src.benchmark_api            # live latency (p50/p95/p99) and monthly cost
```

All data the scripts need is in the repository (`data/`), along with the trained model (`models/`). Only
`src.gee_extract` needs an Earth Engine login. Seeds are fixed (`SEED = 0`); two consecutive runs of
`src.robustness_audit` produced byte-identical result files.

---

## 1. Data hygiene, features and collinearity

| Checkpoint | Evidence | Result |
|---|---|---|
| Unit and target | [README](README.md), [Phase 1 report](phase1_strategy/Phase1_Strategy_Report.pdf) | county × 12-day window; target = water fraction in window *w*+2 > county's 2017–2023 P75 (floor 1 km²) |
| Cost framing | [cost_analysis_validation.csv](phase3_modeling/audit/cost_analysis_validation.csv) | expected cost per county-cycle when a missed flood costs *r* × a false alarm (*r* = 1–10); see rubric 3 |
| Formal missingness tests | [missingness_tests.json](phase3_modeling/audit/missingness_tests.json) | **Little's MCAR test rejects MCAR**: d² = 202, df = 23, p ≈ 10⁻³⁰, and it rejects within each sensor era too. **Cramér's V** of "water missing" with sensor era 0.36, state 0.33, month 0.04 |
| MAR vs MNAR | same file | A coarse logit finds that rainfall predicts gaps (p = 0.01; more rain, fewer gaps). The effect disappears with county and date fixed effects (p = 0.61; p = 0.98 in the 2022–23 gap years). The gaps follow the satellite acquisition plan, so the data are **MAR** on observed variables |
| Missing-data handling | [transformers.py](src/transformers.py) `SeasonalBlendImputer`, [imputation_experiment.csv](phase3_modeling/feedback/imputation_experiment.csv) | Fills gaps from past data only: the last observation, fading (τ = 120 d) toward the county × month training median. A missingness flag is kept as a feature. On a hidden year (2020), fill error is **−29 %** versus the static median |
| Outliers | [B_skewness_outliers_train.csv](phase1_strategy/audit/B_skewness_outliers_train.csv), `OutlierCapper` | Rainfall is clipped to mean ± 3 sd, with bounds **learned in `fit()` on training rows**. Water outliers are kept because they are the floods |
| Collinearity | [C_collinear_pairs_train.csv](phase1_strategy/audit/C_collinear_pairs_train.csv), [C_vif_train.csv](phase1_strategy/audit/C_vif_train.csv), [feedback_checks.json](phase3_modeling/feedback/feedback_checks.json), figures `fig3_1`, `fig5_1` | Checked with Spearman \|ρ\| ≥ 0.85, VIF > 10 and mutual information. The 12-month mean is dropped (an exact linear combination, VIF infinite). The six terrain metrics are collapsed to **one PCA index**, because the MI tie-break chose its winner in only 22 % of 200 county bootstraps |
| Shortcut / proxy screen | [ablation_validation.csv](phase3_modeling/audit/ablation_validation.csv), [permutation_importance_validation.csv](phase3_modeling/audit/permutation_importance_validation.csv) | A model given only state, month and acquisition metadata scores PR-AUC **0.697**, about climatology (0.666) and far below the full model (0.891). Permutation importance is led by the physical signals (water level 0.41, county threshold 0.13). The county ID is not a model feature: it only groups the imputer, with importance 0.006 |

## 2. Split integrity and leakage-free pipeline

| Checkpoint | Evidence | Result |
|---|---|---|
| Time-disjoint splits with purge | [train_model.py](src/train_model.py) `load()` | train < 2024-01-01 ≤ validation < 2025-07-01 ≤ test. The **2 windows before each boundary are purged**, equal to the forecast horizon, so no training label overlaps a validation or test feature window |
| CV that respects time | `cv_ap()` | `TimeSeriesSplit(5, gap = 2 windows × 79 counties)` for tuning, on training rows only |
| Training-only fitting | [pipeline_v2.py](src/pipeline_v2.py) | One sklearn `Pipeline` (imputers → features → clipper → `ColumnTransformer` with scalers, PCA, one-hot), re-fitted inside every CV fold. County thresholds use 2017–2023 only. Platt calibration is fitted on out-of-sample **training** predictions; the decision threshold is chosen on **validation**; the test set is scored once |
| Past-only features (no future leakage) | [tests/test_features_v2.py](tests/test_features_v2.py) | The panel is truncated at 3 cutoff dates; features computed on the truncated data must equal the full-data features up to each cutoff. Passes |
| Group-disjoint check | [county_holdout_validation.csv](phase3_modeling/audit/county_holdout_validation.csv) | Training on 4/5 of counties and scoring the unseen 1/5 (validation years) gives PR-AUC **0.874**, versus 0.891 for seen counties. The model beats persistence in all 5 folds, so it is not memorising counties |
| Disclosed deviation | [Response_to_Phase2_Feedback.md](phase3_modeling/Response_to_Phase2_Feedback.md) | Run 1's test result led to adding the county-threshold feature (selected on validation only), so the test set was **looked at twice**. This is disclosed; run 1 is archived |

Counties appear in all three splits by design: the deployed system forecasts these same 79 counties. The split
that matters is time, and the county hold-out above covers the group question.

## 3. Modeling strategy and evaluation rigour

**Baselines (validation PR-AUC, prevalence 0.369)**: [training_summary.json](phase3_modeling/results/training_summary.json)

| Model | PR-AUC |
|---|---|
| Climatology (county × month training rate) | 0.666 |
| Persistence (flooded now → flooded in 2 windows) | 0.835 |
| Logistic regression, same features | 0.851 |
| Gradient boosting, Phase 2 preprocessing | 0.856 |
| **Gradient boosting, Phase 3 preprocessing (deployed)** | **0.891** |
| + VH −26 dB feature (not adopted) | 0.889 |

**Test, with 95 % block-bootstrap intervals** (2,000 resamples of whole forecast cycles, 35 cycles, 2,662 rows):
[ci_bootstrap_test.csv](phase3_modeling/audit/ci_bootstrap_test.csv)

| Metric | Model | Persistence | Difference |
|---|---|---|---|
| PR-AUC | 0.925 [0.885, 0.957] | 0.919 [0.892, 0.946] | +0.006 [−0.013, +0.022] |
| Macro-F1 (threshold 0.44) | 0.888 [0.862, 0.911] | 0.901 [0.879, 0.921] | −0.013 [−0.024, −0.003] |
| False-alarm rate | 0.105 [0.067, 0.152] | 0.070 [0.043, 0.102] | |
| Precision@10 counties per cycle | 0.954 [0.909, 0.989] | 0.949 [0.909, 0.980] | +0.006 [−0.011, +0.023] |
| Calibration error (ECE) | 0.031 [0.023, 0.051] | n/a (not a probability) | |

Tuning-CV spread of the chosen setting: PR-AUC 0.704 ± 0.074 over 5 time folds.
[precision_at_k_test.csv](phase3_modeling/audit/precision_at_k_test.csv) gives k = 5, 10, 20.

**Reading this honestly.** On the test year the model and persistence have the same ranking skill (the PR-AUC
interval includes 0). With symmetric errors, the simple "flooded now" rule is slightly better on Macro-F1. Floods in the Sudd last
for months, so persistence is a very strong baseline. The model's value is elsewhere:

- **It is the only method that warns of new floods.** Persistence can never flag a county that is dry today. The model catches 32 % of onsets (40 of 126) at a 4 % false-alarm rate on dry-now counties (rubric 4).
- **It has the lowest expected cost once a miss costs more than a false alarm.** At the deployed threshold, the cost per county-cycle versus persistence is 0.181 vs 0.205 at *r* = 2, and 0.322 vs 0.400 at *r* = 5 ([cost_analysis_validation.csv](phase3_modeling/audit/cost_analysis_validation.csv)). Under a 3:1 policy the cost-optimal threshold would be 0.25, which is a decision for the agency using it.
- **Its probabilities are calibrated** (ECE 0.03), and **its threshold adapts** to the recently observed flood rate, the dynamic prior shift answering Phase 2 feedback.

The test PR-AUC is 0.927 in the training summary (calibrated scores) and 0.925 here (the deployed, prior-shifted scores);
both are computed on the same rows.

## 4. Error analysis and robustness audit

**Slices on the test year**: [error_slices_test.csv](phase3_modeling/audit/error_slices_test.csv), [phase3_robustness.png](figures/phase3_robustness.png)

| Slice | Rows | PR-AUC model / persistence | Floods caught (model / persistence) | False alarms (model) |
|---|---|---|---|---|
| Dry now (onsets) | 1,690 | 0.343 / 0.290 | **32 % / 0 %** | 4.1 % |
| Wet now (persistence/recession) | 972 | 0.953 / 0.947 | 98 % / 100 % | 94.9 % (118 recessions; most are not foreseen) |
| Western Equatoria (dry state) | 332 | 0.408 / 0.312 | 57 % / 29 % | 4.7 % |
| Western Bahr el Ghazal | 99 | 0.508 / 0.433 | 90 % / 30 % | 14.6 % |
| Jonglei | 371 | 0.959 / 0.963 | 92 % / 90 % | **19.8 %** |
| Upper Nile | 402 | 0.933 / 0.919 | 89 % / 88 % | **18.4 %** |
| Dry season (Nov–Apr) | 1,174 | 0.922 / 0.905 | 92 % / 86 % | **19.6 %** |

Also covered:
- **By year, 2017–2026** ([metrics_by_year.csv](phase3_modeling/results/metrics_by_year.csv)): the weakest year is 2021, PR-AUC 0.51, a low-flood year.
- **By sensor era** ([metrics_by_sensor_era.csv](phase3_modeling/results/metrics_by_sensor_era.csv)).
- **By county wetness** (in the slice file).

**Ablation (validation PR-AUC change when a feature group is neutralised)**: [ablation_validation.csv](phase3_modeling/audit/ablation_validation.csv)

| Group removed | Δ PR-AUC |
|---|---|
| Sentinel-1 water level (now and past) | −0.140 |
| County flood threshold | −0.023 |
| Terrain and static data | −0.002 |
| Rainfall (CHIRPS) | −0.001 |
| State / season / acquisition metadata | 0.000 / +0.005 / +0.006 |

**Stress tests on the deployed model (test PR-AUC change)**: [stress_tests_test.csv](phase3_modeling/audit/stress_tests_test.csv)

| Scenario | Δ PR-AUC | Recall |
|---|---|---|
| One radar pass missed | −0.016 | 0.85 |
| Two passes missed (24 days) | −0.037 | 0.82 |
| Six-month outage (like 2022–2025) | **−0.154** | **0.61** |
| ±20 % noise on water measurements | −0.011 | 0.87 |
| Rainfall bias −20 % / +20 % | −0.001 / −0.001 | 0.90 / 0.89 |
| Fewer Sentinel-1 scenes | +0.010 | 0.89 |

**Concrete deployment limits** (the first one is enforced by the API):

1. **Radar data age.** Each forecast returns `water_observation_age_days` and a `data_quality` flag: `ok` ≤ 12 d, `degraded` ≤ 24 d, `low confidence` ≤ 60 d, `unreliable` beyond. These cut-offs come from the stress tests.
2. **Onset detection is limited.** 32 % of new floods are caught. The model is strongest at confirming that a flood continues.
3. **False alarms exceed the 15 % target** in Jonglei (19.8 %), Upper Nile (18.4 %) and the dry season (19.6 %). Users there should expect about one false alert in five non-flood cases.
4. **Dry counties have weak ranking** (Western Equatoria PR-AUC 0.41). There are few events there, so alerts are rare and less reliable.
5. **Rainfall adds almost nothing at this horizon** (ablation, stress test). Forecasts rest on the radar signal, so a long radar outage degrades them towards climatology.
6. **Thresholds come from 2017–2023.** The 2024–26 flood rate is twice the training rate. The dynamic threshold adapts the decision, but the county thresholds themselves will need re-estimation.
7. **It has not been validated against ground reports beyond the satellite label.** The label is Sentinel-1 water, not people affected.

## 5. Code reproducibility and cloud deployment

| Checkpoint | Evidence |
|---|---|
| Modular repo | `src/` has one module per step (`gee_extract`, `build_panel`, `features_v2`, `transformers`, `pipeline_v2`, `train_model`, `calibration`, `feedback_checks`, `missingness_tests`, `robustness_audit`, `benchmark_api`, `export_latest`, `build_web_assets`). `api/` holds the service, `tests/` the tests, `deploy/` the deployment script |
| Runnable README, pinned environment | [README.md](README.md) §3 and §6; `requirements.txt` and `api/requirements.txt` pin exact versions (Python 3.14) |
| Seeds and determinism | `SEED = 0` for tuning, models, bootstrap and permutations; repeated audit runs are byte-identical |
| Container | [Dockerfile](Dockerfile): `python:3.14-slim`, **non-root user**, whitelist [.dockerignore](.dockerignore) (only the API, the model and the 4 source files it imports) |
| Live endpoint | Azure Container Apps, South Africa North; 0.5 vCPU / 1 GiB, scales 0 → 1. Version 1.2.0 |
| Schema checks | Pydantic models in [api/main.py](api/main.py) check ranges (`obs_frac` 0–1, `month` 1–12, threshold > 0), require fields, check types and cap a batch at 1–500 rows. `tests/test_api.py::test_schema_rejects_bad_input` checks that bad input returns 422 |
| Latency and cost | [latency_cost.json](phase3_modeling/deployment/latency_cost.json): see the table below |
| Logging | One JSON log line per request (method, path, status, ms) goes to Azure Log Analytics. Every response carries an `X-Process-Time-ms` header |
| Secrets | No secrets in the repo or its history (scanned). The app needs no secrets at runtime. The only secret is the registry pull password, held as an encrypted Container Apps secret (`passwordSecretRef`), not in code or environment variables |
| Redeploy | [deploy/azure_deploy.ps1](deploy/azure_deploy.ps1) `-Tag x.y.z`. The header documents the student-subscription workarounds |

**Latency of the live endpoint** (`src.benchmark_api`, 40 warm requests per endpoint, 2026-11-25). The client time
includes the internet round trip from the machine running the benchmark; the server time is the API's own processing.

| Endpoint | Client p50 / p95 / p99 (ms) | Server p50 / p95 (ms) |
|---|---|---|
| `GET /health` | 95 / 103 / 107 | 0.5 / 0.6 |
| `GET /forecast/latest/{county}` | 136 / 247 / 330 | 28 / 31 |
| `GET /forecast/latest` (79 counties) | 130 / 291 / 329 | 30 / 31 |
| `POST /predict` (1 row) | 122 / 184 / 286 | 27 / 29 |
| `POST /predict` (79 rows) | 158 / 187 / 237 | 60 / 61 |

**Cold start.** After the app has scaled to zero, the first request takes **6.2 s** (measured 2026-11-25): a container
starts and loads the model, and `/health` reports `model_load_ms`. Later requests are warm. For a live demo, run
`az containerapp update -g suddwatch-rg -n suddwatch-api --min-replicas 1` beforehand and set it back to 0 afterwards.

**Cost** (Azure retail prices for South Africa North, fetched by the benchmark; free monthly grant applied):

| Scenario | USD / month |
|---|---|
| As deployed: scale to zero, demo-level traffic | **5.00** (container registry only; compute within the free grant) |
| Scale to zero, heavy use (8 h/day active, 100k requests/day) | 12.96 |
| One replica always on (`min-replicas 1`) | 15.69 |

Paid from the Azure for Students credit; actual spend is visible in Azure portal → Cost Management.
