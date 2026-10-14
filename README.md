# SuddWatch — County-Level Flood-Risk Forecasting for South Sudan's Sudd Wetlands

**04-652 AI System Design · Assignment 2: Data Preprocessing · Carnegie Mellon University Africa**
Author: Jok Akech Atem Mabior · Repository: https://github.com/Super-Talen/SuddWatch

| | |
|---|---|
| **Task type** | **Binary Classification** (supervised, short-horizon spatio-temporal forecasting) |
| **Unit of analysis** | one row = one South Sudan county × one 12-day Sentinel-1 window |
| **Target (Y)** | `y_flood_2_4wk` = 1 if the county's surface-water fraction 13–24 days after the issue date exceeds that county's own 75th percentile (computed on training years only) |
| **Dataset size** | **N = 23,305 rows × P = 26 columns** (20 candidate features + identifiers, split tag and target) |
| **Period** | issue windows 2017-01-12 → 2026-08-29 · 79 counties (COD-AB admin2) |
| **SDGs** | SDG 13 (Climate Action) — linked to SDG 2 (Zero Hunger), SDG 1 (No Poverty) |

## 1. Problem statement

Recurrent flooding in the Sudd wetlands now multiplies South Sudan's hunger and displacement crises.
By October 2025 floods affected over one million people in 29 counties, 87 % of them in Jonglei and
Unity, repeatedly cutting food access to famine-risk counties such as Nasir and Fangak. Existing
satellite products map flood extent *after* it happens or forecast regional extent months ahead; none
tells humanitarian planners **which county will flood in the next 2–4 weeks**, which is the lead time
needed to pre-position food, medicine and boats before roads and river routes close.

SuddWatch learns, from ten years of Sentinel-1 radar flood history, rainfall and terrain, to predict
whether each county's surface-water extent will exceed its own normal high-water level 13–24 days
after the forecast is issued.

## 2. Dataset overview

| Source | Role | Access | Link |
|---|---|---|---|
| Copernicus **Sentinel-1** SAR GRD (VV, IW) | label source + autoregressive features (surface-water fraction per county per window) | Google Earth Engine `COPERNICUS/S1_GRD` | https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD |
| **CHIRPS** v2.0 daily rainfall | rainfall totals (12/36/90 days) and 90-day anomaly vs 2001–2016 | GEE `UCSB-CHG/CHIRPS/DAILY` | https://www.chc.ucsb.edu/data/chirps |
| **SRTM** 1-arc-second DEM | elevation mean/std, slope | GEE `USGS/SRTMGL1_003` | https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003 |
| **MERIT Hydro** | height above nearest drainage (HAND), upstream area → topographic wetness index | GEE `MERIT/Hydro/v1_0_1` | https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1 |
| **JRC Global Surface Water** v1.4 | permanent-water share, long-term water occurrence | GEE `JRC/GSW1_4/GlobalSurfaceWater` | https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_GlobalSurfaceWater |
| **COD-AB** South Sudan admin boundaries (OCHA) | 79 county polygons, join key `adm2_pcode` | HDX | https://data.humdata.org/dataset/cod-ab-ssd |
| **UNOSAT** VIIRS water extents 2022–2024 | *independent validation* of the Sentinel-1 label (not a model input) | HDX | https://data.humdata.org/dataset/satellite-detected-water-extents-between-20-and-24-april-2022-over-south-sudan |
| IOM DTM flood locations (2019) | reference only (downstream humanitarian cross-referencing) | HDX | https://data.humdata.org/dataset/south-sudan-flood-locations |

**Target definition and class balance.** Of 21,160 labelled rows, the positive class (Y=1) is 19.2% in train (4.21:1), 36.4% in validation and 36.8% in test. 2,145 rows have no label because the target window was not imaged by Sentinel-1 (MAR, acquisition plan); they are excluded from fitting, never imputed. Mitigation: `class_weight="balanced"`, validation-tuned decision threshold, PR-AUC/F1/recall metrics. Full audit: [`phase1_strategy/Phase1_Strategy_Report.pdf`](phase1_strategy/Phase1_Strategy_Report.pdf).

## 3. Repository structure

```
SuddWatch/
├── .gitignore                 excludes .venv, credentials, large raw binaries, EE chunk caches
├── requirements.txt           pinned package versions
├── README.md
├── data/
│   ├── raw/                   original / as-extracted data (unaltered)
│   │   ├── gee/               Earth Engine county-level extracts (Sentinel-1, static terrain)
│   │   ├── ssd_admin_boundaries/   COD-AB admin0-2 GeoJSON
│   │   ├── suddwatch_panel_unprocessed.csv   assembled panel audited in Phase 1
│   │   └── county_thresholds_train_only.csv  per-county P75 thresholds (training years only)
│   └── processed/             Phase 2: cleaned, encoded, scaled feature matrix
├── phase1_strategy/
│   ├── Phase1_Strategy_Report.pdf   (IEEE conference format)
│   └── audit/                 every audit table behind the report (CSV/JSON)
├── phase2_implementation/
│   ├── notebooks/pipeline_execution.ipynb   (Phase 2)
│   └── Phase2_Final_Report.pdf              (Phase 2)
├── src/
│   ├── __init__.py
│   ├── gee_extract.py         Earth Engine extraction (Sentinel-1, CHIRPS, SRTM/MERIT/JRC)
│   ├── build_panel.py         joins, lags, train-only thresholds, target
│   ├── audit.py               Phase 1 audit (shape, types, missingness, skew, outliers, associations, VIF)
│   ├── transformers.py        custom scikit-learn transformers (county-median imputer, IQR capper, cyclical encoder)
│   └── pipeline_builder.py    ColumnTransformer + Pipeline + temporal CV definitions
└── figures/                   300-DPI diagnostic figures (fig1_1 … fig5_1)
```

## 4. Reproduce

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt          # Windows (use .venv/bin/pip on macOS/Linux)
.venv\Scripts\earthengine authenticate                  # one-time Google Earth Engine login
# edit PROJECT in src/gee_extract.py to your own registered Earth Engine Cloud project
.venv\Scripts\python -m src.gee_extract all             # ~1 h on the free Community tier; cached & resumable
.venv\Scripts\python -m src.build_panel                 # -> data/raw/suddwatch_panel_unprocessed.csv
.venv\Scripts\python -m src.audit                       # -> phase1_strategy/audit/*, figures/fig1_*
```

The UNOSAT GeoPackage (67 MB) is excluded from Git because of size; download it from the HDX link
above into `data/raw/` to re-run the label validation.

## 5. Pipeline summary (Phase 1 plan → Phase 2 implementation)

1. **Split first** (temporal): train = issue dates < 2024-01-01, validation = 2024-01-01 – 2025-06-30,
   test = 2025-07-01 – 2026-08-31 (covers the 2025 flood season). Inner tuning uses
   `TimeSeriesSplit` with a 2-window purge gap because Y looks two windows ahead.
2. **Impute** Sentinel-1 gaps (MAR — satellite acquisition plan) with county-level *training* medians
   plus missing-indicator flags.
3. **Cap** heavy-tailed features at training IQR fences (features only; labels are never capped).
4. **Transform & scale**: log1p + StandardScaler for right-skewed features, MinMax for [0,1] fractions,
   StandardScaler for symmetric ones; one-hot for the 11 states; sin/cos for month.
5. **Collinearity**: drop one feature of every pair with |Spearman ρ| ≥ 0.85 or VIF > 10 (keep the one
   with higher mutual information with Y).
6. **Model** (later assignments): gradient-boosted trees, logistic-regression baseline, `class_weight="balanced"`.

All fitted statistics live inside one scikit-learn `Pipeline` that is fitted on `X_train` only.

## AI assistance

Claude (Anthropic) assisted with the Google Earth Engine extraction code, the panel-construction and audit scripts, and
drafting the Phase 1 report. All data were extracted from the cited public sources with the author's Earth Engine account,
every reported number is generated directly by the audit scripts, and all design decisions were reviewed and approved by
the author.
