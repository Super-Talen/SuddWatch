# SuddWatch dataset

> **This is a panel time-series dataset.** It follows **79 South Sudan counties** through time, one observation per county
> **every 12 days** (one Sentinel-1 satellite revisit window) from **2017-01-12 to 2026-08-29** — **295 time steps per county**,
> **23,305 rows × 26 columns** in total. It is stored in *long format*: each row is **one county at one point in time**, so the
> file looks like an ordinary table, but every county is a time series and the rows are linked through time.

![The panel drawn as time series](../figures/data_overview_panel_timeseries.png)

## 1. How the panel is organised

```
                 time  →   window 1     window 2     ...   window 295
                           2017-01-12   2017-01-24         2026-08-29
 county  Ayod   (SS0302)   row          row          ...   row
    ↓    Duk    (SS0305)   row          row          ...   row
         ...               ...          ...          ...   ...
         Juba   (SS0101)   row          row          ...   row
                           = 79 counties × 295 windows = 23,305 rows   (balanced: every county has all 295 windows)
```

How the time dimension is used (details in the Phase 1 and Phase 2 reports):

| Time-series element | Where it appears |
|---|---|
| **Lagged values** of the series | `water_frac_lag1` (t − 1), `water_frac_lag2` (t − 2); Phase 2 adds the change `water_frac − water_frac_lag1` |
| **Look-back windows** of an external driver | rainfall over the 12, 36 and 90 days ending on the issue date, and its anomaly vs 2001–2016 |
| **Forecast target in the future** | `y_flood_2_4wk` is the county's water state **two windows ahead** (13–24 days after the forecast date), never the present |
| **Seasonality** | `month`, encoded as sin/cos in the pipeline |
| **Chronological validation** | train 2017–2023 → validation 2024 – mid-2025 → test mid-2025 – 2026, no shuffling; `TimeSeriesSplit` with a 2-window gap inside training |
| **Time-dependent missingness** | Sentinel-1B failed in Dec 2021, leaving some counties un-imaged until Sentinel-1C/D (mid-2025) — diagnosed as MAR |
| **Drift over time** | flood exceedances rise year by year after 2022 (Phase 2 report, Section V-C) |

**Why the forecasting task is framed as classification on the panel:** each county series is short (295 steps), has satellite gaps, and is
driven by external factors (rainfall, terrain). Building lag and look-back features and predicting the state two windows ahead lets a
gradient-boosted classifier learn from all 79 series at once — a standard approach for panels of many short series. A sequence model
(ConvLSTM) is planned as a comparison baseline in the project proposal.

## 2. One county followed through time (snapshot)

Ayod County, Jonglei — consecutive 12-day windows. Note that `water_frac_lag1` on each row equals the previous row's `water_frac`.
The complete series (all 295 windows) is in [`sample_ayod_county_timeseries.csv`](sample_ayod_county_timeseries.csv), which GitHub
shows as a searchable table.

| adm2_name | adm1_name | issue_date | obs_frac | water_frac | water_frac_lag1 | rain_36d_mm | rain_90d_anom_mm | month | y_flood_2_4wk |
|---|---|---|---|---|---|---|---|---|---|
| Ayod | Jonglei | 2024-07-10 | 1 | 0.0171 |  | 145.3 | -7.23 | 7 | 1 |
| Ayod | Jonglei | 2024-07-22 | 1 | 0.0193 | 0.0171 | 174 | 10.26 | 7 | 1 |
| Ayod | Jonglei | 2024-08-03 | 1 | 0.0179 | 0.0193 | 201.3 | 20.56 | 8 | 1 |
| Ayod | Jonglei | 2024-08-15 | 1 | 0.0212 | 0.0179 | 222.3 | 47.38 | 8 | 1 |
| Ayod | Jonglei | 2024-08-27 | 1 | 0.0208 | 0.0212 | 201 | 11.49 | 8 | 1 |
| Ayod | Jonglei | 2024-09-08 | 1 | 0.0206 | 0.0208 | 181.4 | -9.354 | 9 | 1 |
| Ayod | Jonglei | 2024-09-20 | 1 | 0.0163 | 0.0206 | 145.2 | 15.23 | 9 | 1 |
| Ayod | Jonglei | 2024-10-02 | 1 | 0.0184 | 0.0163 | 146.9 | 37.04 | 10 | 1 |
| Ayod | Jonglei | 2024-10-14 | 1 | 0.0278 | 0.0184 | 137.3 | 10.32 | 10 | 1 |
| Ayod | Jonglei | 2024-10-26 | 1 | 0.0256 | 0.0278 | 122.5 | 18.34 | 10 | 1 |

## 3. Many counties at one point in time (snapshot)

The same panel sliced the other way: several counties on one issue date (2025-09-15).

| adm2_name | adm1_name | issue_date | obs_frac | water_frac | water_frac_lag1 | rain_36d_mm | rain_90d_anom_mm | month | y_flood_2_4wk |
|---|---|---|---|---|---|---|---|---|---|
| Abyei | Abyei Region | 2025-09-15 | 1 | 0 | 0 | 201 | 14.31 | 9 | 0 |
| Juba | Central Equatoria | 2025-09-15 | 0.5896 | 0.003 | 0.0028 | 163.4 | -44.13 | 9 | 1 |
| Kajo-keji | Central Equatoria | 2025-09-15 | 0.9998 | 0.0002 | 0.0001 | 175.8 | -54.47 | 9 | 0 |
| Lainya | Central Equatoria | 2025-09-15 | 0.3142 |  |  | 167.6 | -69.62 | 9 | 0 |
| Morobo | Central Equatoria | 2025-09-15 | 1 | 0 | 0 | 173.8 | -85.83 | 9 | 0 |
| Terekeka | Central Equatoria | 2025-09-15 | 0.3051 |  |  | 189 | 9.839 | 9 | 0 |

## 4. Data dictionary (all 26 columns)

| column | role | type | unit | meaning | source | missing % |
|---|---|---|---|---|---|---|
| `adm2_pcode` | ID | nominal | – | County P-code; identifies the series a row belongs to | OCHA COD-AB | 0.00 |
| `adm2_name` | ID | nominal | – | County name | OCHA COD-AB | 0.00 |
| `adm1_name` | feature | nominal (11) | – | State the county belongs to | OCHA COD-AB | 0.00 |
| `window_start` | time index | datetime | date | First day of the 12-day Sentinel-1 window | derived | 0.00 |
| `issue_date` | time index | datetime | date | Last day of the window = the date the forecast is issued | derived | 0.00 |
| `split` | meta | nominal (3) | – | train (< 2024-01-01), validation (< 2025-07-01) or test, by issue date | derived | 0.00 |
| `n_scenes` | feature | discrete | count | Sentinel-1 scenes acquired over South Sudan in the window | Sentinel-1 | 0.00 |
| `obs_frac` | feature | continuous [0,1] | fraction | Share of the county imaged by Sentinel-1 in the window | Sentinel-1 | 0.00 |
| `water_frac` | feature | continuous [0,1] | fraction | Share of observable county area that is surface water (VV < −18 dB, HAND < 20 m); missing if < 50 % of the county was imaged | Sentinel-1, MERIT Hydro | 9.20 |
| `water_frac_lag1` | feature | continuous [0,1] | fraction | `water_frac` of the previous window (t − 1) | derived | 9.54 |
| `water_frac_lag2` | feature | continuous [0,1] | fraction | `water_frac` two windows back (t − 2) | derived | 9.88 |
| `rain_12d_mm` | feature | continuous | mm | County-mean rainfall summed over the 12 days ending on the issue date | CHIRPS | 0.00 |
| `rain_36d_mm` | feature | continuous | mm | Same, over 36 days | CHIRPS | 0.00 |
| `rain_90d_mm` | feature | continuous | mm | Same, over 90 days | CHIRPS | 0.00 |
| `rain_90d_anom_mm` | feature | continuous | mm | 90-day rainfall minus its 2001–2016 average for the same calendar day | CHIRPS | 0.00 |
| `county_area_km2` | feature (static) | continuous | km² | County area | OCHA COD-AB | 0.00 |
| `elev_mean_m` | feature (static) | continuous | m | Mean elevation | SRTM | 0.00 |
| `elev_std_m` | feature (static) | continuous | m | Standard deviation of elevation (terrain roughness) | SRTM | 0.00 |
| `slope_mean_deg` | feature (static) | continuous | degrees | Mean terrain slope | SRTM | 0.00 |
| `hand_mean_m` | feature (static) | continuous | m | Mean height above nearest drainage | MERIT Hydro | 0.00 |
| `hand_lt5m_frac` | feature (static) | continuous [0,1] | fraction | Share of the county less than 5 m above the nearest drainage | MERIT Hydro | 0.00 |
| `twi_mean` | feature (static) | continuous | – | Mean topographic wetness index ln(a / tan β) | MERIT Hydro | 0.00 |
| `perm_water_frac` | feature (static) | continuous [0,1] | fraction | Share of the county that is water ≥ 80 % of the time (1984–2021) | JRC Global Surface Water | 0.00 |
| `jrc_water_occurrence_pct` | feature (static) | continuous | % | Mean long-term surface-water occurrence | JRC Global Surface Water | 0.00 |
| `month` | feature | discrete (cyclical) | 1–12 | Month of the issue date | derived | 0.00 |
| `y_flood_2_4wk` | **target** | binary | 0/1 | 1 if `water_frac` two windows ahead (13–24 days after the issue date) exceeds the county's own 75th percentile from training years (floor 1 km²) | derived | 9.20 |

## 5. Full descriptive statistics (all 23,305 rows)

| column | count | missing | mean | std | min | 5% | 25% | 50% | 75% | 95% | max | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `n_scenes` | 2.33e+04 | 0 | 54.17 | 16.99 | 22 | 30 | 36 | 60 | 66 | 74 | 113 | 0.07454 |
| `obs_frac` | 2.33e+04 | 0 | 0.8988 | 0.2689 | 0 | 0.004946 | 0.9996 | 1 | 1 | 1 | 1 | -2.625 |
| `water_frac` | 2.116e+04 | 2145 | 0.00514 | 0.0174 | 0 | 0 | 1.384e-05 | 0.0004573 | 0.003911 | 0.02137 | 0.7409 | 11.67 |
| `water_frac_lag1` | 2.108e+04 | 2224 | 0.005118 | 0.01734 | 0 | 0 | 1.384e-05 | 0.0004585 | 0.003908 | 0.02124 | 0.7409 | 11.76 |
| `water_frac_lag2` | 2.1e+04 | 2303 | 0.005102 | 0.01731 | 0 | 0 | 1.384e-05 | 0.0004601 | 0.003904 | 0.02114 | 0.7409 | 11.82 |
| `rain_12d_mm` | 2.33e+04 | 0 | 31.94 | 29.93 | 0 | 0 | 0.9868 | 28.24 | 55.81 | 83.67 | 170.3 | 0.5429 |
| `rain_36d_mm` | 2.33e+04 | 0 | 95.43 | 81.47 | 0 | 0 | 8.257 | 91.54 | 166.8 | 226.7 | 361.2 | 0.2669 |
| `rain_90d_mm` | 2.33e+04 | 0 | 235.7 | 181 | 0 | 0.7513 | 54.13 | 227 | 395.2 | 526.2 | 753.1 | 0.2024 |
| `rain_90d_anom_mm` | 2.33e+04 | 0 | 5.016 | 40.91 | -193.4 | -59.68 | -14.02 | 0.8055 | 23.52 | 75.9 | 270.9 | 0.3239 |
| `county_area_km2` | 2.33e+04 | 0 | 8,138 | 8,126 | 754.7 | 1,611 | 4,397 | 5,873 | 9,608 | 1.921e+04 | 6.201e+04 | 4.285 |
| `elev_mean_m` | 2.33e+04 | 0 | 512.8 | 167.6 | 392.6 | 395.2 | 404.6 | 432.3 | 585.3 | 865.7 | 1,144 | 1.907 |
| `elev_std_m` | 2.33e+04 | 0 | 50.81 | 87.32 | 1.7 | 1.878 | 3.065 | 16.7 | 60.46 | 209.3 | 548.9 | 3.429 |
| `slope_mean_deg` | 2.33e+04 | 0 | 0.7605 | 1.08 | 0.1285 | 0.1414 | 0.1648 | 0.2463 | 1.082 | 2.724 | 6.606 | 3.072 |
| `hand_mean_m` | 2.33e+04 | 0 | 6.525 | 10.71 | 0.2525 | 0.3011 | 0.4501 | 1.493 | 9.904 | 23.85 | 68.2 | 3.275 |
| `hand_lt5m_frac` | 2.33e+04 | 0 | 0.7406 | 0.3168 | 0.1723 | 0.2221 | 0.3655 | 0.9353 | 0.9995 | 1 | 1 | -0.697 |
| `twi_mean` | 2.33e+04 | 0 | 12.24 | 1.344 | 9.639 | 10.01 | 10.71 | 12.95 | 13.36 | 13.46 | 13.49 | -0.6755 |
| `perm_water_frac` | 2.33e+04 | 0 | 0.0009088 | 0.001953 | 0 | 0 | 0 | 1.014e-06 | 0.0008578 | 0.006359 | 0.009913 | 2.991 |
| `jrc_water_occurrence_pct` | 2.33e+04 | 0 | 0.1567 | 0.2674 | 0 | 1.705e-06 | 0.0005519 | 0.01939 | 0.2099 | 0.8854 | 1.255 | 2.344 |
| `month` | 2.33e+04 | 0 | 6.397 | 3.418 | 1 | 1 | 3 | 6 | 9 | 12 | 12 | 0.03716 |
| `y_flood_2_4wk` | 2.116e+04 | 2145 | 0.2381 | 0.426 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1.23 |

**Rows per state**

| state | counties | rows |
|---|---|---|
| Abyei Region | 1 | 295 |
| Central Equatoria | 6 | 1770 |
| Eastern Equatoria | 8 | 2360 |
| Jonglei | 11 | 3245 |
| Lakes | 8 | 2360 |
| Northern Bahr el Ghazal | 5 | 1475 |
| Unity | 9 | 2655 |
| Upper Nile | 12 | 3540 |
| Warrap | 6 | 1770 |
| Western Bahr el Ghazal | 3 | 885 |
| Western Equatoria | 10 | 2950 |

**Target by temporal split** (`labelled` excludes rows whose target window was not imaged; those are never imputed)

| split | rows | labelled | positives | positive rate % |
|---|---|---|---|---|
| train | 16906 | 15550 | 2987 | 19.2 |
| validation | 3634 | 2948 | 1072 | 36.4 |
| test | 2765 | 2662 | 980 | 36.8 |

## 6. Source datasets and files

| Location | Contents |
|---|---|
| [`raw/suddwatch_panel_unprocessed.csv`](raw/suddwatch_panel_unprocessed.csv) | the panel described above (audited in Phase 1) |
| [`processed/`](processed/) | model-ready train / validation / test matrices (23 features) produced by the Phase 2 pipeline |

The panel was assembled (`src/build_panel.py`) from these Google Earth Engine extracts (`src/gee_extract.py`):

| dataset | file | rows | columns |
|---|---|---|---|
| Sentinel-1 water per county × window | [`gee/s1_water_by_county_window.csv`](raw/gee/s1_water_by_county_window.csv) | 23,463 | 7 |
| CHIRPS daily rainfall per county, 2001–2026 | [`gee/chirps_daily_by_county.csv`](raw/gee/chirps_daily_by_county.csv) | 740,546 | 3 |
| Static terrain / hydrology per county | [`gee/static_terrain_by_county.csv`](raw/gee/static_terrain_by_county.csv) | 79 | 9 |
| County 75th-percentile thresholds (training years only) | [`county_thresholds_train_only.csv`](raw/county_thresholds_train_only.csv) | 79 | 6 |

Plus the OCHA COD-AB county boundaries ([`raw/ssd_admin_boundaries/`](raw/ssd_admin_boundaries/)). UNOSAT water extents (2022–2024)
are used only to validate the Sentinel-1 label; the 67 MB file is not stored in Git (download link in the main README).

*Generated by `src/data_card.py` from the data files; re-run it to refresh every number on this page.*
