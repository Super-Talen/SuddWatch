"""
SuddWatch - assemble the county x 12-day-window panel from the raw Earth Engine extracts.

This step only *constructs* the modelling table (joins, ratios, lags, target).
It deliberately performs NO cleaning, imputation, capping, encoding or scaling:
those are Phase 2 pipeline steps fitted on X_train only (see src/pipeline_builder.py).

Row grain   : one row per (county, issue window w); the forecast is issued at window_end of w.
Target (Y)  : 1 if the county's surface-water fraction in window w+2 (13-24 days after issue)
              exceeds that county's own 75th percentile, computed on TRAINING windows only.
Missing     : water fraction is set to NaN when < MIN_OBS_FRAC of the county was imaged in a
              window (Sentinel-1 acquisition-plan gaps, 2022-2025). Y is NaN if w+2 is unobserved.

Usage: .venv\\Scripts\\python.exe -m src.build_panel
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
GEE = RAW / "gee"
OUT = RAW / "suddwatch_panel_unprocessed.csv"     # assembled but un-preprocessed -> audited in Phase 1

MIN_OBS_FRAC = 0.5          # below this a window does not represent the county
HORIZON = 2                 # target window = w + 2  (13-24 days ahead)
TARGET_PCTL = 0.75          # county-specific exceedance threshold
FLOOR_KM2 = 1.0             # threshold floor: smallest meaningful water area
TRAIN_END = "2024-01-01"    # issue windows before this date are training data
VAL_END = "2025-07-01"      # [TRAIN_END, VAL_END) validation; [VAL_END, ...) test
ISSUE_LAST = "2026-08-31"   # CHIRPS availability limits the last issue date
CLIM_YEARS = (2001, 2016)   # rainfall climatology baseline, strictly before the study period


def load_counties():
    g = gpd.read_file(RAW / "ssd_admin_boundaries/ssd_admin2.geojson")
    return pd.DataFrame(g[["adm2_pcode", "adm2_name", "adm1_name", "area_sqkm"]]).rename(
        columns={"area_sqkm": "county_area_km2"})


def rainfall_features(issue_dates: pd.DataFrame) -> pd.DataFrame:
    """Rain sums over the 12/36/90 days ending on (and including) the issue date + 90-day anomaly."""
    ch = pd.read_csv(GEE / "chirps_daily_by_county.csv", parse_dates=["date"])
    ch = ch.sort_values(["adm2_pcode", "date"])
    g = ch.groupby("adm2_pcode")["precip_mm"]
    for n in (12, 36, 90):
        ch[f"rain_{n}d_mm"] = g.transform(lambda s: s.rolling(n, min_periods=n).sum())
    # climatology: mean 90-day total ending on the same calendar day, 2001-2016 only
    ch["md"] = ch["date"].dt.strftime("%m-%d").replace("02-29", "02-28")
    base = ch[ch["date"].dt.year.between(*CLIM_YEARS)]
    clim = base.groupby(["adm2_pcode", "md"])["rain_90d_mm"].mean().rename("rain_90d_clim_mm").reset_index()
    ch = ch.merge(clim, on=["adm2_pcode", "md"], how="left")
    ch["rain_90d_anom_mm"] = ch["rain_90d_mm"] - ch["rain_90d_clim_mm"]
    cols = ["adm2_pcode", "date", "rain_12d_mm", "rain_36d_mm", "rain_90d_mm", "rain_90d_anom_mm"]
    return issue_dates.merge(ch[cols].rename(columns={"date": "issue_date"}),
                             on=["adm2_pcode", "issue_date"], how="left")


def build():
    s1 = pd.read_csv(GEE / "s1_water_by_county_window.csv", parse_dates=["window_start", "window_end"])
    counties = load_counties()
    df = s1.merge(counties, on="adm2_pcode", how="left").sort_values(["adm2_pcode", "window_start"])

    # --- surface-water fraction of the observable (HAND < 20 m) county area ---
    df["obs_frac"] = (df["obs_km2"] / df["county_area_km2"]).clip(upper=1.0)
    observed = df["obs_frac"] >= MIN_OBS_FRAC
    df["water_frac"] = np.where(observed, df["water_km2"] / df["eligible_km2"], np.nan)

    # --- autoregressive features (only information available at issue time) ---
    g = df.groupby("adm2_pcode")["water_frac"]
    df["water_frac_lag1"] = g.shift(1)
    df["water_frac_lag2"] = g.shift(2)
    df["target_water_frac"] = g.shift(-HORIZON)          # used to define Y only, never a feature

    # --- split labels (temporal) ---
    df["split"] = np.select([df["window_start"] < TRAIN_END, df["window_start"] < VAL_END],
                            ["train", "validation"], "test")

    # --- county-specific threshold from TRAINING windows only (no look-ahead) ---
    tr = df[df["split"] == "train"]
    thr = tr.groupby("adm2_pcode")["water_frac"].quantile(TARGET_PCTL).rename("thr_p75").to_frame()
    elig = df.groupby("adm2_pcode")["eligible_km2"].median()
    thr["thr_floor"] = FLOOR_KM2 / elig
    thr["y_threshold_frac"] = thr[["thr_p75", "thr_floor"]].max(axis=1)
    thr["n_train_obs"] = tr.groupby("adm2_pcode")["water_frac"].count()
    df = df.merge(thr[["y_threshold_frac"]], left_on="adm2_pcode", right_index=True, how="left")
    df["y_flood_2_4wk"] = np.where(df["target_water_frac"].isna(), np.nan,
                                   (df["target_water_frac"] > df["y_threshold_frac"]).astype(float))

    # --- rainfall up to the issue date ---
    df["issue_date"] = df["window_end"]
    df = df[df["issue_date"] <= ISSUE_LAST]
    df = rainfall_features(df)

    # --- static terrain / hydrology ---
    st = pd.read_csv(GEE / "static_terrain_by_county.csv").rename(columns={
        "elev_m_mean": "elev_mean_m", "elev_m_stdDev": "elev_std_m", "slope_deg_mean": "slope_mean_deg",
        "hand_m_mean": "hand_mean_m", "hand_lt5_frac_mean": "hand_lt5m_frac", "twi_mean": "twi_mean",
        "perm_water_frac_mean": "perm_water_frac", "water_occurrence_pct_mean": "jrc_water_occurrence_pct"})
    df = df.merge(st, on="adm2_pcode", how="left")
    df["month"] = df["issue_date"].dt.month

    cols = ["adm2_pcode", "adm2_name", "adm1_name", "window_start", "issue_date", "split",
            "n_scenes", "obs_frac", "water_frac", "water_frac_lag1", "water_frac_lag2",
            "rain_12d_mm", "rain_36d_mm", "rain_90d_mm", "rain_90d_anom_mm",
            "county_area_km2", "elev_mean_m", "elev_std_m", "slope_mean_deg", "hand_mean_m",
            "hand_lt5m_frac", "twi_mean", "perm_water_frac", "jrc_water_occurrence_pct",
            "month", "y_flood_2_4wk"]
    panel = df[cols].sort_values(["issue_date", "adm2_pcode"]).reset_index(drop=True)
    panel.to_csv(OUT, index=False)
    thr.reset_index().merge(counties[["adm2_pcode", "adm2_name"]], on="adm2_pcode").to_csv(
        RAW / "county_thresholds_train_only.csv", index=False)
    print("wrote", OUT, panel.shape)
    return panel


if __name__ == "__main__":
    build()
