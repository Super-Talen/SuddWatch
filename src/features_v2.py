"""
SuddWatch Phase 3 - causal (past-only) feature layer added in response to the Phase 2 feedback.

Reads the audited Phase 1 panel (data/raw/suddwatch_panel_unprocessed.csv, left unchanged) and adds,
per county in time order, columns that use ONLY information available at each issue date:

  water_frac_last, water_frac_days      last OBSERVED water fraction up to window w, and its age in days
  water_frac_lag1_last/_days, ..._lag2_  the same for the lagged columns (last observation up to w-1, w-2)
  water_frac_12m_mean                   trailing 12-month (30-window) mean of observed water fraction,
                                        min 8 observations; during longer gaps the last available
                                        12-month mean is carried forward (still past-only)
  sensor_era                            Sentinel-1 constellation period (for reporting domain shift)
  vh26_frac                             VH < -26 dB water fraction (Phase 3 label/feature variant), if extracted
  y_threshold_frac                      the county's target threshold (training-period P75, floor 1 km2):
                                        fixed from 2017-2023 data only, so known at every issue date

Nothing here is fitted, so it cannot leak across the train/validation/test split. Fitted steps
(seasonal medians, scalers, thresholds) live in the scikit-learn pipeline and use X_train only.

Usage: .venv\\Scripts\\python.exe -m src.features_v2
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/raw/suddwatch_panel_unprocessed.csv"
VH = ROOT / "data/raw/gee/s1_vh26_by_county_window.csv"
THR = ROOT / "data/raw/county_thresholds_train_only.csv"
OUT = ROOT / "data/processed/panel_v2_features.csv"

ROLL_WINDOWS = 30        # 30 x 12 days = 360 days, the "12-month" baseline the feedback asks for
ROLL_MIN_OBS = 8         # at least ~3 months of real observations inside the year
ERAS = [("2017-01-01", "2021-12-23", "S1A+S1B"),
        ("2021-12-23", "2025-07-01", "S1A only"),
        ("2025-07-01", "2100-01-01", "S1A+S1C/D")]
MIN_OBS_FRAC = 0.5       # same coverage rule as the label (src/build_panel.py)


def sensor_era(dates: pd.Series) -> pd.Series:
    out = pd.Series(index=dates.index, dtype=object)
    for a, b, name in ERAS:
        out[(dates >= a) & (dates < b)] = name
    return out


def add_causal_features(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.sort_values(["adm2_pcode", "window_start"]).copy()
    g = d.groupby("adm2_pcode", group_keys=False)

    obs_date = d["issue_date"].where(d["water_frac"].notna())
    d["water_frac_last"] = g["water_frac"].ffill()
    last_date = obs_date.groupby(d["adm2_pcode"]).ffill()
    d["water_frac_days"] = (d["issue_date"] - last_date).dt.days

    # lagged columns: their "last observation" is the one available one / two windows earlier
    for k in (1, 2):
        d[f"water_frac_lag{k}_last"] = g["water_frac_last"].shift(k)
        d[f"water_frac_lag{k}_days"] = g["water_frac_days"].shift(k) + 12 * k

    roll = g["water_frac"].rolling(ROLL_WINDOWS, min_periods=ROLL_MIN_OBS).mean().reset_index(level=0, drop=True)
    d["water_frac_12m_mean"] = roll.groupby(d["adm2_pcode"]).ffill()

    d["sensor_era"] = sensor_era(d["issue_date"])
    thr = pd.read_csv(THR)[["adm2_pcode", "y_threshold_frac"]]
    d = d.merge(thr, on="adm2_pcode", how="left")

    if VH.exists():
        vh = pd.read_csv(VH, parse_dates=["window_start"])
        d = d.merge(vh, on=["adm2_pcode", "window_start"], how="left")
        ok = (d["obs_frac"] >= MIN_OBS_FRAC) & (d["vh_eligible_km2"] > 0)
        d["vh26_frac"] = np.where(ok, d["vh_water_km2"] / d["vh_eligible_km2"], np.nan)
        d = d.drop(columns=["vh_water_km2", "vh_eligible_km2"])
    return d.sort_values(["issue_date", "adm2_pcode"]).reset_index(drop=True)


def main():
    panel = pd.read_csv(PANEL, parse_dates=["window_start", "issue_date"])
    out = add_causal_features(panel)
    out.to_csv(OUT, index=False)
    new = [c for c in out.columns if c not in panel.columns]
    print(f"wrote {OUT.relative_to(ROOT)} {out.shape}; new columns: {new}")
    print(out[new].isna().mean().mul(100).round(2).rename("missing_%").to_string())


if __name__ == "__main__":
    main()
