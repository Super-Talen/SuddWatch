"""
SuddWatch - Phase 2 label-sensitivity check (promised in the Phase 1 report, Section II).

Re-extracts ONLY the Sentinel-1 windows that overlap a UNOSAT monitoring date (2022-2024) and
computes alternative surface-water labels per county:
    VV < -18 dB  (Phase 1 label, recomputed as a control)
    VV < -16 dB  (sensitivity variant)
    VH < -22 / -24 / -26 dB  (cross-pol variants)
Each variant is scored against UNOSAT exactly as in src/audit.py: pooled Spearman across all
county-date pairs and median within-county temporal Spearman (overall and Sudd core).

Decision rule, fixed BEFORE running (Phase 2 decision log):
    adopt a variant only if   median within-county rho  improves by >= 0.10  over VV -18
                         and  pooled rho  does not drop by more than 0.05;
    otherwise keep the Phase 1 label (VV < -18 dB).

Usage: .venv\\Scripts\\python.exe -m src.label_sensitivity
Output: data/raw/gee/label_sensitivity_windows.csv, phase2_implementation/label_sensitivity.json
"""
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from scipy import stats

from .gee_extract import HAND_MAX_M, OUT, PROJECT, S1_SCALE_M, WINDOW_DAYS, counties_fc, with_retry

ROOT = Path(__file__).resolve().parents[1]
CACHE = OUT / "sensitivity"
RESULT_CSV = OUT / "label_sensitivity_windows.csv"
RESULT_JSON = ROOT / "phase2_implementation/label_sensitivity.json"
VARIANTS = {"vv18": ("VV", -18), "vv16": ("VV", -16), "vh22": ("VH", -22), "vh24": ("VH", -24), "vh26": ("VH", -26)}
CORE = ["Jonglei", "Unity", "Upper Nile", "Lakes"]
FIX = {"Cueibit": "Cueibet", "Gogrial Eest": "Gogrial East", "Kajo Keji": "Kajo-keji", "Lopa/Lafon": "Lafon",
       "Old Fangak": "Fangak", "Payinjar": "Panyijiar", "Peji": "Canal/Pigi", "Raga": "Raja", "Wuror": "Uror"}


def window(counties, start):
    out = CACHE / f"sens_{start:%Y%m%d}.csv"
    if out.exists():
        return out, "cached"
    end = start + timedelta(days=WINDOW_DAYS)
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(counties.geometry().bounds())
           .filterDate(str(start.date()), str(end.date()))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    med = col.select(["VV", "VH"]).median()
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
    px = ee.Image.pixelArea().divide(1e6)
    eligible = med.select("VV").mask().gt(0).And(hand.lt(HAND_MAX_M))
    bands = [eligible.unmask(0).multiply(px).rename("eligible_km2")]
    for name, (pol, db) in VARIANTS.items():
        bands.append(med.select(pol).lt(db).And(eligible).unmask(0).multiply(px).rename(name))
    fc = ee.Image.cat(bands).reduceRegions(collection=counties, reducer=ee.Reducer.sum(),
                                           scale=S1_SCALE_M, tileScale=4).map(lambda f: f.setGeometry(None))
    res = with_retry(lambda: fc.getInfo())
    rows = [{"adm2_pcode": f["properties"]["adm2_pcode"], "window_start": str(start.date()),
             **{k: f["properties"].get(k, 0.0) for k in ["eligible_km2", *VARIANTS]}} for f in res["features"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out, "fetched"


def unosat_grid(panel):
    con = sqlite3.connect(ROOT / "data/raw/UNOSAT_SouthSudan_FloodExtent_2022-2024.gpkg")
    u = pd.read_sql("select period_start, Adm2, Area_km2 from 'UNOSAT_SouthSudan_FloodExtent_2022-2024' "
                    "where Adm2 is not null", con, parse_dates=["period_start"])
    u["adm2_name"] = u["Adm2"].replace(FIX)
    names = panel[["adm2_name", "adm2_pcode", "adm1_name"]].drop_duplicates()
    u = u.merge(names, on="adm2_name")
    grid = pd.MultiIndex.from_product([names["adm2_pcode"], sorted(u["period_start"].unique())],
                                      names=["adm2_pcode", "period_start"]).to_frame(index=False)
    grid = grid.merge(u[["adm2_pcode", "period_start", "Area_km2"]], how="left").fillna({"Area_km2": 0})
    return grid[grid["adm2_pcode"] != "SS0001"].merge(names[["adm2_pcode", "adm1_name"]], on="adm2_pcode")


def score(m, col):
    pooled = stats.spearmanr(m["Area_km2"], m[col])[0]
    per = m.groupby("adm2_pcode").apply(
        lambda d: stats.spearmanr(d["Area_km2"], d[col])[0] if d["Area_km2"].nunique() > 2 and d[col].nunique() > 2 else np.nan,
        include_groups=False).dropna()
    st = m.drop_duplicates("adm2_pcode").set_index("adm2_pcode")["adm1_name"]
    core = per[per.index.map(st).isin(CORE)]
    return {"pooled_rho": round(float(pooled), 3), "median_within_county_rho": round(float(per.median()), 3),
            "core_median_within_county_rho": round(float(core.median()), 3), "n_counties": int(len(per))}


def main():
    ee.Initialize(project=PROJECT)
    panel = pd.read_csv(ROOT / "data/raw/suddwatch_panel_unprocessed.csv", parse_dates=["window_start"])
    grid = unosat_grid(panel)
    starts = sorted(panel["window_start"].unique())
    need = sorted({s for d in grid["period_start"].unique() for s in starts
                   if s <= d <= s + np.timedelta64(WINDOW_DAYS - 1, "D")})
    print(f"{len(need)} windows overlap UNOSAT dates", flush=True)
    counties = counties_fc()
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for f in as_completed([ex.submit(window, counties, pd.Timestamp(s)) for s in need]):
            print(" ", f.result()[1], f"{time.time() - t0:.0f}s", flush=True)
    w = pd.concat([pd.read_csv(p, parse_dates=["window_start"]) for p in sorted(CACHE.glob("sens_*.csv"))])
    w.to_csv(RESULT_CSV, index=False)
    for k in VARIANTS:
        w[k] = w[k] / w["eligible_km2"]
    w["window_end"] = w["window_start"] + pd.Timedelta(days=WINDOW_DAYS - 1)
    obs = panel[["adm2_pcode", "window_start", "obs_frac"]]
    w = w.merge(obs, on=["adm2_pcode", "window_start"])
    w = w[w["obs_frac"] >= 0.5]                            # same coverage rule as the label
    m = grid.merge(w, on="adm2_pcode")
    m = m[(m["period_start"] >= m["window_start"]) & (m["period_start"] <= m["window_end"])]
    res = {k: score(m, k) for k in VARIANTS}
    base = res["vv18"]
    for k, r in res.items():
        r["adopt"] = bool(k != "vv18"
                          and r["median_within_county_rho"] - base["median_within_county_rho"] >= 0.10
                          and base["pooled_rho"] - r["pooled_rho"] <= 0.05)
    out = {"decision_rule": "adopt only if median within-county rho improves >= 0.10 and pooled rho drops <= 0.05",
           "matched_pairs": int(len(m)), "results": res,
           "adopted": [k for k, r in res.items() if r["adopt"]] or ["vv18 (Phase 1 label kept)"]}
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
