"""
SuddWatch - Google Earth Engine extraction (raw data acquisition).

Pulls three county-level raw tables for South Sudan's 79 counties (COD-AB admin2):
  1. Sentinel-1 surface-water area per county per 12-day window (2017-01-01 .. 2026-09-30)
  2. CHIRPS daily rainfall, county mean, 2001-01-01 .. 2026-08-31
     (2001-2016 is used only as an external climatology baseline for anomalies)
  3. Static terrain / hydrology descriptors (SRTM, MERIT Hydro, JRC Global Surface Water)

Everything is reduced to county level server-side, so only small CSVs are downloaded.
Each chunk is cached under data/raw/gee/, so re-running only fetches what is missing.

Usage:  .venv\\Scripts\\python.exe -m src.gee_extract [s1|chirps|static|all]
Requires: earthengine authenticate + a registered Cloud project (default: suddwatch-2026).
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd

PROJECT = "suddwatch-2026"
ROOT = Path(__file__).resolve().parents[1]
ADMIN2 = ROOT / "data/raw/ssd_admin_boundaries/ssd_admin2.geojson"
OUT = ROOT / "data/raw/gee"

# --- Design constants (justified in phase1_strategy/Phase1_Strategy_Report) ---
WINDOW_DAYS = 12                 # Sentinel-1A repeat cycle
S1_START, S1_END = date(2017, 1, 1), date(2026, 9, 30)
WATER_DB = -18                   # VV threshold; best county-level agreement with UNOSAT (Spearman 0.70)
HAND_MAX_M = 20                  # exclude terrain far above drainage (radar-shadow false positives)
S1_SCALE_M = 200                 # county-scale analysis; smallest county (755 km2) still ~19k pixels
CHIRPS_START, CHIRPS_END = date(2001, 1, 1), date(2026, 8, 31)
THREADS = 10


def init():
    ee.Initialize(project=PROJECT)


def counties_fc():
    g = gpd.read_file(ADMIN2)[["adm2_pcode", "geometry"]]
    # ~200 m simplification keeps the request payload small; negligible at county scale
    g["geometry"] = g.geometry.simplify(0.002, preserve_topology=True)
    return ee.FeatureCollection(json.loads(g.to_json()))


def with_retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # EE quota / transient network errors
            if i == tries - 1:
                raise
            wait = 15 * (i + 1)
            print(f"   retry {i + 1} after error: {str(e)[:120]} (sleep {wait}s)", flush=True)
            time.sleep(wait)


# ---------------------------------------------------------------- Sentinel-1
def s1_window(counties, start):
    end = start + timedelta(days=WINDOW_DAYS)
    out = OUT / "s1" / f"s1_{start:%Y%m%d}.csv"
    if out.exists():
        return out, "cached"
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(counties.geometry().bounds())
           .filterDate(str(start), str(end))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .select("VV"))
    n_scenes = col.size()
    vv = col.median()                                    # per-pixel median over the window (speckle reduction)
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
    px = ee.Image.pixelArea().divide(1e6)
    observed = vv.mask().gt(0)
    eligible = observed.And(hand.lt(HAND_MAX_M))
    water = vv.lt(WATER_DB).And(eligible)
    stack = ee.Image.cat([
        observed.unmask(0).multiply(px).rename("obs_km2"),
        eligible.unmask(0).multiply(px).rename("eligible_km2"),
        water.unmask(0).multiply(px).rename("water_km2"),
    ])
    fc = stack.reduceRegions(collection=counties, reducer=ee.Reducer.sum(),
                             scale=S1_SCALE_M, tileScale=4)
    fc = fc.map(lambda f: f.set("n_scenes", n_scenes).setGeometry(None))
    res = with_retry(lambda: fc.getInfo())
    rows = [{"adm2_pcode": f["properties"]["adm2_pcode"],
             "window_start": str(start), "window_end": str(end - timedelta(days=1)),
             "n_scenes": f["properties"]["n_scenes"],
             "obs_km2": f["properties"].get("obs_km2", 0.0),
             "eligible_km2": f["properties"].get("eligible_km2", 0.0),
             "water_km2": f["properties"].get("water_km2", 0.0)} for f in res["features"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out, "fetched"


def run_s1():
    counties = counties_fc()
    starts, d = [], S1_START
    while d + timedelta(days=WINDOW_DAYS - 1) <= S1_END:
        starts.append(d)
        d += timedelta(days=WINDOW_DAYS)
    print(f"Sentinel-1: {len(starts)} windows x 79 counties", flush=True)
    _run_parallel(lambda s: s1_window(counties, s), starts, "S1")
    df = pd.concat([pd.read_csv(p) for p in sorted((OUT / "s1").glob("s1_*.csv"))])
    df.to_csv(OUT / "s1_water_by_county_window.csv", index=False)
    print("wrote", OUT / "s1_water_by_county_window.csv", df.shape, flush=True)


# ---------------------------------------------------------------- CHIRPS
def chirps_year(counties, y):
    out = OUT / "chirps" / f"chirps_{y}.csv"
    if out.exists():
        return out, "cached"
    end = min(date(y + 1, 1, 1), CHIRPS_END + timedelta(days=1))
    col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(f"{y}-01-01", str(end)).select("precipitation")
    # stack every day of the year as a band -> a single reduceRegions call per year
    names = col.aggregate_array("system:index")
    img = col.toBands().rename(names)
    fc = img.reduceRegions(collection=counties, reducer=ee.Reducer.mean(), scale=5566, tileScale=2)             .map(lambda f: f.setGeometry(None))
    res = with_retry(lambda: fc.getInfo())
    rows = []
    for f in res["features"]:
        p = f["properties"]
        for k, v in p.items():
            if k != "adm2_pcode":
                rows.append({"adm2_pcode": p["adm2_pcode"], "date": f"{k[:4]}-{k[4:6]}-{k[6:8]}", "precip_mm": v})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["adm2_pcode", "date"]).to_csv(out, index=False)
    return out, "fetched"


def run_chirps():
    counties = counties_fc()
    years = list(range(CHIRPS_START.year, CHIRPS_END.year + 1))
    print(f"CHIRPS: {len(years)} years x 79 counties (daily)", flush=True)
    _run_parallel(lambda y: chirps_year(counties, y), years, "CHIRPS")
    df = pd.concat([pd.read_csv(p) for p in sorted((OUT / "chirps").glob("chirps_*.csv"))])
    df.to_csv(OUT / "chirps_daily_by_county.csv", index=False)
    print("wrote", OUT / "chirps_daily_by_county.csv", df.shape, flush=True)


# ---------------------------------------------------------------- Static terrain
def run_static():
    out = OUT / "static_terrain_by_county.csv"
    if out.exists():
        print("static: cached"); return
    counties = counties_fc()
    srtm = ee.Image("USGS/SRTMGL1_003").select("elevation")
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    slope_deg = ee.Terrain.slope(srtm)
    # TWI = ln(a / tan(beta)); a = specific catchment area (m2 per m contour width), beta = local slope.
    # MERIT 'upa' is upstream area in km2 at ~90 m; MERIT 'elv' slope keeps a and beta on the same grid.
    merit_slope = ee.Terrain.slope(merit.select("elv")).multiply(3.141592653589793 / 180)
    a = merit.select("upa").multiply(1e6).divide(90)
    twi = a.divide(merit_slope.tan().max(0.001)).log().rename("twi")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    img = ee.Image.cat([
        srtm.rename("elev_m"),
        slope_deg.rename("slope_deg"),
        merit.select("hnd").rename("hand_m"),
        merit.select("hnd").lt(5).rename("hand_lt5_frac"),
        twi,
        jrc.gte(80).rename("perm_water_frac"),
        jrc.rename("water_occurrence_pct"),
    ])
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    fc = img.reduceRegions(collection=counties, reducer=reducer, scale=250, tileScale=8)
    res = with_retry(lambda: fc.getInfo())
    df = pd.DataFrame([f["properties"] for f in res["features"]])
    keep = ["adm2_pcode", "elev_m_mean", "elev_m_stdDev", "slope_deg_mean", "hand_m_mean",
            "hand_lt5_frac_mean", "twi_mean", "perm_water_frac_mean", "water_occurrence_pct_mean"]
    df[keep].to_csv(out, index=False)
    print("wrote", out, df[keep].shape)


def _run_parallel(fn, items, label):
    t0, done = time.time(), 0
    with ThreadPoolExecutor(THREADS) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for f in as_completed(futs):
            done += 1
            try:
                _, status = f.result()
            except Exception as e:
                status = f"FAILED: {str(e)[:150]}"
            if status != "cached" or done % 50 == 0:
                print(f"[{label}] {done}/{len(items)} {futs[f]} {status} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    init()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("static", "all"):
        run_static()
    if what in ("chirps", "all"):
        run_chirps()
    if what in ("s1", "all"):
        run_s1()
