"""
SuddWatch API - county-level flood-risk forecasts for South Sudan, 13-24 days ahead.

Run locally:  .venv\\Scripts\\python.exe -m uvicorn api.main:app --reload
Docs:         http://127.0.0.1:8000/docs
"""
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.calibration import prior_shift

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/suddwatch_model.joblib"
LATEST_PATH = ROOT / "api/latest_features.csv"

bundle = joblib.load(MODEL_PATH)
PIPE, PLATT = bundle["pipeline"], bundle["platt"]
THRESHOLD = bundle["threshold_dynamic"]
INPUTS = bundle["input_columns"]
latest = pd.read_csv(LATEST_PATH)

app = FastAPI(
    title="SuddWatch flood-risk API",
    version="1.0.0",
    description=("Forecasts whether each South Sudan county's surface water will exceed its own normal "
                 "high-water level (training-period 75th percentile) 13-24 days after the issue date. "
                 "Model: gradient-boosted trees on Sentinel-1, CHIRPS and terrain features, calibrated, "
                 "with a dynamic decision threshold that adapts to the recently observed flood rate."),
)


def score(df: pd.DataFrame, known_prevalence: Optional[float] = None) -> pd.DataFrame:
    """Calibrated probability, prior-shift-corrected to the recent known flood rate, and the alert flag."""
    X = df.reindex(columns=INPUTS)
    p_cal = PLATT(PIPE.predict_proba(X)[:, 1])
    pi = bundle["pi_latest_known"] if known_prevalence is None else known_prevalence
    p = prior_shift(p_cal, PLATT.base_prior, float(np.clip(pi, 0.02, 0.98)))
    return pd.DataFrame({"probability": np.round(p, 4), "alert": p >= THRESHOLD}, index=df.index)


def window(issue: str) -> dict:
    d = date.fromisoformat(str(issue))
    return {"issue_date": d.isoformat(), "forecast_window_start": (d + timedelta(days=13)).isoformat(),
            "forecast_window_end": (d + timedelta(days=24)).isoformat()}


def as_records(df: pd.DataFrame) -> list:
    s = score(df)
    out = df[["adm2_pcode", "adm2_name", "adm1_name"]].join(s)
    out["risk_level"] = np.where(out["alert"], "HIGH: above county normal expected", "LOW")
    out = out.rename(columns={"adm2_pcode": "county_pcode", "adm2_name": "county", "adm1_name": "state"})
    return out.sort_values("probability", ascending=False).to_dict(orient="records")


@app.get("/", tags=["info"])
def root():
    return {"service": "SuddWatch flood-risk API", "docs": "/docs", "latest_forecast": "/forecast/latest",
            "model_card": "/model", **window(latest["issue_date"].iloc[0])}


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "model_loaded": True, "counties_available": int(len(latest))}


@app.get("/model", tags=["info"])
def model_card():
    return {"target": "county surface-water fraction in window w+2 exceeds the county's training-period P75 (floor 1 km2)",
            "horizon": bundle["horizon"], "trained_on": bundle["trained_on"], "created": bundle["created"],
            "decision_threshold": THRESHOLD, "recent_known_flood_rate": round(bundle["pi_latest_known"], 4),
            "calibration_base_rate": round(PLATT.base_prior, 4), "uses_vh_feature": bundle["uses_vh"],
            "test_metrics": bundle["metrics"]["test"], "validation_metrics": bundle["metrics"]["validation"],
            "versions": bundle["versions"], "input_columns": INPUTS}


@app.get("/counties", tags=["data"])
def counties():
    c = latest[["adm2_pcode", "adm2_name", "adm1_name", "y_threshold_frac"]].sort_values("adm2_pcode")
    return c.rename(columns={"adm2_pcode": "county_pcode", "adm2_name": "county", "adm1_name": "state",
                             "y_threshold_frac": "flood_threshold_water_fraction"}).to_dict(orient="records")


@app.get("/forecast/latest", tags=["forecast"])
def forecast_latest(state: Optional[str] = Query(None, description="Filter by state, e.g. Jonglei"),
                    top: Optional[int] = Query(None, ge=1, le=79, description="Return only the N highest-risk counties")):
    df = latest if state is None else latest[latest["adm1_name"].str.lower() == state.lower()]
    if df.empty:
        raise HTTPException(404, f"No counties for state '{state}'. See /counties.")
    recs = as_records(df)
    return {**window(latest["issue_date"].iloc[0]), "counties_on_alert": int(sum(r["alert"] for r in recs)),
            "forecasts": recs[:top] if top else recs}


@app.get("/forecast/latest/{county_pcode}", tags=["forecast"])
def forecast_county(county_pcode: str):
    df = latest[latest["adm2_pcode"].str.upper() == county_pcode.upper()]
    if df.empty:
        raise HTTPException(404, f"Unknown county pcode '{county_pcode}'. See /counties.")
    return {**window(latest["issue_date"].iloc[0]), **as_records(df)[0]}


class FeatureRow(BaseModel):
    adm2_pcode: str = Field(..., examples=["SS0302"])
    adm1_name: str = Field(..., examples=["Jonglei"])
    month: int = Field(..., ge=1, le=12)
    n_scenes: float
    obs_frac: float = Field(..., ge=0, le=1)
    water_frac: Optional[float] = None
    water_frac_last: Optional[float] = None
    water_frac_days: Optional[float] = None
    water_frac_lag1: Optional[float] = None
    water_frac_lag1_last: Optional[float] = None
    water_frac_lag1_days: Optional[float] = None
    water_frac_12m_mean: Optional[float] = None
    rain_36d_mm: float
    rain_90d_anom_mm: float
    county_area_km2: float
    jrc_water_occurrence_pct: float
    y_threshold_frac: float = Field(..., gt=0, description="County flood threshold (training-period P75); see /counties")
    elev_mean_m: float
    elev_std_m: float
    slope_mean_deg: float
    hand_mean_m: float
    hand_lt5m_frac: float
    twi_mean: float
    vh26_frac: Optional[float] = None


class PredictRequest(BaseModel):
    rows: list[FeatureRow]
    known_prevalence: Optional[float] = Field(None, ge=0.0, le=1.0,
                                              description="Recent national flood rate; defaults to the latest known rate")


@app.post("/predict", tags=["forecast"])
def predict(req: PredictRequest):
    df = pd.DataFrame([r.model_dump() for r in req.rows]).astype({c: float for c in INPUTS if c not in ("adm2_pcode", "adm1_name", "month")})
    s = score(df, req.known_prevalence)
    return {"threshold": THRESHOLD, "predictions": [{"county_pcode": c, "probability": float(p), "alert": bool(a)}
                                                     for c, p, a in zip(df["adm2_pcode"], s["probability"], s["alert"])]}
