"""API tests (run after src.train_model and src.export_latest)."""
import math

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, latest

client = TestClient(app)


def test_homepage_and_static_assets():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "SuddWatch" in r.text
    assert client.get("/static/app.js").status_code == 200
    geo = client.get("/static/data/counties.geojson").json()
    codes = {f["properties"]["adm2_pcode"] for f in geo["features"]}
    assert codes == set(latest["adm2_pcode"])          # every forecast county has a shape on the map
    assert client.get("/api").json()["docs"] == "/docs"


def test_health_and_model_card():
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/model").json()
    assert 0 < card["decision_threshold"] < 1
    assert "pr_auc" in card["test_metrics"]


def test_counties():
    c = client.get("/counties").json()
    assert len(c) == 79 and {"county_pcode", "county", "state"} <= set(c[0])


def test_latest_forecast_is_sorted_and_complete():
    r = client.get("/forecast/latest").json()
    probs = [f["probability"] for f in r["forecasts"]]
    assert len(probs) == 79 and probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs)
    assert r["forecast_window_start"] > r["issue_date"]


def test_state_filter_and_unknown_county():
    j = client.get("/forecast/latest", params={"state": "Jonglei"}).json()
    assert {f["state"] for f in j["forecasts"]} == {"Jonglei"}
    assert client.get("/forecast/latest/SS9999").status_code == 404


def _valid_payload():
    row = latest.iloc[0]
    payload = {k: (None if isinstance(v, float) and math.isnan(v) else v)
               for k, v in row.drop(["adm2_name", "issue_date"]).to_dict().items()}
    payload["month"] = int(payload["month"])
    return payload


def test_schema_rejects_bad_input():
    """Out-of-range, missing and wrongly typed fields are refused with 422 before reaching the model."""
    good = _valid_payload()
    bad_cases = [{**good, "obs_frac": 1.5}, {**good, "month": 13}, {**good, "y_threshold_frac": 0},
                 {k: v for k, v in good.items() if k != "rain_36d_mm"}, {**good, "rain_36d_mm": "heavy"}]
    for bad in bad_cases:
        assert client.post("/predict", json={"rows": [bad]}).status_code == 422
    assert client.post("/predict", json={"rows": []}).status_code == 422                       # empty batch
    assert client.post("/predict", json={"rows": [good], "known_prevalence": 1.5}).status_code == 422
    assert client.post("/predict", json={"rows": [good]}).status_code == 200


def test_data_quality_flag():
    from api.main import data_quality
    q = data_quality(pd.Series([0, 12, 20, 40, 200, None]))
    assert list(q[:2]) == ["ok", "ok"] and q[2].startswith("degraded") and q[3].startswith("low") and q[4].startswith("unreliable")
    assert q[5].startswith("unreliable")
    rec = client.get("/forecast/latest/SS0302").json()
    assert "data_quality" in rec and "water_observation_age_days" in rec


def test_timing_header():
    r = client.get("/health")
    assert float(r.headers["X-Process-Time-ms"]) >= 0 and r.json()["model_load_ms"] > 0


def test_predict_matches_forecast():
    row = latest.iloc[0]
    payload = {k: (None if isinstance(v, float) and math.isnan(v) else v)
               for k, v in row.drop(["adm2_name", "issue_date"]).to_dict().items()}
    payload["month"] = int(payload["month"])
    p = client.post("/predict", json={"rows": [payload]}).json()["predictions"][0]["probability"]
    f = client.get(f"/forecast/latest/{row['adm2_pcode']}").json()["probability"]
    assert abs(p - f) < 1e-6
