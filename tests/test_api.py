"""API tests (run after src.train_model and src.export_latest)."""
import math

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app, latest

client = TestClient(app)


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


def test_predict_matches_forecast():
    row = latest.iloc[0]
    payload = {k: (None if isinstance(v, float) and math.isnan(v) else v)
               for k, v in row.drop(["adm2_name", "issue_date"]).to_dict().items()}
    payload["month"] = int(payload["month"])
    p = client.post("/predict", json={"rows": [payload]}).json()["predictions"][0]["probability"]
    f = client.get(f"/forecast/latest/{row['adm2_pcode']}").json()["probability"]
    assert abs(p - f) < 1e-6
