"""Leakage tests: every causal feature at issue date t must be unchanged when all rows after t are removed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features_v2 import add_causal_features

ROOT = Path(__file__).resolve().parents[1]
CAUSAL = ["water_frac_last", "water_frac_days", "water_frac_lag1_last", "water_frac_lag1_days",
          "water_frac_lag2_last", "water_frac_12m_mean"]


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(ROOT / "data/raw/suddwatch_panel_unprocessed.csv", parse_dates=["window_start", "issue_date"])


@pytest.mark.parametrize("cutoff", ["2019-06-30", "2022-03-31", "2024-12-31"])
def test_features_use_only_the_past(panel, cutoff):
    full = add_causal_features(panel)
    past = add_causal_features(panel[panel["issue_date"] <= cutoff])
    m = full[full["issue_date"] <= cutoff].merge(past, on=["adm2_pcode", "issue_date"], suffixes=("", "_p"))
    assert len(m) == (panel["issue_date"] <= cutoff).sum()
    for c in CAUSAL:
        a, b = m[c].to_numpy(dtype=float), m[c + "_p"].to_numpy(dtype=float)
        assert np.array_equal(a, b, equal_nan=True), f"{c} changes when future rows are removed"


def test_last_observation_is_never_from_the_future(panel):
    d = add_causal_features(panel)
    observed = d["water_frac"].notna()
    assert (d.loc[observed, "water_frac_days"] == 0).all()
    assert np.allclose(d.loc[observed, "water_frac_last"], d.loc[observed, "water_frac"])
    assert (d["water_frac_days"].dropna() >= 0).all()
