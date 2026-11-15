"""Export the most recent issue date's county feature rows for the API (api/latest_features.csv)."""
from pathlib import Path

import pandas as pd

from .pipeline_v2 import INPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


def main():
    d = pd.read_csv(ROOT / "data/processed/panel_v2_features.csv", parse_dates=["issue_date"])
    latest = d[d["issue_date"] == d["issue_date"].max()]
    cols = ["adm2_name", "issue_date"] + INPUT_COLUMNS + ["vh26_frac"]
    out = ROOT / "api/latest_features.csv"
    latest[cols].assign(issue_date=latest["issue_date"].dt.date).to_csv(out, index=False)
    print("wrote", out.relative_to(ROOT), latest.shape[0], "counties, issue date", latest["issue_date"].max().date())


if __name__ == "__main__":
    main()
