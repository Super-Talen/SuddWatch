"""
SuddWatch Phase 3 - evidence for the response to the Phase 2 feedback (training rows only).

1. Imputation: hide every real observation of 2020 (a full year, like the 2022-2025 gaps), refill it with
   (a) the Phase 2 static county median, (b) last observation carried forward, (c) the seasonal blend at
   several half-lives, and score each against the hidden truth. Chooses tau without touching labels.
2. Terrain tie-break stability: 200 bootstrap resamples of counties; which terrain feature has the highest MI.
3. Collinearity screen (|rho| >= 0.85, VIF > 10) on the new Phase 3 features.
4. Figure: one county's multi-year gap filled the Phase 2 way and the Phase 3 way.

Usage: .venv\\Scripts\\python.exe -m src.feedback_checks
Outputs: phase3_modeling/feedback/*.json|csv, figures/phase3_imputation_gap_example.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .features_v2 import add_causal_features
from .transformers import AnomalyFeature, GroupMedianImputer, MomentumFeature, SeasonalBlendImputer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase3_modeling/feedback"
FIG = ROOT / "figures"
TARGET = "y_flood_2_4wk"
TERRAIN = ["elev_mean_m", "elev_std_m", "slope_mean_deg", "hand_mean_m", "hand_lt5m_frac", "twi_mean"]
TAUS = [12, 24, 60, 120, 240, 480]
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9aa5b1", "#0b0b0b"


def load_train():
    raw = pd.read_csv(ROOT / "data/raw/suddwatch_panel_unprocessed.csv", parse_dates=["window_start", "issue_date"])
    return raw, raw[raw["split"] == "train"].copy()


def imputation_experiment(raw):
    """Mask all 2020 observations (training period) and compare refills against the hidden truth."""
    masked = raw.copy()
    hide = (masked["issue_date"].dt.year == 2020) & masked["water_frac"].notna()
    truth = masked.loc[hide, "water_frac"].copy()
    masked.loc[masked["issue_date"].dt.year == 2020, "water_frac"] = np.nan
    feat = add_causal_features(masked)
    tr = feat[feat["split"] == "train"]
    key = feat[["adm2_pcode", "issue_date"]].copy()
    rows = feat.merge(raw.loc[hide, ["adm2_pcode", "issue_date"]].assign(_t=truth.values), on=["adm2_pcode", "issue_date"])
    res = {}
    static = GroupMedianImputer(["water_frac"], add_indicator=False).fit(tr).transform(rows)["water_frac"]
    res["Phase 2: static county median"] = static
    res["last observation only"] = rows["water_frac_last"].fillna(static)
    for tau in TAUS:
        res[f"seasonal blend, tau={tau} d"] = SeasonalBlendImputer(["water_frac"], tau_days=tau, add_indicator=False).fit(tr).transform(rows)["water_frac"]
    t = rows["_t"]
    table = []
    for name, v in res.items():
        err = (v - t).abs()
        within = rows.assign(v=v).groupby("adm2_pcode")["v"].std().median()
        table.append({"method": name, "MAE_pct_points": 100 * err.mean(), "median_AE_pct_points": 100 * err.median(),
                      "spearman_with_truth": stats.spearmanr(v, t)[0],
                      "within_county_sd_of_fill": 100 * within})
    tab = pd.DataFrame(table)
    blends = tab[tab["method"].str.startswith("seasonal")]
    best = blends.loc[blends["MAE_pct_points"].idxmin(), "method"]
    best_tau = int(best.split("tau=")[1].split()[0])
    return tab, best_tau, int(hide.sum())


def terrain_stability(train, B=200, seed=0):
    lab = train.dropna(subset=[TARGET])
    counties = lab["adm2_pcode"].unique()
    rng = np.random.default_rng(seed)
    by = {c: g for c, g in lab.groupby("adm2_pcode")}
    wins, mis = [], []
    for b in range(B):
        pick = rng.choice(counties, size=len(counties), replace=True)
        d = pd.concat([by[c] for c in pick])
        mi = mutual_info_classif(d[TERRAIN], d[TARGET].astype(int), random_state=b)
        mis.append(mi)
        wins.append(TERRAIN[int(np.argmax(mi))])
    freq = pd.Series(wins).value_counts().reindex(TERRAIN, fill_value=0) / B
    mi = pd.DataFrame(mis, columns=TERRAIN)
    return pd.DataFrame({"win_share": freq, "mi_median": mi.median(), "mi_p05": mi.quantile(0.05), "mi_p95": mi.quantile(0.95)})


def collinearity_v2(train, tau):
    pre = [SeasonalBlendImputer(["water_frac", "water_frac_lag1"], tau_days=tau, add_indicator=False),
           GroupMedianImputer(["water_frac_12m_mean"], add_indicator=False), MomentumFeature(), AnomalyFeature()]
    X = train.copy()
    for step in pre:
        X = step.fit(X).transform(X) if not hasattr(step, "seasonal_") else step.transform(X)
    num = ["water_frac", "water_frac_12m_mean", "water_frac_anomaly", "water_frac_delta", "water_frac_days",
           "obs_frac", "n_scenes", "rain_36d_mm", "rain_90d_anom_mm", "county_area_km2", "jrc_water_occurrence_pct",
           "hand_lt5m_frac", "twi_mean"]
    rho = X[num].corr(method="spearman")
    pairs = [{"a": a, "b": b, "spearman": round(float(rho.loc[a, b]), 3)} for i, a in enumerate(num) for b in num[i + 1:]
             if abs(rho.loc[a, b]) >= 0.85]
    Z = X[num].dropna()
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    Zc = np.column_stack([np.ones(len(Z)), Z.values])
    vif = {c: round(float(variance_inflation_factor(Zc, i + 1)), 2) for i, c in enumerate(num)}
    lab = X.dropna(subset=[TARGET])
    mi = dict(zip(num, np.round(mutual_info_classif(lab[num], lab[TARGET].astype(int), random_state=0), 4)))
    return pairs, vif, mi, rho.loc["water_frac_anomaly"].round(3).to_dict()


def gap_figure(raw, tau, pcode="SS0608"):
    feat = add_causal_features(raw)
    tr = feat[feat["split"] == "train"]
    s = feat[feat["adm2_pcode"] == pcode].sort_values("issue_date")
    static = GroupMedianImputer(["water_frac"], add_indicator=False).fit(tr).transform(s)["water_frac"]
    blend = SeasonalBlendImputer(["water_frac"], tau_days=tau, add_indicator=False).fit(tr).transform(s)["water_frac"]
    miss = s["water_frac"].isna()
    name = s["adm2_name"].iloc[0]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(s["issue_date"], 100 * s["water_frac"], color=BLUE, lw=1.4, label="observed (Sentinel-1)")
    ax.scatter(s.loc[miss, "issue_date"], 100 * static[miss], s=10, color=GRAY, label="Phase 2 fill: static county median", zorder=3)
    ax.scatter(s.loc[miss, "issue_date"], 100 * blend[miss], s=10, color=ORANGE, label=f"Phase 3 fill: seasonal blend (tau = {tau} d)", zorder=4)
    ax.set_ylabel("surface water (% of county)")
    ax.grid(color="#e6e6e3", lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper center")
    ax.set_title(f"{name} ({pcode}): filling the 2022-2025 Sentinel-1 gap. Static medians draw a flat line; "
                 "the seasonal blend keeps the yearly cycle", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "phase3_imputation_gap_example.png", dpi=200, facecolor="white"); plt.close(fig)
    return name, int(miss.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw, _ = load_train()
    tab, tau, n_hidden = imputation_experiment(raw)
    tab.round(4).to_csv(OUT / "imputation_experiment.csv", index=False)
    print(f"Imputation experiment ({n_hidden} hidden 2020 observations):\n{tab.round(4).to_string(index=False)}\n-> chosen tau = {tau} days")
    feat = add_causal_features(raw)
    train = feat[feat["split"] == "train"]
    stab = terrain_stability(train)
    stab.round(4).to_csv(OUT / "terrain_stability.csv")
    print("\nTerrain tie-break stability (200 county bootstraps):\n", stab.round(4).to_string())
    pairs, vif, mi, rho_anom = collinearity_v2(train, tau)
    print("\nNew-feature collinearity pairs |rho|>=0.85:", pairs)
    print("VIF:", vif)
    print("MI with Y:", mi)
    name, n_gap = gap_figure(raw, tau)
    summary = {"chosen_tau_days": tau, "hidden_observations": n_hidden,
               "imputation": tab.round(4).to_dict(orient="records"),
               "terrain_stability": stab.round(4).reset_index(names="feature").to_dict(orient="records"),
               "terrain_winner": stab["win_share"].idxmax(),
               "collinear_pairs_v2": pairs, "vif_v2": vif, "mi_v2": mi, "anomaly_spearman": rho_anom,
               "gap_example": {"county": name, "filled_windows": n_gap}}
    (OUT / "feedback_checks.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote", OUT / "feedback_checks.json")


if __name__ == "__main__":
    main()
