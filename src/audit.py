"""
SuddWatch - Phase 1 pre-implementation audit of the unprocessed panel.

Produces every number used in phase1_strategy/Phase1_Strategy_Report:
  A. shape, variable-type inventory, baseline missingness, target balance
  B. missingness-mechanism evidence, skewness, outlier counts (IQR / Z), label validation vs UNOSAT
  C. feature-target association matrix, collinearity screen (|r| >= 0.85, VIF > 10), split sizes
Statistics that could leak (associations, collinearity, outlier bounds) use TRAINING rows only.

Usage: .venv\\Scripts\\python.exe -m src.audit
"""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/raw/suddwatch_panel_unprocessed.csv"
OUT = ROOT / "phase1_strategy/audit"
FIG = ROOT / "figures"

TYPES = {
    "adm2_pcode": "Nominal (ID / join key)", "adm2_name": "Nominal (ID / label)",
    "adm1_name": "Nominal categorical", "window_start": "Datetime", "issue_date": "Datetime",
    "split": "Nominal (partition tag, not a feature)", "n_scenes": "Discrete (count)",
    "month": "Discrete (cyclical)", "y_flood_2_4wk": "Binary target (Y)",
}
NON_FEATURES = ["adm2_pcode", "adm2_name", "window_start", "issue_date", "split", "y_flood_2_4wk"]


def vtype(c):
    return TYPES.get(c, "Continuous")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PANEL, parse_dates=["window_start", "issue_date"])
    R = {}

    # ---------------- A1/A2: shape and type inventory
    R["N"], R["P_total_columns"] = df.shape
    feats = [c for c in df.columns if c not in NON_FEATURES]
    R["P_candidate_features"] = len(feats)
    inv = pd.DataFrame({"column": df.columns, "type": [vtype(c) for c in df.columns],
                        "dtype": df.dtypes.astype(str).values,
                        "n_unique": df.nunique().values,
                        "role": ["feature" if c in feats else ("target" if c == "y_flood_2_4wk" else "identifier/meta")
                                 for c in df.columns]})
    R["type_counts"] = inv["type"].value_counts().to_dict()

    # ---------------- A3: baseline missingness
    miss = df.isna().sum()
    inv["n_missing"] = miss.values
    inv["pct_missing"] = (100 * miss / len(df)).round(2).values
    inv.to_csv(OUT / "A_variable_inventory_missingness.csv", index=False)
    R["overall_missing_pct"] = round(100 * df.isna().sum().sum() / df.size, 2)
    R["complete_rows"] = int(df.notna().all(axis=1).sum())
    R["rows_with_missing"] = int(len(df) - R["complete_rows"])

    # ---------------- A4: target
    y = df["y_flood_2_4wk"]
    R["y_labelled_rows"] = int(y.notna().sum())
    for s in ["train", "validation", "test"]:
        ys = df.loc[df.split == s, "y_flood_2_4wk"].dropna()
        R[f"y_{s}"] = {"n_rows": int((df.split == s).sum()), "n_labelled": int(len(ys)),
                       "pos": int(ys.sum()), "neg": int((ys == 0).sum()),
                       "pos_rate": round(float(ys.mean()), 4),
                       "neg_to_pos_ratio": round(float((ys == 0).sum() / max(ys.sum(), 1)), 2),
                       "first_issue": str(df.loc[df.split == s, "issue_date"].min().date()),
                       "last_issue": str(df.loc[df.split == s, "issue_date"].max().date())}
    lab = df[y.notna()]
    R["split_pct_of_labelled"] = (100 * lab["split"].value_counts() / len(lab)).round(1).to_dict()

    # ---------------- B1: missingness mechanism evidence (water_frac)
    df["wf_missing"] = df["water_frac"].isna().astype(int)
    df["year"] = df["issue_date"].dt.year
    by_year = df.groupby("year")["wf_missing"].mean().mul(100).round(1)
    by_state = df.groupby("adm1_name")["wf_missing"].mean().mul(100).round(1)
    era = pd.cut(df["issue_date"], [pd.Timestamp("2016-12-31"), pd.Timestamp("2021-12-23"),
                                     pd.Timestamp("2025-06-30"), pd.Timestamp("2027-01-01")],
                 labels=["S1A+S1B (2017-2021)", "S1A only (2022-mid 2025)", "S1A+S1C/D (mid 2025-2026)"])
    by_era = df.groupby(era, observed=True)["wf_missing"].mean().mul(100).round(1)
    ct = pd.crosstab(era, df["wf_missing"])
    chi2, p_era, *_ = stats.chi2_contingency(ct)
    ct2 = pd.crosstab(df["adm1_name"], df["wf_missing"])
    chi2s, p_state, *_ = stats.chi2_contingency(ct2)
    # does missingness depend on rainfall (a proxy for the unobserved flood state)? -> MNAR check
    sub = df[df["year"].between(2022, 2024)].dropna(subset=["rain_36d_mm"])
    u, p_rain = stats.mannwhitneyu(sub.loc[sub.wf_missing == 1, "rain_36d_mm"],
                                   sub.loc[sub.wf_missing == 0, "rain_36d_mm"])
    # adjusted test: does rainfall still predict missingness once location and season are controlled?
    sub = sub.assign(rain_z=(sub["rain_36d_mm"] - sub["rain_36d_mm"].mean()) / sub["rain_36d_mm"].std())
    lg = smf.logit("wf_missing ~ C(adm1_name) + C(month) + rain_z", data=sub).fit(disp=0)
    R["mechanism"] = {"water_frac_missing_pct_by_era": by_era.to_dict(),
                      "logit_rain_coef_adj_state_month": float(lg.params["rain_z"]),
                      "logit_rain_p_adj_state_month": float(lg.pvalues["rain_z"]),
                      "chi2_era_p": float(p_era), "chi2_state_p": float(p_state),
                      "rain36_missing_vs_observed_median_2022_24": [
                          float(sub.loc[sub.wf_missing == 1, "rain_36d_mm"].median()),
                          float(sub.loc[sub.wf_missing == 0, "rain_36d_mm"].median())],
                      "mannwhitney_rain_p_2022_24": float(p_rain),
                      "missing_pct_by_state": by_state.to_dict(),
                      "missing_pct_by_year": by_year.to_dict()}

    # ---------------- B2: skewness + outliers (training rows; continuous features)
    tr = df[df["split"] == "train"]
    cont = [c for c in feats if vtype(c) == "Continuous"]
    rows = []
    for c in cont:
        x = tr[c].dropna()
        q1, q3 = x.quantile([0.25, 0.75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (x - x.mean()) / x.std(ddof=0)
        sk = float(stats.skew(x))
        rows.append({"feature": c, "n": len(x), "mean": x.mean(), "median": x.median(), "skew": sk,
                     "rule": "IQR (skewed)" if abs(sk) > 1 else ("IQR (moderate skew)" if abs(sk) > 0.5 else "Z-score (|z|>3)"),
                     "iqr_low": lo, "iqr_high": hi, "pct_iqr_outliers": 100 * ((x < lo) | (x > hi)).mean(),
                     "pct_z3_outliers": 100 * (z.abs() > 3).mean(), "min": x.min(), "max": x.max()})
    sk = pd.DataFrame(rows).round(4)
    sk.to_csv(OUT / "B_skewness_outliers_train.csv", index=False)

    # ---------------- C1: feature-target association (training rows, labelled)
    trl = tr.dropna(subset=["y_flood_2_4wk"])
    yt = trl["y_flood_2_4wk"].astype(int)
    assoc = []
    for c in cont + ["n_scenes", "month"]:
        m = trl[c].notna()
        r_pb, p_pb = stats.pointbiserialr(yt[m], trl.loc[m, c])
        _, p_mw = stats.mannwhitneyu(trl.loc[m & (yt == 1), c], trl.loc[m & (yt == 0), c])
        mi = mutual_info_classif(trl.loc[m, [c]], yt[m], random_state=0)[0]
        assoc.append({"feature": c, "type": vtype(c), "test": "Point-biserial r + Mann-Whitney U + MI",
                      "abs_r_pb": abs(r_pb), "p_pb": p_pb, "p_mannwhitney": p_mw, "mutual_info": mi})
    chi2s_, p_chi, *_ = stats.chi2_contingency(pd.crosstab(trl["adm1_name"], yt))
    cv = np.sqrt(chi2s_ / (len(trl) * (min(pd.crosstab(trl["adm1_name"], yt).shape) - 1)))
    mi_state = mutual_info_classif(pd.factorize(trl["adm1_name"])[0].reshape(-1, 1), yt,
                                   discrete_features=True, random_state=0)[0]
    assoc.append({"feature": "adm1_name", "type": "Nominal categorical", "test": "Chi-square + Cramer's V + MI",
                  "abs_r_pb": cv, "p_pb": p_chi, "p_mannwhitney": np.nan, "mutual_info": mi_state})
    A = pd.DataFrame(assoc).sort_values("mutual_info", ascending=False).round(5)
    A.to_csv(OUT / "C_feature_target_association_train.csv", index=False)

    # ---------------- C2: collinearity (Spearman because of skew) + VIF
    num = [c for c in cont + ["n_scenes"]]
    corr = tr[num].corr(method="spearman")
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:] if abs(corr.loc[a, b]) >= 0.85]
    mi_map = A.set_index("feature")["mutual_info"].to_dict()
    pair_df = pd.DataFrame([{"feature_a": a, "feature_b": b, "spearman": round(r, 3)} for a, b, r in pairs])
    # Greedy decision rule: walk pairs from strongest |rho| down; if both still present, drop the one
    # with LOWER mutual information with Y. Then iterate VIF: drop the highest-VIF feature (> 10)
    # until every remaining VIF <= 10.
    kept, dropped = list(num), []
    for a, b, r in sorted(pairs, key=lambda t: -abs(t[2])):
        if a in kept and b in kept:
            d = a if mi_map.get(a, 0) < mi_map.get(b, 0) else b
            kept.remove(d); dropped.append({"feature": d, "reason": f"|rho|={abs(r):.3f} with {b if d == a else a}"})
    def vif_table(cols):
        X = tr[cols].dropna(); X = (X - X.mean()) / X.std(ddof=0)
        Xc = np.column_stack([np.ones(len(X)), X.values])
        return pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(len(cols))], index=cols)
    vif_before = vif_table([c for c in num if tr[c].std() > 0])
    v = vif_table(kept)
    while v.max() > 10:
        d = v.idxmax(); kept.remove(d); dropped.append({"feature": d, "reason": f"VIF={v.max():.1f} > 10"}); v = vif_table(kept)
    pair_df["dropped_by_rule"] = pair_df.apply(lambda r: next((x["feature"] for x in dropped if x["feature"] in (r.feature_a, r.feature_b)), ""), axis=1)
    pair_df.to_csv(OUT / "C_collinear_pairs_train.csv", index=False)
    vif = pd.DataFrame({"feature": vif_before.index, "VIF_before": vif_before.values.round(2),
                        "VIF_after": [round(v[c], 2) if c in v.index else np.nan for c in vif_before.index]})
    vif.sort_values("VIF_before", ascending=False).to_csv(OUT / "C_vif_train.csv", index=False)
    pd.DataFrame(dropped).to_csv(OUT / "C_collinearity_drop_list.csv", index=False)
    R["collinearity_dropped"] = dropped
    R["collinearity_kept_numeric"] = kept

    # ---------------- Label validation vs UNOSAT (independent sensor, 2022-2024)
    try:
        R["unosat_validation"] = unosat_validation(df)
    except Exception as e:
        R["unosat_validation"] = f"failed: {e}"

    (OUT / "audit_summary.json").write_text(json.dumps(R, indent=2, default=str))
    figures(df)
    print(json.dumps(R, indent=2, default=str)[:6000])
    print("\nTop associations:\n", A.head(10).to_string())
    print("\nCollinear pairs:\n", pair_df.to_string())
    print("\nVIF:\n", vif.to_string())
    print("\nSkew/outliers:\n", sk[["feature", "skew", "rule", "pct_iqr_outliers", "pct_z3_outliers"]].to_string())


def unosat_validation(df):
    con = sqlite3.connect(ROOT / "data/raw/UNOSAT_SouthSudan_FloodExtent_2022-2024.gpkg")
    u = pd.read_sql("select period_start, Adm2, Area_km2 from 'UNOSAT_SouthSudan_FloodExtent_2022-2024' "
                    "where Adm2 is not null", con, parse_dates=["period_start"])
    fix = {"Cueibit": "Cueibet", "Gogrial Eest": "Gogrial East", "Kajo Keji": "Kajo-keji", "Lopa/Lafon": "Lafon",
           "Old Fangak": "Fangak", "Payinjar": "Panyijiar", "Peji": "Canal/Pigi", "Raga": "Raja", "Wuror": "Uror"}
    u["adm2_name"] = u["Adm2"].replace(fix)
    names = df[["adm2_name", "adm2_pcode"]].drop_duplicates()
    unmatched = sorted(set(u["adm2_name"]) - set(names["adm2_name"]))
    u = u.merge(names, on="adm2_name")
    # full county x date grid with true zeros for absent counties (UNOSAT omits dry counties)
    grid = pd.MultiIndex.from_product([names["adm2_pcode"], sorted(u["period_start"].unique())],
                                      names=["adm2_pcode", "period_start"]).to_frame(index=False)
    grid = grid.merge(u[["adm2_pcode", "period_start", "Area_km2"]], how="left").fillna({"Area_km2": 0})
    grid = grid[grid["adm2_pcode"] != "SS0001"]          # Abyei never monitored by UNOSAT
    s1 = df[["adm2_pcode", "window_start", "water_frac"]].copy()
    s1["window_end"] = s1["window_start"] + pd.Timedelta(days=11)
    m = grid.merge(s1, on="adm2_pcode")
    m = m[(m["period_start"] >= m["window_start"]) & (m["period_start"] <= m["window_end"])].dropna(subset=["water_frac"])
    pooled = stats.spearmanr(m["Area_km2"], m["water_frac"])
    per = m.groupby("adm2_pcode").apply(lambda d: stats.spearmanr(d["Area_km2"], d["water_frac"])[0]
                                        if d["Area_km2"].nunique() > 2 else np.nan, include_groups=False).dropna()
    st = df[["adm2_pcode", "adm1_name"]].drop_duplicates().set_index("adm2_pcode")["adm1_name"]
    by_state = per.groupby(per.index.map(st)).median().round(2).to_dict()
    nat = m.groupby("period_start").agg(u=("Area_km2", "sum"), s=("water_frac", "mean"))
    return {"unmatched_names_after_crosswalk": unmatched, "matched_pairs": int(len(m)),
            "per_state_median_temporal_spearman": by_state,
            "national_per_date_spearman": round(float(stats.spearmanr(nat["u"], nat["s"])[0]), 3),
            "n_dates": int(len(nat)),
            "pooled_spearman": round(float(pooled[0]), 3),
            "per_county_temporal_spearman_median": round(float(per.median()), 3),
            "per_county_n": int(len(per)), "pct_counties_positive_rho": round(float(100 * (per > 0).mean()), 1)}


def figures(df):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 100})
    # Fig 1.1 missingness pattern: county x time for water_frac + column-level bars
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1.2]})
    piv = df.pivot_table(index="adm2_pcode", columns="issue_date", values="wf_missing", aggfunc="max")
    ax[0].imshow(piv.values, aspect="auto", cmap="Greys", interpolation="nearest")
    xt = np.linspace(0, piv.shape[1] - 1, 10).astype(int)
    ax[0].set_xticks(xt); ax[0].set_xticklabels([str(piv.columns[i].date())[:7] for i in xt], rotation=45)
    ax[0].set_ylabel("County (79, sorted by pcode)"); ax[0].set_title("water_frac missing (black) by county and issue window")
    for d in ["2021-12-23", "2025-06-30"]:
        ax[0].axvline(np.searchsorted(piv.columns.values, np.datetime64(d)), color="tab:red", ls="--", lw=1)
    pm = df.drop(columns=["wf_missing", "year"]).isna().mean().mul(100).sort_values()
    pm = pm[pm > 0]
    ax[1].barh(pm.index, pm.values, color="tab:gray"); ax[1].set_xlabel("% missing")
    ax[1].set_title("Columns with missing values")
    fig.suptitle("Fig 1.1 - Missingness pattern (red dashes: Sentinel-1B loss, Sentinel-1C/D coverage)")
    fig.tight_layout(); fig.savefig(FIG / "fig1_1_missingness_heatmap.png", dpi=300); plt.close(fig)
    # Fig 1.2 target class balance per split
    fig, ax = plt.subplots(figsize=(7, 4))
    t = df.dropna(subset=["y_flood_2_4wk"]).groupby(["split", "y_flood_2_4wk"]).size().unstack().reindex(["train", "validation", "test"])
    t.plot.bar(ax=ax, color=["#9aa5b1", "#1f6feb"], rot=0)
    for i, (n0, n1) in enumerate(t.values):
        ax.text(i, max(n0, n1) * 1.02, f"pos {100 * n1 / (n0 + n1):.1f}%", ha="center")
    ax.set_ylabel("Labelled rows"); ax.legend(["Y=0 (below county P75)", "Y=1 (exceeds county P75)"])
    ax.set_title("Fig 1.2 - Target class balance by temporal split")
    fig.tight_layout(); fig.savefig(FIG / "fig1_2_target_distribution.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
