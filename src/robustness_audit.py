"""
SuddWatch Phase 3 - evaluation rigour and robustness audit (rubrics 2, 3 and 4 of the technical evaluation).

Nothing here tunes or selects anything on the test set: the deployed model, its calibration and its threshold are
loaded as they are, and every refit (ablation, county hold-out) uses TRAIN rows and is scored on VALIDATION rows.

  A. Confidence intervals: block bootstrap over issue dates (whole forecast cycles resampled, which keeps the
     spatial correlation between counties of the same cycle) for PR-AUC, Macro-F1, false-alarm rate, ECE and
     precision@k, for the model, the persistence baseline and their difference. Fold spread from tuning CV.
  B. Precision@k: of the k counties ranked highest in each forecast cycle, how many really flooded.
  C. Cost framing: expected cost of each decision rule when a missed flood costs r times a false alarm.
  D. Error slicing (test): by regime (wet now / dry now = flood persistence vs flood onset), by state, by
     county wetness, by month; model vs persistence on the same rows.
  E. Ablation and shortcut screen (validation): refit with each feature group neutralised (set to a constant);
     a "shortcut-only" model that sees just state, month and acquisition metadata.
  F. Group-disjoint check: train on 4/5 of counties (train years), score the unseen 1/5 (validation years).
  G. Stress tests (test, deployed model): radar outages of 1, 2 and 15 passes, water-measurement noise,
     rainfall bias of +/-20 %, fewer Sentinel-1 scenes.
  H. Permutation importance (validation, model fitted on train).

Usage: .venv\\Scripts\\python.exe -m src.robustness_audit      (about 3-5 minutes)
Outputs: phase3_modeling/audit/*.csv|json, figures/phase3_robustness.png
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from .calibration import prior_shift
from .pipeline_v2 import INPUT_COLUMNS, TARGET, TERRAIN, build_pipeline_v2
from .train_model import X_of, ece, false_alarm_rate, hgb, load, macro_f1

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase3_modeling/audit"
RES = ROOT / "phase3_modeling/results"
FIG = ROOT / "figures"
B = 2000
SEED = 0
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9aa5b1", "#0b0b0b"

GROUPS = {
    "water level now and past (Sentinel-1)": ["water_frac", "water_frac_last", "water_frac_days", "water_frac_lag1",
                                              "water_frac_lag1_last", "water_frac_lag1_days", "water_frac_12m_mean"],
    "county flood threshold": ["y_threshold_frac"],
    "rainfall (CHIRPS)": ["rain_36d_mm", "rain_90d_anom_mm"],
    "terrain and static county data": TERRAIN + ["county_area_km2", "jrc_water_occurrence_pct"],
    "state": ["adm1_name"],
    "season (month)": ["month"],
    "acquisition metadata (scenes, coverage)": ["n_scenes", "obs_frac"],
}
SHORTCUT_DROP = ["water level now and past (Sentinel-1)", "county flood threshold", "rainfall (CHIRPS)",
                 "terrain and static county data"]


# ----------------------------------------------------------------------------- helpers
def pr_auc(y, s):
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def recall(y, alert):
    y, a = np.asarray(y), np.asarray(alert)
    return float((a & (y == 1)).sum() / max((y == 1).sum(), 1))


def precision_at_k(df, score, k):
    """Mean over forecast cycles of the share of true floods among the k highest-scored counties."""
    return float(df.groupby("issue_date").apply(
        lambda g: g.nlargest(k, score)[TARGET].mean(), include_groups=False).mean())


def neutralise(df, cols, ref):
    """Remove a feature group's information: every row gets the training median (or one constant category)."""
    df = df.copy()
    for c in cols:
        if c == "adm1_name":
            df[c] = "ALL"
        elif c == "month":
            df[c] = 6
        else:
            df[c] = float(ref[c].median())
    return df


# ----------------------------------------------------------------------------- data
def frames():
    full, lab = load()
    tr, va, te = (lab[lab.split == s].copy() for s in ("train", "validation", "test"))
    preds = pd.read_csv(RES / "predictions.csv", parse_dates=["issue_date"])
    key = ["adm2_pcode", "issue_date"]
    va = va.merge(preds[preds.split == "validation"][key + ["p_cal", "p_dyn", "pi_known"]], on=key, how="left")
    te = te.merge(preds[preds.split == "test"][key + ["p_cal", "p_dyn", "pi_known"]], on=key, how="left")
    assert va["p_dyn"].notna().all() and te["p_dyn"].notna().all()
    return full, lab, tr, va, te


def persistence(pipe_first_step, df):
    """Persistence score = current (gap-filled) water level relative to the county threshold; alert if > 1."""
    filled = pipe_first_step.transform(X_of(df))["water_frac"].values
    return filled / df["y_threshold_frac"].values


# ----------------------------------------------------------------------------- A + B: CIs and precision@k
def _mf1(y, a):
    """Macro-F1 from 0/1 arrays (fast, for the bootstrap)."""
    tp = (a & (y == 1)).sum(); fp = (a & (y == 0)).sum(); fn = (~a & (y == 1)).sum(); tn = (~a & (y == 0)).sum()
    f_pos = 2 * tp / max(2 * tp + fp + fn, 1); f_neg = 2 * tn / max(2 * tn + fn + fp, 1)
    return (f_pos + f_neg) / 2, fp / max(fp + tn, 1)


def bootstrap(te, t_dyn):
    groups = [g for _, g in te.groupby("issue_date")]
    Y = [g[TARGET].to_numpy() for g in groups]
    PM = [g["p_dyn"].to_numpy() for g in groups]
    PP = [g["persist"].to_numpy() for g in groups]
    # precision@10 of one cycle does not change when cycles are resampled: compute it once per cycle
    pk_m = np.array([g.nlargest(10, "p_dyn")[TARGET].mean() for g in groups])
    pk_p = np.array([g.nlargest(10, "persist")[TARGET].mean() for g in groups])
    rng = np.random.default_rng(SEED)

    def stats(idx):
        y = np.concatenate([Y[i] for i in idx]); pm = np.concatenate([PM[i] for i in idx]); pp = np.concatenate([PP[i] for i in idx])
        f_m, fa_m = _mf1(y, pm >= t_dyn); f_p, fa_p = _mf1(y, pp > 1)
        return {"pr_auc_model": pr_auc(y, pm), "pr_auc_persistence": pr_auc(y, pp),
                "macro_f1_model": f_m, "macro_f1_persistence": f_p,
                "false_alarm_model": fa_m, "false_alarm_persistence": fa_p, "ece_model": ece(y, pm),
                "p_at_10_model": float(pk_m[idx].mean()), "p_at_10_persistence": float(pk_p[idx].mean())}
    dates = np.arange(len(groups))
    point = stats(dates)
    draws = pd.DataFrame([stats(rng.choice(dates, size=len(dates), replace=True)) for _ in range(B)])
    draws["pr_auc_difference"] = draws["pr_auc_model"] - draws["pr_auc_persistence"]
    draws["macro_f1_difference"] = draws["macro_f1_model"] - draws["macro_f1_persistence"]
    draws["p_at_10_difference"] = draws["p_at_10_model"] - draws["p_at_10_persistence"]
    point["pr_auc_difference"] = point["pr_auc_model"] - point["pr_auc_persistence"]
    point["macro_f1_difference"] = point["macro_f1_model"] - point["macro_f1_persistence"]
    point["p_at_10_difference"] = point["p_at_10_model"] - point["p_at_10_persistence"]
    ci = pd.DataFrame({"metric": list(point), "estimate": list(point.values()),
                       "ci95_low": [draws[m].quantile(0.025) for m in point],
                       "ci95_high": [draws[m].quantile(0.975) for m in point],
                       "share_of_draws_above_0": [float((draws[m] > 0).mean()) if "difference" in m else np.nan for m in point]})
    pk = []
    for k in (5, 10, 20):
        pk.append({"k": k, "model": precision_at_k(te, "p_dyn", k), "persistence": precision_at_k(te, "persist", k),
                   "random": float(te[TARGET].mean())})
    return ci, pd.DataFrame(pk), len(groups)


# ----------------------------------------------------------------------------- C: cost framing
def cost_table(va, t_dyn):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    y = va[TARGET].values
    rows = []
    for r in (1, 2, 3, 5, 10):
        def cost(alert):
            fp = ((alert == 1) & (y == 0)).sum(); fn = ((alert == 0) & (y == 1)).sum()
            return (fp + r * fn) / len(y)
        c_model = {t: cost((va["p_dyn"].values >= t).astype(int)) for t in grid}
        t_best = min(c_model, key=c_model.get)
        rows.append({"miss_cost_over_false_alarm_cost": r, "cost_deployed_threshold": c_model[round(t_dyn, 2)],
                     "deployed_threshold": t_dyn, "cost_optimal_threshold_on_validation": t_best,
                     "cost_at_optimal_threshold": c_model[t_best], "theory_optimal_threshold_1_over_1_plus_r": round(1 / (1 + r), 3),
                     "cost_persistence_rule": cost((va["persist"].values > 1).astype(int)),
                     "cost_alert_nobody": cost(np.zeros_like(y)), "cost_alert_everybody": cost(np.ones_like(y))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- D: slicing
def slices(te, t_dyn):
    te = te.copy()
    te["regime"] = np.where(te["persist"] > 1, "wet now (persistence / recession cases)", "dry now (onset cases)")
    te["county_wetness"] = pd.qcut(te["y_threshold_frac"], 3, labels=["dry counties (low threshold)", "middle", "wet counties (high threshold)"])
    te["season"] = np.where(te["month"].between(5, 10), "rainy season (May-Oct)", "dry season (Nov-Apr)")
    out = []
    for dim in ("regime", "adm1_name", "county_wetness", "season", "sensor_era"):
        for lvl, g in te.groupby(dim, observed=True):
            y = g[TARGET].values
            out.append({"dimension": dim, "slice": lvl, "rows": len(g), "positives": int(y.sum()), "prevalence": y.mean(),
                        "pr_auc_model": pr_auc(y, g["p_dyn"]), "pr_auc_persistence": pr_auc(y, g["persist"]),
                        "recall_model": recall(y, g["p_dyn"] >= t_dyn), "recall_persistence": recall(y, g["persist"] > 1),
                        "false_alarm_model": false_alarm_rate(y, g["p_dyn"].values, t_dyn),
                        "false_alarm_persistence": false_alarm_rate(y, g["persist"].values, 1.0 + 1e-12),
                        "macro_f1_model": macro_f1(y, g["p_dyn"].values, t_dyn)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- E + F: refits (train -> validation)
def ablation(tr, va, best):
    def fit_score(a, b):
        m = build_pipeline_v2(hgb(best)).fit(X_of(a), a[TARGET])
        return pr_auc(b[TARGET], m.predict_proba(X_of(b))[:, 1])
    full_score = fit_score(tr, va)
    rows = [{"variant": "full model (all features)", "neutralised": "-", "val_pr_auc": full_score}]
    for name, cols in GROUPS.items():
        s = fit_score(neutralise(tr, cols, tr), neutralise(va, cols, tr))
        rows.append({"variant": f"without {name}", "neutralised": ", ".join(cols), "val_pr_auc": s})
        print(f"  ablation without {name}: {s:.4f}", flush=True)
    cols = sum((GROUPS[g] for g in SHORTCUT_DROP), [])
    s = fit_score(neutralise(tr, cols, tr), neutralise(va, cols, tr))
    rows.append({"variant": "shortcut-only (state, month, acquisition metadata)", "neutralised": ", ".join(cols), "val_pr_auc": s})
    df = pd.DataFrame(rows)
    df["delta_vs_full"] = df["val_pr_auc"] - full_score
    return df


def group_disjoint(tr, va, best):
    counties = np.array(sorted(tr["adm2_pcode"].unique()))
    gkf = GroupKFold(n_splits=5)
    rows, held = [], []
    for k, (_, test_idx) in enumerate(gkf.split(counties, groups=counties)):
        hold = set(counties[test_idx])
        a = tr[~tr["adm2_pcode"].isin(hold)]
        b = va[va["adm2_pcode"].isin(hold)]
        m = build_pipeline_v2(hgb(best)).fit(X_of(a), a[TARGET])
        p = m.predict_proba(X_of(b))[:, 1]
        held.append(b.assign(p_unseen=p))
        rows.append({"fold": k + 1, "held_out_counties": len(hold), "rows": len(b), "prevalence": b[TARGET].mean(),
                     "pr_auc_unseen_counties": pr_auc(b[TARGET], p), "pr_auc_persistence_same_rows": pr_auc(b[TARGET], b["persist"])})
    held = pd.concat(held)
    m_all = build_pipeline_v2(hgb(best)).fit(X_of(tr), tr[TARGET])
    seen = pr_auc(va[TARGET], m_all.predict_proba(X_of(va))[:, 1])
    return pd.DataFrame(rows), {"pr_auc_unseen_counties_pooled": pr_auc(held[TARGET], held["p_unseen"]),
                                "pr_auc_seen_counties_same_rows": seen}, m_all


# ----------------------------------------------------------------------------- G: stress tests (deployed model)
def stress(te, full, bundle):
    pipe, platt, t = bundle["pipeline"], bundle["platt"], bundle["threshold_dynamic"]

    def score(df):
        p = platt(pipe.predict_proba(df[INPUT_COLUMNS])[:, 1])
        return prior_shift(p, platt.base_prior, df["pi_known"].clip(0.02, 0.98).values)

    def summary(name, df, note):
        p = score(df); y = df[TARGET].values
        return {"scenario": name, "pr_auc": pr_auc(y, p), "macro_f1": macro_f1(y, p, t),
                "false_alarm": false_alarm_rate(y, p, t), "recall": recall(y, p >= t), "alerts": int((p >= t).sum()), "how": note}

    rng = np.random.default_rng(SEED)
    rows = [summary("baseline (as deployed)", te, "unchanged test inputs")]

    a = te.copy()                                           # current pass missed: last observation is one window older
    a["water_frac"] = np.nan
    a["water_frac_last"], a["water_frac_days"] = a["water_frac_lag1_last"], a["water_frac_lag1_days"] + 12
    rows.append(summary("radar outage: current pass missed", a, "water_frac removed; last observation = previous window's"))

    b = a.copy()                                            # two passes missed
    b["water_frac_lag1"] = np.nan
    b["water_frac_lag1_last"], b["water_frac_lag1_days"] = te["water_frac_lag2_last"], te["water_frac_lag2_days"] + 12
    b["water_frac_last"], b["water_frac_days"] = te["water_frac_lag2_last"], te["water_frac_lag2_days"] + 24
    rows.append(summary("radar outage: two passes missed (24 days)", b, "current and previous water removed"))

    f = full.sort_values(["adm2_pcode", "window_start"]).copy()       # six-month outage (like the 2022-2025 gaps)
    g = f.groupby("adm2_pcode")
    f["last_15"], f["days_15"] = g["water_frac_last"].shift(15), g["water_frac_days"].shift(15)
    c = te.merge(f[["adm2_pcode", "issue_date", "last_15", "days_15"]], on=["adm2_pcode", "issue_date"], how="left")
    c["water_frac"], c["water_frac_lag1"] = np.nan, np.nan
    c["water_frac_last"], c["water_frac_lag1_last"] = c["last_15"], c["last_15"]
    c["water_frac_days"], c["water_frac_lag1_days"] = c["days_15"] + 180, c["days_15"] + 168
    rows.append(summary("radar outage: six months (15 passes)", c, "only observations older than 180 days remain"))

    d = te.copy()
    for col in ("water_frac", "water_frac_last", "water_frac_lag1", "water_frac_lag1_last"):
        d[col] = d[col] * np.exp(rng.normal(0, 0.2, len(d)))
    rows.append(summary("water measurement noise (+/-20 %)", d, "log-normal noise, sd 0.2, on all water inputs"))

    for k, lab_ in ((0.8, "-20 %"), (1.2, "+20 %")):
        e = te.copy()
        e["rain_36d_mm"] = e["rain_36d_mm"] * k
        e["rain_90d_anom_mm"] = e["rain_90d_anom_mm"] + (k - 1) * e["rain_90d_mm"]
        rows.append(summary(f"rainfall bias {lab_}", e, "CHIRPS totals scaled; 90-day anomaly shifted consistently"))

    h = te.copy()
    h["n_scenes"], h["obs_frac"] = 1, h["obs_frac"] * 0.5
    rows.append(summary("fewer Sentinel-1 scenes", h, "one scene per window, half the county observed"))
    df = pd.DataFrame(rows)
    df["delta_pr_auc"] = df["pr_auc"] - df.loc[0, "pr_auc"]
    return df


# ----------------------------------------------------------------------------- figure
def figure(ci, sl, abl, st):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # (a) CIs
    a = ax[0, 0]
    mets = [("pr_auc", "PR-AUC"), ("macro_f1", "Macro-F1"), ("p_at_10", "precision@10")]
    civ = ci.set_index("metric")
    for i, (m, lab_) in enumerate(mets):
        for j, (who, col) in enumerate((("model", BLUE), ("persistence", GRAY))):
            r = civ.loc[f"{m}_{who}"]
            yv = i * 1.0 + (0.18 if j == 0 else -0.18)
            a.plot([r.ci95_low, r.ci95_high], [yv, yv], color=col, lw=2)
            a.plot(r.estimate, yv, "o", color=col, ms=7, label=("SuddWatch model" if j == 0 else "persistence baseline") if i == 0 else None)
    a.set_yticks(range(len(mets))); a.set_yticklabels([l for _, l in mets])
    a.set_xlim(0.6, 1.0); a.legend(frameon=False, loc="lower left")
    a.set_title("Test metrics, 95% block-bootstrap intervals (2,000 resamples of cycles)", loc="left", fontsize=9.5)
    # (b) regime slice
    b = ax[0, 1]
    rs = sl[sl.dimension == "regime"].set_index("slice")
    x = np.arange(len(rs)); w = 0.36
    b.bar(x - w / 2, rs["recall_model"], w, color=BLUE, label="model: share of floods caught")
    b.bar(x + w / 2, rs["recall_persistence"], w, color=GRAY, label="persistence: share of floods caught")
    for i, (_, r) in enumerate(rs.iterrows()):
        b.text(i - w / 2, r.recall_model + 0.02, f"{r.recall_model:.0%}", ha="center", fontsize=9)
        b.text(i + w / 2, r.recall_persistence + 0.02, f"{r.recall_persistence:.0%}", ha="center", fontsize=9)
    short = {"dry now (onset cases)": "dry now: new flood onsets", "wet now (persistence / recession cases)": "wet now: flood persists or recedes"}
    b.set_xticks(x); b.set_xticklabels([f"{short.get(s, s)}\n{int(r.positives)} floods in {int(r.rows)} county-cycles" for s, r in rs.iterrows()], fontsize=8.5)
    b.set_ylim(0, 1.12); b.legend(frameon=False, loc="upper left", fontsize=8.5)
    b.set_title("Where the model adds value: floods caught by regime (test)", loc="left", fontsize=9.5)
    # (c) ablation
    c = ax[1, 0]
    ab = abl[abl.variant != "full model (all features)"].sort_values("delta_vs_full")
    c.barh(ab["variant"].str.replace("without ", "– "), ab["delta_vs_full"], color=[ORANGE if "shortcut" in v else BLUE for v in ab["variant"]])
    c.axvline(0, color=INK, lw=0.8)
    c.set_xlabel("change in validation PR-AUC vs full model")
    c.set_title(f"Ablation and shortcut screen (full model validation PR-AUC {abl.iloc[0].val_pr_auc:.3f})", loc="left", fontsize=9.5)
    c.tick_params(axis="y", labelsize=8.5)
    # (d) stress
    d = ax[1, 1]
    s2 = st.iloc[1:].sort_values("delta_pr_auc")
    d.barh(s2["scenario"], s2["delta_pr_auc"], color=BLUE)
    d.axvline(0, color=INK, lw=0.8)
    d.set_xlabel("change in test PR-AUC vs deployed baseline")
    d.set_title(f"Stress tests, deployed model (baseline test PR-AUC {st.iloc[0].pr_auc:.3f})", loc="left", fontsize=9.5)
    d.tick_params(axis="y", labelsize=8.5)
    for a_ in (c, d):
        a_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        a_.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}" if v else "0"))
    for a_ in ax.ravel():
        a_.grid(color="#e6e6e3", lw=0.6, axis="x"); a_.set_axisbelow(True)
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "phase3_robustness.png", dpi=200, facecolor="white"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    full, lab, tr, va, te = frames()
    best = json.loads((RES / "training_summary.json").read_text())["best_params"]
    bundle = joblib.load(ROOT / "models/suddwatch_model.joblib")
    t_dyn = bundle["threshold_dynamic"]
    # persistence baselines: gap filling fitted on train (validation rows) / train+validation (test rows), as in training
    va["persist"] = persistence(build_pipeline_v2(hgb(best))[:1].fit(X_of(tr)), va)
    te["persist"] = persistence(bundle["pipeline"][:1], te)

    print("A/B bootstrap CIs and precision@k ...", flush=True)
    ci, pk, n_dates = bootstrap(te, t_dyn)
    ci.round(4).to_csv(OUT / "ci_bootstrap_test.csv", index=False); pk.round(4).to_csv(OUT / "precision_at_k_test.csv", index=False)
    trials = pd.read_csv(RES / "tuning_trials.csv").iloc[0]
    print("C cost framing ...", flush=True)
    cost = cost_table(va, t_dyn); cost.round(4).to_csv(OUT / "cost_analysis_validation.csv", index=False)
    print("D slicing ...", flush=True)
    sl = slices(te, t_dyn); sl.round(4).to_csv(OUT / "error_slices_test.csv", index=False)
    print("E ablation ...", flush=True)
    abl = ablation(tr, va, best); abl.round(4).to_csv(OUT / "ablation_validation.csv", index=False)
    print("F county hold-out ...", flush=True)
    gd, gd_sum, m_tr = group_disjoint(tr, va, best); gd.round(4).to_csv(OUT / "county_holdout_validation.csv", index=False)
    print("G stress tests ...", flush=True)
    st = stress(te, full, bundle); st.round(4).to_csv(OUT / "stress_tests_test.csv", index=False)
    print("H permutation importance ...", flush=True)
    pi = permutation_importance(m_tr, X_of(va), va[TARGET], scoring="average_precision", n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": INPUT_COLUMNS, "pr_auc_drop_mean": pi.importances_mean, "pr_auc_drop_sd": pi.importances_std}) \
        .sort_values("pr_auc_drop_mean", ascending=False)
    imp.round(4).to_csv(OUT / "permutation_importance_validation.csv", index=False)
    figure(ci, sl, abl, st)

    summary = {"bootstrap": {"resamples": B, "blocks": "issue dates (forecast cycles)", "test_cycles": n_dates,
                             "ci95": ci.set_index("metric").round(4).to_dict(orient="index")},
               "tuning_cv_fold_spread": {"cv_pr_auc_mean": round(float(trials.cv_pr_auc), 4), "cv_pr_auc_sd": round(float(trials.cv_sd), 4), "folds": 5},
               "precision_at_k_test": pk.round(4).to_dict(orient="records"),
               "cost_analysis_validation": cost.round(4).to_dict(orient="records"),
               "regime_slices_test": sl[sl.dimension == "regime"].round(4).to_dict(orient="records"),
               "ablation_validation": abl[["variant", "val_pr_auc", "delta_vs_full"]].round(4).to_dict(orient="records"),
               "county_holdout": {k: round(v, 4) for k, v in gd_sum.items()},
               "stress_tests_test": st.drop(columns="how").round(4).to_dict(orient="records"),
               "top_permutation_importance": imp.head(8).round(4).to_dict(orient="records"),
               "runtime_s": round(time.time() - t0)}
    (OUT / "robustness_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(ci.round(3).to_string(index=False)); print(pk.round(3).to_string(index=False))
    print(cost.round(3).to_string(index=False)); print(sl.round(3).to_string(index=False))
    print(abl.round(3).to_string(index=False)); print(gd.round(3).to_string(index=False), gd_sum)
    print(st.round(3).to_string(index=False)); print(imp.head(10).round(4).to_string(index=False))
    print("runtime", summary["runtime_s"], "s")


if __name__ == "__main__":
    main()
