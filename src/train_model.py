"""
SuddWatch Phase 3 - model training, calibration, thresholds and year-by-year evaluation.

Protocol (leakage-safe; the test set is scored exactly once, at the end):
  1. Splits as Phase 2 (train < 2024-01-01 <= validation < 2025-07-01 <= test), 2 boundary windows purged.
  2. Tune HistGradientBoosting on TRAIN with TimeSeriesSplit(5, gap = 2 windows); compare with persistence,
     climatology and logistic-regression baselines, the Phase 2 pipeline and the VH -26 dB variant (validation).
  3. Out-of-sample predictions for 2018-2023 by expanding years (fit on years < Y); 2017 by leave-one-year-out
     (retrospective only). These fit the Platt calibration - no validation or test rows.
  4. Thresholds chosen on VALIDATION: static (max Macro-F1) and dynamic (prior-shift correction with the
     national positive rate of labels already known at the issue date, trailing 30 windows).
  5. Final model = same hyper-parameters refit on train + validation; scored once on TEST.
  6. Report PR-AUC, Macro-F1, false-alarm rate and ECE per calendar year (2017-2026) and per sensor era.

Usage: .venv\\Scripts\\python.exe -m src.train_model
Outputs: models/suddwatch_model.joblib, phase3_modeling/results/*.json|csv, figures/phase3_*.png
"""
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit

from . import pipeline_builder as PB2
from .calibration import Platt, prior_shift
from .pipeline_v2 import INPUT_COLUMNS, TAU_DAYS, TARGET, build_pipeline_v2

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/panel_v2_features.csv"
THR = ROOT / "data/raw/county_thresholds_train_only.csv"
RES = ROOT / "phase3_modeling/results"
MODELS = ROOT / "models"
FIG = ROOT / "figures"
TRAIN_END, VAL_END, GAP = "2024-01-01", "2025-07-01", 2
SEED = 0
BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#9aa5b1", "#0b0b0b"
PHASE2_COLS = [c for c in pd.read_csv(ROOT / "data/raw/suddwatch_panel_unprocessed.csv", nrows=1).columns if c != TARGET]


# ----------------------------------------------------------------------------- data
def load():
    d = pd.read_csv(PANEL, parse_dates=["window_start", "issue_date"])
    if "y_threshold_frac" not in d.columns:      # features_v2 already carries the county threshold
        d = d.merge(pd.read_csv(THR)[["adm2_pcode", "y_threshold_frac"]], on="adm2_pcode", how="left")
    windows = np.sort(d["window_start"].unique())
    d["w_idx"] = np.searchsorted(windows, d["window_start"].values)
    lab = d.dropna(subset=[TARGET]).copy()
    lab[TARGET] = lab[TARGET].astype(int)

    def last_k(before):
        w = windows[windows < np.datetime64(before)]
        return set(w[-GAP:])
    purged = last_k(TRAIN_END) | last_k(VAL_END)
    lab = lab[~lab["window_start"].isin(purged)].sort_values(["issue_date", "adm2_pcode"]).reset_index(drop=True)
    return d, lab


def X_of(df, vh=False):
    return df[INPUT_COLUMNS + (["vh26_frac"] if vh else [])]


# ----------------------------------------------------------------------------- metrics
def macro_f1(y, p, t):
    return f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)


def false_alarm_rate(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return fp / max(fp + tn, 1)


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


MAX_FALSE_ALARM = 0.15   # proposal success metric: < 15% false-alarm rate, enforced on VALIDATION


def best_threshold(y, p):
    """Highest Macro-F1 among thresholds whose false-alarm rate on these (validation) rows is <= 15%."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    ok = [t for t in grid if false_alarm_rate(y, p, t) <= MAX_FALSE_ALARM]
    scores = [macro_f1(y, p, t) for t in ok]
    return float(ok[int(np.argmax(scores))])


# ----------------------------------------------------------------------------- prior shift helpers
def known_prevalence(lab, window=30):
    """For each window index i: positive rate of labels KNOWN at the end of window i, i.e. rows j <= i-2
    (their target window j+2 has ended), over the trailing `window` windows. National, all counties."""
    rate = lab.groupby("w_idx")[TARGET].agg(["sum", "count"])
    idx = np.arange(0, lab["w_idx"].max() + 1)
    rate = rate.reindex(idx, fill_value=0)
    s = rate["sum"].rolling(window, min_periods=10).sum().shift(GAP)
    n = rate["count"].rolling(window, min_periods=10).sum().shift(GAP)
    return (s / n).ffill()


# ----------------------------------------------------------------------------- models
def hgb(params):
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params)


SPACE = {"learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50, 100],
         "l2_regularization": [0.0, 0.1, 1.0], "max_iter": [200, 400], "class_weight": [None, "balanced"]}


def cv_ap(make, X, y, n_splits=5):
    cv = TimeSeriesSplit(n_splits=n_splits, gap=GAP * 79)
    out = []
    for tr, te in cv.split(X):
        m = make().fit(X.iloc[tr], y.iloc[tr])
        out.append(average_precision_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(out)), float(np.std(out))


def main():
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True); MODELS.mkdir(exist_ok=True)
    full, lab = load()
    tr, va, te = (lab[lab.split == s] for s in ("train", "validation", "test"))
    Xtr, ytr = X_of(tr), tr[TARGET]
    log = {}

    # 1. tuning on train (time-series CV)
    trials = []
    for i, params in enumerate(ParameterSampler(SPACE, n_iter=20, random_state=SEED)):
        m, s = cv_ap(lambda: build_pipeline_v2(hgb(params)), Xtr, ytr)
        trials.append({**params, "cv_pr_auc": m, "cv_sd": s})
        print(f"  trial {i + 1:2d}/20 cv PR-AUC {m:.4f} +/- {s:.4f}  {params}", flush=True)
    trials = pd.DataFrame(trials).sort_values("cv_pr_auc", ascending=False)
    trials.to_csv(RES / "tuning_trials.csv", index=False)
    best = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in trials.iloc[0][list(SPACE)].to_dict().items()}
    best["max_leaf_nodes"], best["min_samples_leaf"], best["max_iter"] = int(best["max_leaf_nodes"]), int(best["min_samples_leaf"]), int(best["max_iter"])
    log["best_params"] = best
    print("best:", best, flush=True)

    # 2. model comparison on validation (all fitted on train only)
    thr_map = full.drop_duplicates("adm2_pcode").set_index("adm2_pcode")["y_threshold_frac"]
    clim = tr.groupby(["adm2_pcode", "month"])[TARGET].mean()
    comp = {}
    def val_ap(scores):
        return round(float(average_precision_score(va[TARGET], scores)), 4)
    filled = build_pipeline_v2(hgb(best))[:1].fit(Xtr).transform(X_of(va))["water_frac"]   # seasonal-blend filled water
    comp["Persistence (flooded now -> flooded in 2 windows)"] = val_ap(filled.values / va["adm2_pcode"].map(thr_map).values)
    comp["Climatology (county x month training rate)"] = val_ap(pd.MultiIndex.from_frame(va[["adm2_pcode", "month"]]).map(clim).fillna(ytr.mean()))
    lr = build_pipeline_v2(LogisticRegression(class_weight="balanced", max_iter=3000)).fit(Xtr, ytr)
    comp["Logistic regression, Phase 3 features"] = val_ap(lr.predict_proba(X_of(va))[:, 1])
    p2 = PB2.build_pipeline(hgb(best), scale_tree_inputs=False).fit(tr[PHASE2_COLS], ytr)
    comp["Gradient boosting, Phase 2 preprocessing"] = val_ap(p2.predict_proba(va[PHASE2_COLS])[:, 1])
    m_tr = build_pipeline_v2(hgb(best)).fit(Xtr, ytr)
    p_va_raw = m_tr.predict_proba(X_of(va))[:, 1]
    comp["Gradient boosting, Phase 3 preprocessing (selected)"] = val_ap(p_va_raw)
    vh = build_pipeline_v2(hgb(best), vh=True).fit(X_of(tr, vh=True), ytr)
    comp["Gradient boosting, Phase 3 + VH -26 dB feature"] = val_ap(vh.predict_proba(X_of(va, vh=True))[:, 1])
    log["validation_comparison"] = comp
    log["validation_prevalence"] = round(float(va[TARGET].mean()), 4)
    print(json.dumps(comp, indent=1), flush=True)
    use_vh = comp["Gradient boosting, Phase 3 + VH -26 dB feature"] >= comp["Gradient boosting, Phase 3 preprocessing (selected)"] + 0.01
    log["vh_adopted"] = bool(use_vh)

    # 3. out-of-sample predictions on training years (expanding by year; 2017 leave-one-year-out)
    rows = []
    for Y in range(2017, 2024):
        test_rows = tr[tr.issue_date.dt.year == Y]
        if Y == 2017:
            fit_rows = tr[tr.issue_date.dt.year > 2017]
            fit_rows = fit_rows[fit_rows.w_idx > test_rows.w_idx.max() + GAP]
            source = "train (retrospective: fitted on 2018-2023)"
        else:
            fit_rows = tr[tr.w_idx < test_rows.w_idx.min() - GAP]
            source = f"train (out-of-sample: fitted on 2017-{Y - 1})"
        m = build_pipeline_v2(hgb(best), vh=use_vh).fit(X_of(fit_rows, use_vh), fit_rows[TARGET])
        rows.append(test_rows.assign(p_raw=m.predict_proba(X_of(test_rows, use_vh))[:, 1], source=source))
    oof = pd.concat(rows)
    platt_a = Platt().fit(oof.loc[oof.issue_date.dt.year >= 2018, "p_raw"].values, oof.loc[oof.issue_date.dt.year >= 2018, TARGET].values)

    # 4. thresholds on validation
    m_tr = build_pipeline_v2(hgb(best), vh=use_vh).fit(X_of(tr, use_vh), ytr)
    va = va.assign(p_raw=m_tr.predict_proba(X_of(va, use_vh))[:, 1], source="validation (fitted on 2017-2023)")
    prev = known_prevalence(lab)
    def finish(df, cal):
        df = df.assign(p_cal=cal(df["p_raw"].values))
        pi_t = df["w_idx"].map(prev).fillna(cal.base_prior).clip(0.02, 0.98)
        return df.assign(pi_known=pi_t.values, p_dyn=prior_shift(df["p_cal"].values, cal.base_prior, pi_t.values))
    va = finish(va, platt_a)
    t_static = best_threshold(va[TARGET], va["p_cal"])
    t_dynamic = best_threshold(va[TARGET], va["p_dyn"])
    log["thresholds"] = {"static": t_static, "dynamic": t_dynamic, "calibration_base_prior": round(platt_a.base_prior, 4)}

    # 5. final model on train + validation, test scored once
    trva = pd.concat([tr, lab[lab.split == "validation"]])
    final = build_pipeline_v2(hgb(best), vh=use_vh).fit(X_of(trva, use_vh), trva[TARGET])
    platt_b = Platt().fit(np.r_[oof.loc[oof.issue_date.dt.year >= 2018, "p_raw"].values, va["p_raw"].values],
                          np.r_[oof.loc[oof.issue_date.dt.year >= 2018, TARGET].values, va[TARGET].values])
    te = finish(te.assign(p_raw=final.predict_proba(X_of(te, use_vh))[:, 1], source="test (fitted on 2017 - mid-2025)"), platt_b)
    oof = finish(oof, platt_a)
    preds = pd.concat([oof, va, te])
    preds["year"] = preds.issue_date.dt.year
    preds["period"] = preds["year"].astype(str) + np.where(preds["split"] == "validation", " (validation)",
                                                     np.where(preds["split"] == "test", " (test)", ""))

    # 6. reporting
    def summarize(g):
        y = g[TARGET].values
        ok = len(np.unique(y)) > 1
        return pd.Series({"rows": len(g), "prevalence": y.mean(),
                          "pr_auc": average_precision_score(y, g["p_cal"]) if ok else np.nan,
                          "macro_f1_static": macro_f1(y, g["p_cal"], t_static),
                          "macro_f1_dynamic": macro_f1(y, g["p_dyn"], t_dynamic),
                          "false_alarm_static": false_alarm_rate(y, g["p_cal"], t_static),
                          "false_alarm_dynamic": false_alarm_rate(y, g["p_dyn"], t_dynamic),
                          "ece_static": ece(y, g["p_cal"]), "ece_dynamic": ece(y, g["p_dyn"]),
                          "source": g["source"].iloc[0]})
    order = preds.drop_duplicates("period").assign(k=lambda d: d["year"] * 10 + d["split"].map({"train": 0, "validation": 1, "test": 2}))
    by_period = preds.groupby("period").apply(summarize, include_groups=False).reindex(order.sort_values("k")["period"]).reset_index()
    by_era = preds.groupby("sensor_era").apply(summarize, include_groups=False).reset_index()
    by_split = preds.assign(s=preds["split"]).groupby("s").apply(summarize, include_groups=False).reset_index()
    for name, df in [("metrics_by_year", by_period), ("metrics_by_sensor_era", by_era), ("metrics_by_split", by_split)]:
        df.round(4).to_csv(RES / f"{name}.csv", index=False)
    test_summary = by_split.set_index("s").loc["test"].drop("source").astype(float).round(4).to_dict()
    log["test"] = test_summary
    log["validation"] = by_split.set_index("s").loc["validation"].drop("source").astype(float).round(4).to_dict()
    # persistence & climatology on test for honest comparison (no fitting on test)
    fil_te = build_pipeline_v2(hgb(best))[:1].fit(X_of(trva)).transform(X_of(te))["water_frac"]
    clim2 = trva.groupby(["adm2_pcode", "month"])[TARGET].mean()
    log["test_baselines_pr_auc"] = {
        "persistence": round(float(average_precision_score(te[TARGET], fil_te.values / te["adm2_pcode"].map(thr_map).values)), 4),
        "climatology": round(float(average_precision_score(te[TARGET], pd.MultiIndex.from_frame(te[["adm2_pcode", "month"]]).map(clim2).fillna(trva[TARGET].mean()))), 4),
        "random (prevalence)": round(float(te[TARGET].mean()), 4)}
    preds[["adm2_pcode", "issue_date", "split", "source", "sensor_era", TARGET, "p_raw", "p_cal", "pi_known", "p_dyn"]].to_csv(RES / "predictions.csv", index=False)

    # figures
    yr = by_period.copy()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(yr))
    ax[0].bar(x, yr["pr_auc"], color=[GRAY if "test" not in p and "validation" not in p else (ORANGE if "test" in p else BLUE) for p in yr["period"]], width=0.65)
    ax[0].plot(x, yr["prevalence"], color=INK, marker="o", ms=4, lw=1, label="prevalence (= random PR-AUC)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(yr["period"], rotation=45, ha="right", fontsize=8)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("PR-AUC"); ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("PR-AUC by year (gray: out-of-sample training years, blue: validation, orange: test)", loc="left", fontsize=9)
    ax[1].plot(x, yr["macro_f1_static"], color=GRAY, marker="o", ms=4, label=f"static threshold {t_static}")
    ax[1].plot(x, yr["macro_f1_dynamic"], color=BLUE, marker="o", ms=4, label=f"dynamic threshold {t_dynamic}")
    ax[1].plot(x, yr["ece_dynamic"], color=ORANGE, marker="s", ms=4, ls="--", label="ECE (dynamic)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(yr["period"], rotation=45, ha="right", fontsize=8)
    ax[1].set_ylim(0, 1); ax[1].legend(frameon=False, loc="upper left")
    ax[1].set_title("Macro-F1 and calibration error by year", loc="left", fontsize=9)
    for a in ax:
        a.grid(color="#e6e6e3", lw=0.6); a.set_axisbelow(True)
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "phase3_metrics_by_year.png", dpi=200, facecolor="white"); plt.close(fig)

    # save the serving bundle
    pi_latest = float(prev.iloc[-1])
    bundle = {"pipeline": final, "platt": platt_b, "threshold_static": t_static, "threshold_dynamic": t_dynamic,
              "pi_latest_known": pi_latest, "uses_vh": use_vh, "input_columns": INPUT_COLUMNS + (["vh26_frac"] if use_vh else []),
              "target": TARGET, "horizon": "window w+2 (13-24 days after the issue date)",
              "trained_on": f"{trva.issue_date.min().date()} to {trva.issue_date.max().date()}",
              "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
              "params": best, "tau_days": TAU_DAYS, "metrics": {"test": test_summary, "validation": log["validation"]}}
    joblib.dump(bundle, MODELS / "suddwatch_model.joblib")
    log["pi_latest_known"] = round(pi_latest, 4)
    log["runtime_s"] = round(time.time() - t0)
    (RES / "training_summary.json").write_text(json.dumps(log, indent=2, default=str))
    print(json.dumps(log, indent=2, default=str))
    print(by_period.round(3).to_string(index=False))
    print(by_era.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
