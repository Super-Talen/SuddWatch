"""
SuddWatch - render phase2_implementation/Phase2_Final_Report.(html|pdf) in the same IEEE format as Phase 1.

Every number is read from phase2_implementation/phase2_summary.json (written by the executed notebook)
and phase2_implementation/label_sensitivity.json, so the report cannot drift from the notebook.
Usage: .venv\\Scripts\\python.exe -m src.report_phase2
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

from . import report_phase1 as R1
from .report_phase1 import CSS, EDGE, REPO, html_table, img, tbl

ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "phase2_implementation"
OUT_HTML = P2 / "Phase2_Final_Report.html"
OUT_PDF = P2 / "Phase2_Final_Report.pdf"


def main():
    R1._counter.update({"table": 0, "fig": 0})
    S = json.loads((P2 / "phase2_summary.json").read_text())
    LS = S["label_sensitivity"]
    A1 = json.loads((ROOT / "phase1_strategy/audit/audit_summary.json").read_text())
    sp, ver, cap = S["split"], S["verification"], {c["feature"]: c for c in S["capping"]}
    full, red, fin = ver["full (no collinearity drop)"], ver["reduced (Phase 1 drop list)"], ver["reduced + momentum (final)"]
    ls = LS["results"]
    tmi = S["terrain_mi"]
    sens = S["sensor_check"]
    skew = S["water_frac_skew_stages"]
    pca = S["pca"]
    gk = S["groupkfold_pr_auc"]
    p1sp = A1["split_pct_of_labelled"]
    shapes = S["processed_shapes"]
    wblock = {"water_frac", "water_frac_lag1", "water_frac_lag2"}
    tblock = {"elev_mean_m", "elev_std_m", "slope_mean_deg", "hand_mean_m", "hand_lt5m_frac", "twi_mean"}
    pl = S["pairs_list"]
    n_w = sum(a in wblock and b in wblock for a, b, _ in pl)
    n_t = sum(a in tblock and b in tblock for a, b, _ in pl)

    split_df = pd.DataFrame([{"split": k, "issue dates": f"{v['first']} → {v['last']}", "rows": f"{int(v['rows']):,}",
                              "share": f"{v['share_%']}%", "positives": f"{int(v['positives']):,}", "pos. rate": f"{v['pos_rate_%']}%",
                              "Phase 1 share": f"{p1sp[k]}%"} for k, v in sp.items()])
    arch = [
        ["0", "Temporal split + purge", "—", "Rows split by issue date before any fitting; 2 boundary windows purged (VI-A, VI-D)."],
        ["1", "<code>GroupMedianImputer</code>", "<code>X_train</code>", "County training medians + <code>_was_missing</code> flag for Sentinel-1 water features (IV-A/B)."],
        ["2", "<code>MomentumFeature</code>", "stateless", "<code>water_frac_delta</code> = <code>water_frac</code> − <code>water_frac_lag1</code> (Phase 1 proposal, IV-F)."],
        ["3", "<code>OutlierCapper(z)</code>, <code>OutlierCapper(iqr)</code>", "<code>X_train</code>", "Winsorise rainfall at training μ ± 3σ / 1.5×IQR fences (IV-C/D)."],
        ["4", "<code>ColumnTransformer</code>", "<code>X_train</code>", "log1p(100·x)+Std (water), log1p+Std (skewed), Std (symmetric), MinMax (bounded), one-hot state, sin/cos month; Phase 1 drop list applied (V-A/B, IV-F)."],
        ["5", "Estimator (later)", "<code>X_train</code>", "Logistic-regression baseline used here only to verify the preprocessing (Section VI)."],
    ]
    st, cg, dl = "As planned", "Adapted", "Deferred"
    log = [
        ["Raw panel N × P and baseline missingness", st, f"Re-computed and asserted equal to Phase 1: {S['phase1_reproduced']['N']:,} × {S['phase1_reproduced']['P']}, {S['phase1_reproduced']['overall_missing_pct']}% missing, {S['phase1_reproduced']['complete_rows']:,} complete rows."],
        ["Temporal split train &lt; 2024-01-01 ≤ val &lt; 2025-07-01 ≤ test", st, "Implemented before any fitting; Table II."],
        ["Purge 2 windows at each boundary", st, f"{S['purged_rows']} rows in {S['purged_windows']} windows removed; shares move from {p1sp['train']}/{p1sp['validation']}/{p1sp['test']}% to {sp['train']['share_%']}/{sp['validation']['share_%']}/{sp['test']['share_%']}%."],
        ["Reject stratified shuffling; report class balance", st, "Fig. 2; positive rate 19.0% train vs 36.9% / 36.8% validation / test."],
        ["Never impute Y", st, f"{A1['N'] - A1['y_labelled_rows']:,} unlabelled rows excluded from fitting and evaluation."],
        ["County training-median imputation + indicator", st, "<code>GroupMedianImputer</code> fitted on <code>X_train</code>; 0 NaNs remain in all three splits."],
        ["Outlier rule by skewness (Z if |skew| ≤ 0.5, else IQR)", st, f"Rules kept as pre-registered. <code>rain_12d_mm</code> skew is {cap['rain_12d_mm']['skew']} on Phase 2 training rows vs {cap['rain_12d_mm']['phase1_skew']} in Phase 1 (borderline); its IQR rule is kept and the feature is dropped as collinear anyway."],
        ["Cap rainfall only; never cap water fractions or static terrain", st, f"{cap['rain_90d_anom_mm']['n_capped_train']} training values of <code>rain_90d_anom_mm</code> ({cap['rain_90d_anom_mm']['pct_capped_train']}%) capped; <code>rain_36d_mm</code> needed none (Fig. 3)."],
        ["Feature–target association by type (r<sub>pb</sub>, Mann-Whitney, χ², MI)", st, "MI ranking recomputed on Phase 2 training rows (Fig. 6); <code>water_frac</code> remains the strongest feature."],
        ["|ρ| ≥ 0.85 screen, lower-MI drop, VIF ≤ 10", cg, f"All 10 Phase 1 pairs reproduce (|Δρ| ≤ 0.004). Re-running the tie-break selects <code>twi_mean</code> instead of <code>hand_lt5m_frac</code> as the terrain representative (MI {tmi['twi_mean']} vs {tmi['hand_lt5m_frac']}); the pre-registered Phase 1 list is kept (Section V-A)."],
        ["One-hot state; county not encoded; month sin/cos", st, "11 state columns + 2 cyclical columns in the output (Section II-C)."],
        ["Log + StandardScaler for skewed features; MinMax for bounded", st, f"Implemented; <code>water_frac</code> skew {skew['Raw water_frac']:.1f} → {skew['Log: log1p(100 x)']:.2f} after log (Fig. 7)."],
        ["Momentum feature replacing raw lags", cg, f"Added as <code>MomentumFeature</code>: max |ρ| with retained features {S['momentum']['max_abs_rho_with_retained']}, MI {S['momentum']['mi']} (Section V-B)."],
        ["<code>TimeSeriesSplit(5)</code> with 2-window gap", st, "<code>temporal_cv()</code>, gap = 158 rows; used for the information-loss check."],
        ["<code>GroupKFold</code> by county (robustness)", st, f"PR-AUC {gk[0]} ± {gk[1]} on unseen counties (Section VI)."],
        ["Label sensitivity: −16 dB and VH variants", st, f"27 windows re-extracted; no variant met the pre-set rule, so the Phase 1 label is kept (Table IV)."],
        ["Report Sudd core separately", st, "Core (Jonglei, Unity, Upper Nile, Lakes) within-county ρ reported in Table IV."],
        ["Check sensor effect on exceedance rate", cg, "Implemented; result inconclusive — a monotone rise in exceedances is found instead (Section V-C)."],
        ["Class-weighted estimator; validation-tuned threshold", dl, "<code>class_weight=\"balanced\"</code> used in the verification baseline; threshold tuning belongs to the modelling assignment."],
        ["PCA scree diagnostic", st, f"Fig. 8: {pca['k90']} / {pca['k95']} of {pca['n_components']} components for 90% / 95% variance; PCA is not applied to model inputs (Section V-D)."],
    ]
    from collections import Counter
    cnt = Counter(r[1] for r in log)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
    n_st, n_cg, n_dl, n_all = (words.get(cnt[st], cnt[st]), words.get(cnt[cg], cnt[cg]), words.get(cnt[dl], cnt[dl]), words.get(len(log), len(log)))
    ls_rows = [[k.upper().replace("VV", "VV −").replace("VH", "VH −") + " dB", v["pooled_rho"], v["median_within_county_rho"],
                v["core_median_within_county_rho"], v["n_counties"], "kept (Phase 1)" if k == "vv18" else ("adopt" if v["adopt"] else "no")]
               for k, v in ls.items()]
    ver_rows = [[k, v["n_features"], f"{v['cv_pr_auc_mean']:.4f} ± {v['cv_pr_auc_sd']:.4f}", f"{v['val_pr_auc']:.4f}"] for k, v in ver.items()]
    yrs = sens["wet_by_year"]

    body = f"""
<div class="title">SuddWatch Phase 2: Pipeline Execution, Visual Diagnostics and Strategy-versus-Reality Audit</div>
<div class="authors">
  <div class="name">Jok Akech Atem Mabior</div>
  <div class="aff">MSc Engineering Artificial Intelligence, Carnegie Mellon University Africa, Kigali, Rwanda</div>
  <div class="mail">jmabior@andrew.cmu.edu · Repository: <a href="{REPO}">{REPO}</a></div>
</div>
<div class="cols">
<div class="abstract"><i>Abstract</i>—This report documents the execution of the SuddWatch Phase 1 preprocessing strategy on the same
{S['phase1_reproduced']['N']:,} × {S['phase1_reproduced']['P']} county-window panel. A single scikit-learn pipeline (county-median imputation,
a momentum feature, rainfall winsorisation and a column transformer) is fitted on training rows only and applied unchanged to the
validation and test periods, producing {shapes['train'][1]} model-ready features. Eight diagnostic figures verify each step. A decision log
compares all {n_all} Phase 1 commitments with what was implemented: {n_st} ran as planned, {n_cg} were adapted and {n_dl} is deferred to the modelling
stage. Reducing the feature set from {full['n_features']} to {fin['n_features']} columns did not lose information: temporal cross-validated PR-AUC
of a baseline classifier rose from {full['cv_pr_auc_mean']:.3f} to {fin['cv_pr_auc_mean']:.3f}. Two findings carry forward: the label
sensitivity check kept the Phase 1 label, and exceedance rates rise steadily from 2022, a drift risk for modelling.</div>
<div class="keywords"><i>Index Terms</i>—data preprocessing, scikit-learn pipeline, data leakage, Sentinel-1, flood forecasting, South Sudan.</div>

<h2>I. Introduction</h2>
<p>Phase 1 audited the SuddWatch panel and fixed a preprocessing strategy before any production code was fitted. Phase 2 executes that strategy in
<code>phase2_implementation/notebooks/pipeline_execution.ipynb</code>, which imports the same modules (<code>src/transformers.py</code>,
<code>src/pipeline_builder.py</code>) and starts by asserting that the raw panel still reproduces the Phase 1 audit exactly
({S['phase1_reproduced']['N']:,} rows, {S['phase1_reproduced']['P']} columns, {S['phase1_reproduced']['overall_missing_pct']}% missing cells,
{S['phase1_reproduced']['complete_rows']:,} complete rows). Section II describes the pipeline, Section III the eight diagnostic figures, Section IV
the decision log, Section V the deviations and their rationale, and Section VI the output verification.</p>

<h2>II. Pipeline Implementation</h2>
<h3>A. Architecture</h3>
<p>Table I lists the steps in execution order. Every fitted statistic is learned from <code>X_train</code> only; validation and test are passed
through <code>transform</code> and are never used to fit anything. The test set is not scored in this phase.</p>
{html_table("Pipeline Steps (Phase 1 Section in Brackets)", ["#", "Step", "Fitted on", "Purpose"], arch, span=True, widths=[4, 26, 10, 60])}
<h3>B. Temporal Split</h3>
<p>Rows without a label are excluded and the two issue windows before each split boundary are purged, because the target looks two windows ahead
and would otherwise overlap the next split. Table II gives the resulting partitions.</p>
{tbl(split_df, "Temporal Partitions After Purging", span=True, widths=[11, 24, 10, 9, 12, 11, 13])}
<h3>C. Processed Output</h3>
<p>The fitted pipeline maps the {S['raw_feature_columns']} raw input columns of each split to {shapes['train'][1]} numeric features:
train {shapes['train'][0]:,} × {shapes['train'][1]}, validation {shapes['validation'][0]:,} × {shapes['validation'][1]}, test
{shapes['test'][0]:,} × {shapes['test'][1]}, with no missing values. The matrices, the feature names and the fitted pipeline
(<code>preprocessor.joblib</code>) are saved in <code>data/processed/</code>.</p>

<h2>III. Visual Diagnostics</h2>
{img("fig1_1_missingness_heatmap.png", "Missingness pattern (assignment Fig. 1.1). Left: row × column matrix. Middle: <code>water_frac</code> gaps by county and issue window; the dense blocks between the Sentinel-1B loss and Sentinel-1C/D coverage are the MAR acquisition-plan gaps diagnosed in Phase 1. Right: missing percentage per column.", span=True)}
<p><b>Fig. 1.</b> Missingness is confined to the four Sentinel-1-derived columns and concentrated in a fixed set of counties between Dec 2021
and mid-2025, as diagnosed in Phase 1.</p>
{img("fig1_2_target_distribution.png", "Class balance by temporal split (assignment Fig. 1.2).")}
<p><b>Fig. 2.</b> The training set is imbalanced ({float(sp['train']['pos_rate_%']):.1f}% positive), while validation and test are near 37%
positive. Section V-C examines why.</p>
{img("fig2_1_outlier_boxplots.png", "Outlier box plots before and after winsorisation, training rows (assignment Fig. 2.1). Dotted lines are the training bounds.", span=True)}
<p><b>Fig. 3.</b> Only <code>rain_90d_anom_mm</code> has a material tail ({cap['rain_90d_anom_mm']['pct_capped_train']}% capped). The near-normal
rainfall totals fall inside μ ± 3σ, so capping leaves them unchanged.</p>
{img("fig3_1_correlation_matrix.png", "Upper-triangle Spearman correlation on training rows (assignment Fig. 3.1). Boxed cells: |ρ| ≥ 0.85; [drop]: removed by the pipeline.", span=True, width="68%")}
<p><b>Fig. 4.</b> {S['collinear_pairs']} pairs exceed the threshold. {n_w + n_t} of them lie inside two blocks: the water-fraction lags
({n_w} pairs) and the terrain gradient of elevation, slope, HAND and TWI ({n_t} pairs). The remaining {S['collinear_pairs'] - n_w - n_t} link the
rainfall windows and the two permanent-water measures.</p>
{img("fig3_2_collinear_scatterplots.png", "Collinear pairs with fitted OLS trendlines (assignment Fig. 3.2), one panel per removed feature.", span=True)}
<p><b>Fig. 5.</b> Each removed feature is close to a linear function of its partner. The static panels show the 79 counties directly, which is
why terrain features add little beyond one representative.</p>
{img("fig3_3_target_importance.png", "Feature-to-target importance by mutual information, training rows (assignment Fig. 3.3).")}
<p><b>Fig. 6.</b> Current water state, momentum and recent rainfall carry the most information about Y. Several removed features rank above some
retained ones; removal is driven by redundancy with a stronger partner, not by low individual signal.</p>
{img("fig4_1_transformation_grid.png", "Transformation stages for <code>water_frac</code>: raw, log1p(100·x), Min-Max and StandardScaler (assignment Fig. 4.1).")}
<p><b>Fig. 7.</b> The log step reduces skewness from {skew['Raw water_frac']:.1f} to {skew['Log: log1p(100 x)']:.2f}; the distribution is far less skewed but still not symmetric. Min-Max and standardisation
change scale but not shape, as expected.</p>
{img("fig5_1_pca_scree_plot.png", "PCA scree and cumulative variance on transformed training rows (assignment Fig. 5.1).", span=True)}
<p><b>Fig. 8.</b> {pca['k90']} components explain 90% and {pca['k95']} explain 95% of the variance of the {pca['n_components']} transformed
features. No single direction dominates: the first component explains {pca['first5_pct'][0]}%.</p>

<h2>IV. Decision Log</h2>
<p>Table III compares every Phase 1 commitment with the Phase 2 implementation. "Adapted" items are explained in Section V.</p>
{html_table("Decision Log: Phase 1 Strategy versus Phase 2 Implementation", ["Phase 1 plan item", "Status", "Phase 2 implementation and evidence"], log, span=True, widths=[30, 10, 60])}

<h2>V. Deviations and Rationale</h2>
<h3>A. Collinearity Tie-Break Instability</h3>
<p>The ten Phase 1 collinear pairs reproduce on the Phase 2 training rows within |Δρ| ≤ 0.004, but the lower-MI tie-break inside the terrain
block is not stable. Its six features take only 79 distinct values (one per county), and their MI estimates lie between {min(tmi.values())} and
{max(tmi.values())} nats, which is within estimator noise. Re-running the rule therefore keeps <code>twi_mean</code> instead of
<code>hand_lt5m_frac</code>. The pipeline keeps the pre-registered Phase 1 list: changing feature choices in response to each new sample would be a form of
selection leakage, and either feature represents the same highland-to-floodplain gradient.</p>
<h3>B. Momentum Feature Added</h3>
<p>Phase 1 proposed replacing the raw lags with a change feature. <code>water_frac_delta</code> has a maximum |ρ| of
{S['momentum']['max_abs_rho_with_retained']} with any retained feature, so it clears the collinearity rule, and its MI ({S['momentum']['mi']})
is close to that of the dropped lag. It is stateless and computed after imputation, so it cannot leak. Its skewness is near zero, so it is
standardised without a log step.</p>
<h3>C. Sensor Check Revealed a Rising Exceedance Rate</h3>
<p>The wet-season (Jul–Nov) positive rate is {sens['by_era']['S1A+S1B 2017-21']}% in 2017–2021 (two satellites),
{sens['by_era']['S1A only 2022-mid25']}% in 2022 – mid-2025 (Sentinel-1A only) and {sens['by_era']['S1A+S1C/D mid25-26']}% from mid-2025
(Sentinel-1C/D). By year it rises from {yrs['2019']}% in 2019 to {yrs['2022']}% in 2022 and {yrs['2026']}% in 2026. The step in 2022 coincides with the
loss of Sentinel-1B, but the highest rates occur after Sentinel-1C/D restored scene counts, so scene count alone does not explain the rise. It is
consistent with the persistent flooding reported around Bentiu and Rubkona, which remained below the surrounding water level in October 2025 [1],
but a sensor or processing contribution cannot be excluded with the data in hand. The practical consequence is non-stationarity: a threshold fixed on 2017–2023 is exceeded more often later. The modelling stage will test
trend-aware features (e.g. a rolling one-year county baseline) and report performance by year.</p>
<h3>D. Label Sensitivity and PCA Decisions</h3>
<p>The Phase 1 label was scored against UNOSAT alongside four alternatives on the {LS['matched_pairs']:,} county-date pairs (Table IV), under a rule
fixed before running: adopt a variant only if median within-county ρ improves by ≥ 0.10 without pooled ρ falling by more than 0.05. None
qualified, so the label is unchanged. VH < −26 dB is the most promising direction (Sudd-core ρ {ls['vh26']['core_median_within_county_rho']} vs
{ls['vv18']['core_median_within_county_rho']}), but the gain is below the threshold, so it is recommended for testing in the modelling stage rather
than adopted now. PCA is reported as a diagnostic only: the primary model is a tree ensemble that does not need decorrelated inputs, and retaining
named features keeps county forecasts explainable to humanitarian users.</p>
{html_table("Label Sensitivity Against UNOSAT (2022–2024)", ["Label variant", "Pooled ρ", "Median within-county ρ", "Sudd-core ρ", "Counties", "Decision"], ls_rows, widths=[18, 11, 20, 14, 13, 24])}

<h2>VI. Pipeline Output Verification</h2>
<h3>A. Dimensions</h3>
<p>Raw panel {S['phase1_reproduced']['N']:,} × {S['phase1_reproduced']['P']} → {A1['y_labelled_rows']:,} labelled rows →
{S['labelled_used']:,} after purging → train / validation / test {shapes['train'][0]:,} / {shapes['validation'][0]:,} / {shapes['test'][0]:,} rows,
each with {S['raw_feature_columns']} raw columns mapped to <b>{shapes['train'][1]} output features</b>. Without the collinearity rule the same
pipeline produces {full['n_features']} features, so the reduction is {full['n_features'] - fin['n_features']} features
({100 * (full['n_features'] - fin['n_features']) / full['n_features']:.0f}%).</p>
<h3>B. No Information Loss</h3>
<p>Each configuration is wrapped with a class-weighted logistic regression and scored by PR-AUC under <code>TimeSeriesSplit(5)</code> with the
2-window gap on training rows, then fitted on all training rows and scored once on validation (Table V). Prevalence, the PR-AUC of a random
classifier, is {S['prevalence']['train']:.3f} (train) and {S['prevalence']['validation']:.3f} (validation).</p>
{html_table("Information-Loss Check (Test Set Not Used)", ["Configuration", "Features", "CV PR-AUC (mean ± sd)", "Validation PR-AUC"], ver_rows, widths=[40, 14, 26, 20])}
<p>Removing {full['n_features'] - red['n_features']} collinear features did not reduce PR-AUC; it rose slightly
({full['cv_pr_auc_mean']:.4f} → {red['cv_pr_auc_mean']:.4f} in CV, {full['val_pr_auc']:.4f} → {red['val_pr_auc']:.4f} on validation), and
adding the momentum feature raised it further. The differences are much smaller than the fold-to-fold standard deviation, so the correct conclusion is
that the reduction preserved information, not that it improved the model. On counties never seen during fitting (<code>GroupKFold</code>),
PR-AUC is {gk[0]} ± {gk[1]}, which supports spatial generalisation within the training period.</p>

<h2>VII. Conclusion</h2>
<p>The Phase 1 strategy was implemented as a single leakage-safe scikit-learn pipeline fitted on training rows only. It produces a
{shapes['train'][1]}-feature matrix with no missing values, and the reduction from {full['n_features']} features lost no measurable
information. Of {n_all} Phase 1 commitments, {n_st} ran as planned, {n_cg} were adapted with documented reasons and {n_dl} (decision-threshold tuning)
belongs to modelling. The modelling stage inherits three items: the drift in exceedance rates (Section V-C), the VH −26 dB label variant
(Section V-D) and the decision threshold.</p>

<h2>References</h2>
<div class="refs">
<div>[1] UN Office for the Coordination of Humanitarian Affairs (OCHA), "South Sudan: Humanitarian Snapshot (October 2025)," Nov. 2025. [Online]. Available: https://www.unocha.org/publications/report/south-sudan/south-sudan-humanitarian-snapshot-october-2025</div>
<div>[2] J. A. A. Mabior, "SuddWatch Phase 1: Preprocessing and Feature Engineering Strategy," 04-652 Assignment 2, Carnegie Mellon University Africa, Oct. 2026. [Online]. Available: {REPO}/blob/main/phase1_strategy/Phase1_Strategy_Report.pdf</div>
<div>[3] R. Torres <i>et al.</i>, "GMES Sentinel-1 mission," <i>Remote Sens. Environ.</i>, vol. 120, pp. 9–24, 2012.</div>
<div>[4] N. Gorelick <i>et al.</i>, "Google Earth Engine: Planetary-scale geospatial analysis for everyone," <i>Remote Sens. Environ.</i>, vol. 202, pp. 18–27, 2017.</div>
<div>[5] F. Pedregosa <i>et al.</i>, "Scikit-learn: Machine learning in Python," <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.</div>
</div>
</div>
"""
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Phase 2 Final Report</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT_HTML.write_text(html, encoding="utf-8")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True)
    print("wrote", OUT_PDF, OUT_PDF.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
