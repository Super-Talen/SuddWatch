"""
SuddWatch - render phase1_strategy/Phase1_Strategy_Report.(html|pdf) from the audit outputs.

Every number in the report is read from phase1_strategy/audit/* (written by src.audit), so the
report cannot drift from the data. PDF is printed with headless Microsoft Edge.
Usage: .venv\\Scripts\\python.exe -m src.report_phase1
"""
import base64
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "phase1_strategy/audit"
FIG = ROOT / "figures"
OUT_HTML = ROOT / "phase1_strategy/Phase1_Strategy_Report.html"
OUT_PDF = ROOT / "phase1_strategy/Phase1_Strategy_Report.pdf"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def tbl(df, fmt=None, cls=""):
    fmt = fmt or {}
    h = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c.startswith("p (") and isinstance(v, float) and not np.isnan(v) and v < 1e-16:
                v = "&lt;1e-16"
            elif c in fmt and pd.notna(v) and not isinstance(v, str):
                v = fmt[c].format(v)
            elif isinstance(v, float):
                v = "" if np.isnan(v) else f"{v:,.3f}"
            cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def pfmt(p):
    return "&lt; 1e-16" if p < 1e-16 else f"= {p:.1e}"


def img(name, caption, width="100%"):
    p = FIG / name
    if not p.exists():
        return f"<p><em>[{name} not found]</em></p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def main():
    R = json.loads((AUD / "audit_summary.json").read_text())
    inv = pd.read_csv(AUD / "A_variable_inventory_missingness.csv")
    sk = pd.read_csv(AUD / "B_skewness_outliers_train.csv")
    asc = pd.read_csv(AUD / "C_feature_target_association_train.csv")
    pairs = pd.read_csv(AUD / "C_collinear_pairs_train.csv")
    vif = pd.read_csv(AUD / "C_vif_train.csv")
    drop = pd.read_csv(AUD / "C_collinearity_drop_list.csv")
    M = R["mechanism"]
    yt, yv, ys = R["y_train"], R["y_validation"], R["y_test"]
    U = R["unosat_validation"]
    wf = sk.set_index("feature")
    tot_lab = yt["n_labelled"] + yv["n_labelled"] + ys["n_labelled"]
    miss_cols = inv[inv.n_missing > 0][["column", "type", "n_missing", "pct_missing"]]
    era = M["water_frac_missing_pct_by_era"]
    era_txt = "; ".join(f"{k}: <b>{v}%</b>" for k, v in era.items())
    worst_states = sorted(M["missing_pct_by_state"].items(), key=lambda t: -t[1])[:3]
    kept_num = R["collinearity_kept_numeric"]
    thr = pd.read_csv(ROOT / "data/raw/county_thresholds_train_only.csv")
    pan = pd.read_csv(ROOT / "data/raw/suddwatch_panel_unprocessed.csv", usecols=["adm2_pcode", "split", "y_flood_2_4wk"])
    rate = pan[pan.split == "train"].dropna().groupby("adm2_pcode")["y_flood_2_4wk"].mean()
    binds = thr.set_index("adm2_pcode").eval("thr_floor > thr_p75")
    FL = {"n_floor": int(binds.sum()), "pos_floor": float(rate[binds].mean()), "pos_other": float(rate[~binds].mean())}

    inv_show = inv[["column", "type", "role", "dtype", "n_unique", "n_missing", "pct_missing"]].copy()
    sk_show = sk[["feature", "skew", "mean", "median", "rule", "iqr_low", "iqr_high", "pct_iqr_outliers", "pct_z3_outliers"]].copy()
    asc_show = asc[["feature", "type", "test", "abs_r_pb", "p_pb", "p_mannwhitney", "mutual_info"]].rename(
        columns={"abs_r_pb": "|r_pb| / Cramér's V", "p_pb": "p (r_pb / χ²)", "p_mannwhitney": "p (Mann-Whitney)"})

    treat = {
        "water_frac": "Flag only (IQR) — NOT capped: outliers are flood events (the signal). log1p(100·x) compresses the tail.",
        "water_frac_lag1": "Same as water_frac.", "water_frac_lag2": "Same as water_frac.",
        "obs_frac": "Not capped: coverage metadata bounded in [0,1]; low values are real acquisition gaps.",
    }
    for f in ["rain_12d_mm", "rain_36d_mm", "rain_90d_mm", "rain_90d_anom_mm"]:
        rule = sk.set_index("feature").loc[f, "rule"]
        treat[f] = "Cap at training μ ± 3σ (near-normal)." if rule.startswith("Z") else "Cap at training 1.5×IQR fences (skewed)."
    sk_show["treatment"] = sk_show["feature"].map(treat).fillna(
        "Static: real geography, not errors → not capped; log1p if |skew| > 1.")

    css = """
    @page { size: A4; margin: 16mm 14mm 16mm 14mm; }
    body { font-family: 'Times New Roman', Georgia, serif; font-size: 10.5pt; line-height: 1.38; color: #111; }
    h1 { font-size: 17pt; text-align: center; margin: 0 0 2px 0; }
    .sub { text-align: center; font-size: 10pt; color: #333; margin-bottom: 10px; }
    h2 { font-size: 13pt; border-bottom: 1.5px solid #222; padding-bottom: 2px; margin-top: 18px; page-break-after: avoid; }
    h3 { font-size: 11pt; margin: 12px 0 4px 0; page-break-after: avoid; }
    table { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 6px 0 10px 0; page-break-inside: auto; }
    th, td { border: 1px solid #999; padding: 2px 4px; text-align: left; vertical-align: top; }
    th { background: #e9edf2; }
    tr { page-break-inside: avoid; }
    figure { margin: 8px 0; text-align: center; page-break-inside: avoid; }
    figcaption { font-size: 9pt; color: #333; }
    .box { border: 1px solid #888; background: #f6f7f9; padding: 6px 9px; margin: 8px 0; }
    .kv td:first-child { width: 28%; font-weight: bold; background: #f3f5f8; }
    code { font-family: Consolas, monospace; font-size: 9pt; }
    """

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Phase 1 Strategy Report</title>
<style>{css}</style></head><body>
<h1>SuddWatch — Phase 1: Preprocessing &amp; Feature Engineering Strategy</h1>
<div class="sub">04-652 AI System Design · Assignment 2 · Carnegie Mellon University Africa<br>
Jok Akech Atem Mabior · Binary classification · Repository: <a href="https://github.com/Super-Talen/SuddWatch">https://github.com/Super-Talen/SuddWatch</a></div>

<h2>0. Problem, task and data</h2>
<table class="kv">
<tr><td>Objective</td><td>Forecast, for each of South Sudan's 79 counties, whether surface-water extent will exceed that county's
normal high-water level <b>13–24 days after the forecast is issued</b>, so humanitarian planners can pre-position aid before roads and
river routes close (SDG 13 → SDG 2, SDG 1).</td></tr>
<tr><td>Task type</td><td><b>Binary classification</b> (supervised, short-horizon spatio-temporal panel).</td></tr>
<tr><td>Unit of observation</td><td>One county × one 12-day Sentinel-1 window (12 days = Sentinel-1A repeat cycle). The forecast is issued at the end of window <i>w</i>.</td></tr>
<tr><td>Target Y (<code>y_flood_2_4wk</code>)</td><td>1 if the county's Sentinel-1 surface-water fraction in window <i>w</i>+2 exceeds the county's own
<b>75th percentile computed on training windows only</b> (floor: 1 km² of water). A county-specific threshold is required because baseline
wetness differs by orders of magnitude between the permanent Sudd swamps and the Equatoria highlands; a single global cut-off would label
Sudd counties "flooded" permanently and Equatoria counties never.</td></tr>
<tr><td>Sources (all open)</td><td>Copernicus Sentinel-1 GRD (labels + autoregressive features), CHIRPS daily rainfall, SRTM DEM, MERIT Hydro (HAND, upstream area),
JRC Global Surface Water — all reduced to county level in Google Earth Engine; OCHA COD-AB county boundaries (HDX); UNOSAT VIIRS water extents 2022–2024 (HDX) used only as independent label validation.</td></tr>
<tr><td>Raw panel</td><td><b>N = {R['N']:,} rows × P = {R['P_total_columns']} columns</b> ({R['P_candidate_features']} candidate features + 2 ID columns, 2 datetime columns, 1 split tag, 1 target).
Issue dates {yt['first_issue']} → {ys['last_issue']}.</td></tr>
</table>
<div class="box"><b>Label construction (decision log).</b> Sentinel-1 provides radar backscatter, not flood maps. Surface water = per-pixel median VV backscatter in the 12-day window
&lt; −18 dB, restricted to terrain within 20 m of the nearest drainage (MERIT HAND), aggregated at 200 m. The −18 dB threshold was
<b>calibrated, not assumed</b>: among −14…−20 dB it gave the best county-level agreement with UNOSAT's independent VIIRS product
(Spearman 0.70 on 2023-09-26). <br><b>Validation against UNOSAT (2022–2024, {U.get('matched_pairs'):,} county-date pairs) — reported honestly.</b> Across counties agreement is good
(pooled Spearman ρ = <b>{U.get('pooled_spearman')}</b>) and the national seasonal cycle agrees moderately (ρ = {U.get('national_per_date_spearman')} over {U.get('n_dates')} dates).
<i>Within</i> counties over time — the comparison that matters for a county-relative target — agreement is uneven: median ρ by state
{", ".join(f"{k} {v}" for k, v in sorted(U.get('per_state_median_temporal_spearman', {}).items(), key=lambda t: -t[1]))}; overall median {U.get('per_county_temporal_spearman_median')}.
It is strongest in Jonglei, the core Sudd state that motivates the project, and weak or negative in drier, more vegetated western/northern counties.
Causes: (i) the two products measure different things — SAR detects open water but misses water under dense papyrus/reed canopies, while VIIRS
(375 m) also captures partly flooded vegetation; (ii) 2022–2024 is the single-satellite Sentinel-1 period with the sparsest coverage.
<b>Phase 2 actions</b>: add VH-based flooded-vegetation detection, run a −16 dB sensitivity variant, and report all metrics both nationally and for the Sudd core
(Jonglei, Unity, Upper Nile, Lakes).</div>

<h2>A. Dataset &amp; Problem Audit</h2>
<h3>A1. Initial shape</h3>
<p>The raw (un-preprocessed) panel <code>data/raw/suddwatch_panel_unprocessed.csv</code> has <b>N = {R['N']:,} observations</b> (79 counties ×
{R['N'] // 79} issue windows) and <b>P = {R['P_total_columns']} columns</b>, of which <b>{R['P_candidate_features']} are candidate predictive features</b>.</p>

<h3>A2. Variable inventory</h3>
<p>Counts by type: {", ".join(f"{k}: {v}" for k, v in R['type_counts'].items())}. <b>No ordinal variables exist</b> in this data: states have no rank, and month is
cyclical (December is adjacent to January), so treating it as ordinal would impose a false 12-to-1 distance.</p>
{tbl(inv_show, {"pct_missing": "{:.2f}"})}

<h3>A3. Baseline missingness (before any processing)</h3>
<p>Overall, <b>{R['overall_missing_pct']}%</b> of all {R['N'] * R['P_total_columns']:,} cells are missing. <b>{R['complete_rows']:,}</b> rows are complete and
<b>{R['rows_with_missing']:,}</b> rows ({100 * R['rows_with_missing'] / R['N']:.1f}%) contain at least one missing value. Missingness is confined to the Sentinel-1-derived columns:</p>
{tbl(miss_cols, {"pct_missing": "{:.2f}"})}
{img("fig1_1_missingness_heatmap.png", "Fig 1.1 — Missingness pattern: water_frac missing by county × issue window (left) and per-column missing % (right).")}

<h3>A4. Target label specification</h3>
<p>Binary classification. Class balance of labelled rows per temporal split:</p>
{tbl(pd.DataFrame([{"split": s, "issue dates": f"{d['first_issue']} → {d['last_issue']}", "rows": d['n_rows'], "labelled": d['n_labelled'],
                    "Y=1": d['pos'], "Y=0": d['neg'], "positive rate": f"{100 * d['pos_rate']:.1f}%", "imbalance (neg:pos)": f"{d['neg_to_pos_ratio']}:1"}
                   for s, d in [("train", yt), ("validation", yv), ("test", ys)]]))}
{img("fig1_2_target_distribution.png", "Fig 1.2 — Class balance by split.", "70%")}
<p><b>Imbalance and mitigation.</b> The training set is moderately imbalanced ({yt['neg_to_pos_ratio']}:1, {100 * yt['pos_rate']:.1f}% positive). The rate is below the
nominal 25% because of the 1 km² floor: in {FL['n_floor']} of 79 dry counties the training P75 is below 1 km² of water, the floor binds, and those counties are positive in only
{100 * FL['pos_floor']:.1f}% of training windows, versus {100 * FL['pos_other']:.1f}% in the other {79 - FL['n_floor']} counties (≈25% by construction). This is intended: in very
dry counties, crossing a sub-1 km² level would reflect radar noise rather than flooding. Positive rates are higher in validation (2024 – mid-2025) and test (mid-2025 – 2026) because
the threshold is fixed from 2017–2023 and later seasons were wetter — the 2025 season affected over one million people (proposal, ref. [1]). A sensor effect cannot be fully excluded
for the test period (Sentinel-1C/D add scenes from mid-2025); Phase 2 will check it by comparing exceedance rates with and without <code>n_scenes</code> stratification.
Mitigation plan, in order: (1) <code>class_weight="balanced"</code> in the estimator (no synthetic rows, no information loss); (2) choose the decision threshold on the
<i>validation</i> precision–recall curve to meet the proposal's &lt;15% false-alarm target; (3) evaluate with PR-AUC, F1 and recall rather than accuracy.
If resampling (SMOTE) is tried, it runs <i>inside</i> training folds only, after the temporal split, never on validation/test. Random under-sampling is rejected
because it discards whole flood seasons from a temporally scarce record.</p>

<h2>B. Cleaning &amp; Imputation Strategy</h2>
<h3>B1. Missingness mechanism classification</h3>
<table><thead><tr><th>Column(s)</th><th>Mechanism</th><th>Evidence from this dataset</th><th>Treatment</th></tr></thead><tbody>
<tr><td><code>water_frac</code> (and inherited <code>obs</code>-based gaps)</td><td><b>MAR</b></td>
<td>Missing when &lt;50% of the county was imaged in the window. Missing share by satellite era: {era_txt} — χ² test of era × missingness
p {pfmt(M['chi2_era_p'])}; by state p {pfmt(M['chi2_state_p'])} (highest: {", ".join(f"{k} {v}%" for k, v in worst_states)}).
The gap is explained by <i>observed</i> variables (date → Sentinel-1B failure on 23 Dec 2021 and Sentinel-1C/D arrival in 2025; location → orbit tracks
excluded from the single-satellite acquisition plan). <b>Not MCAR</b>: it is strongly systematic. <b>Not MNAR</b>: ESA plans acquisitions in advance, independently
of conditions on the ground. Empirically, in 2022–2024 the raw 36-day rainfall before missing windows is lower (median {M['rain36_missing_vs_observed_median_2022_24'][0]:.1f} vs
{M['rain36_missing_vs_observed_median_2022_24'][1]:.1f} mm, Mann-Whitney p = {M['mannwhitney_rain_p_2022_24']:.3f}) — but only because the un-imaged counties are drier places:
in a logistic regression of missingness on state + month + standardised rainfall, the rainfall coefficient is {M['logit_rain_coef_adj_state_month']:.3f} (p = {M['logit_rain_p_adj_state_month']:.2f}).
Once observed location and season are controlled, wetness carries no information about missingness. (MNAR can never be fully excluded from observed data
alone; here it is ruled out by the acquisition mechanism and this adjusted test.)</td>
<td>County-level <b>training-median</b> imputation + binary <code>_was_missing</code> indicator (see justification below).</td></tr>
<tr><td><code>water_frac_lag1</code>, <code>water_frac_lag2</code></td><td><b>MAR</b> (+ structural)</td><td>Shifted copies of <code>water_frac</code>, so they inherit its MAR gaps; the first 1–2 windows of 2017 are missing by construction (no earlier window exists).</td>
<td>Same imputer + indicator; structural leading gaps fall in early training only.</td></tr>
<tr><td><code>y_flood_2_4wk</code></td><td><b>MAR</b></td><td>Undefined when window <i>w</i>+2 was not imaged (same acquisition-plan mechanism).</td>
<td><b>Never imputed.</b> Rows without Y are excluded from fitting and evaluation (imputing a label would fabricate ground truth).</td></tr>
<tr><td>All rainfall, terrain, calendar, ID columns</td><td>—</td><td>0 missing (gridded products with complete coverage; static county attributes).</td><td>None needed.</td></tr>
</tbody></table>
<p><b>Why median, not mean.</b> <code>water_frac</code> is extremely right-skewed (training skewness = <b>{wf.loc['water_frac', 'skew']:.1f}</b>;
mean = {wf.loc['water_frac', 'mean']:.4f} vs median = {wf.loc['water_frac', 'median']:.4f}). A mean fill would inject a value inflated by rare floods into
ordinary dry windows. <b>Why per county:</b> baseline wetness differs by orders of magnitude between counties; a global median would erase that structure.
<b>Why not KNN:</b> the gaps are spatially clustered — whole orbit tracks are missing together — so a county's nearest neighbours are usually missing in the
same window; KNN would borrow from distant, hydrologically unrelated counties. <b>Why not last-observation-carried-forward:</b> 9 counties were not imaged
for up to ~3 years (2022–2025), so carried-forward values would be years stale. <b>Mode</b> is inapplicable to a continuous feature. The indicator columns preserve
the information that a value was imputed.</p>

<h3>B2. Outlier identification rules (training rows)</h3>
<p>Rule chosen by distribution shape: |skew| ≤ 0.5 → Z-score rule (|z| &gt; 3); |skew| &gt; 0.5 → 1.5×IQR rule. Bounds below are computed on training rows only
and are re-learned inside the pipeline (<code>OutlierCapper.fit</code>) on <code>X_train</code>.</p>
{tbl(sk_show, {"skew": "{:.2f}", "mean": "{:.4g}", "median": "{:.4g}", "iqr_low": "{:.4g}", "iqr_high": "{:.4g}", "pct_iqr_outliers": "{:.1f}%", "pct_z3_outliers": "{:.1f}%"})}

<h3>B3. Outlier treatment: capping, not deletion</h3>
<p>No row is ever deleted for an outlier. (a) <b>Preserves N</b>: deleting every row with any flagged value would remove a large share of the panel and,
worse, preferentially remove flood seasons — the minority class. (b) <b>Prevents information loss</b>: a row with one extreme rainfall value still carries valid
terrain, calendar and water-history signal. (c) <b>Domain rule</b>: water-fraction "outliers" are the floods SuddWatch exists to predict, so they are flagged but
<b>not capped</b>; static terrain extremes are real counties (e.g. the Equatoria highlands), so they are log-transformed, not capped. Only dynamic rainfall features are
winsorised, at the training bounds above.</p>

<h3>B4a. Feature–target association matrix (training rows)</h3>
<p>Matching rule: continuous/discrete feature × binary Y → <b>point-biserial r</b> (linear association) <b>plus Mann-Whitney U</b> (rank-based, robust to the
heavy skew found in B2) <b>plus mutual information</b> (captures non-monotonic effects such as seasonality); nominal state × binary Y → <b>χ² test with Cramér's V</b>
and MI. Pearson r is not used between features and Y (Y is binary), and Spearman ρ is used between features (B4b) because most are non-normal.
<b>Caveat:</b> static terrain features take only 79 distinct values (one per county) repeated over time, so their row-level p-values overstate the evidence;
mutual information, not p, is therefore the ranking and tie-break criterion.</p>
{tbl(asc_show, {"|r_pb| / Cramér's V": "{:.3f}", "p (r_pb / χ²)": "{:.1e}", "p (Mann-Whitney)": "{:.1e}", "mutual_info": "{:.4f}"})}

<h3>B4b. Multicollinearity protocol and preliminary result</h3>
<p>Protocol: (1) pairwise screen with |Spearman ρ| ≥ 0.85 on training rows; (2) decision rule — walk pairs from the strongest |ρ| down and, if both are still
present, <b>drop the feature with the lower mutual information with Y</b> (keeps the more predictive member); (3) iterate VIF on the survivors, dropping the highest
VIF &gt; 10 until all VIF ≤ 10. Pairs found:</p>
{tbl(pairs, {"spearman": "{:.3f}"})}
<p>Resulting drop list (applied in Phase 2 via <code>DROP_COLLINEAR</code> in <code>src/pipeline_builder.py</code>):</p>
{tbl(drop)}
{tbl(vif, {"VIF_before": "{:.2f}", "VIF_after": "{:.2f}"})}
<p>Retained numeric features: {", ".join(f"<code>{c}</code>" for c in kept_num)}. The terrain block (elevation, slope, HAND, TWI) is one physical gradient
(highland → floodplain), so one representative suffices. The autoregressive signal is retained through <code>water_frac</code> (the latest observed water state, two
windows before the target); Phase 2 will test a less collinear momentum feature (<code>water_frac − water_frac_lag1</code>) in place of raw lags.</p>

<h2>C. Encoding, Scaling &amp; Transformation Strategy</h2>
<h3>C1. Encoding matched to cardinality</h3>
<table><thead><tr><th>Variable</th><th>Type / cardinality</th><th>Encoding</th><th>Justification</th></tr></thead><tbody>
<tr><td><code>adm1_name</code> (state)</td><td>Nominal, 11 levels</td><td>One-hot (<code>handle_unknown="ignore"</code>)</td><td>Low cardinality, no order; 11 binary columns is cheap and keeps states distinct.</td></tr>
<tr><td><code>adm2_pcode</code> (county)</td><td>Nominal, 79 levels</td><td><b>Not encoded</b> — join key, imputation group and CV group only</td><td>79 one-hot columns on ~{R['N'] // 79} rows per county would let a model memorise county identity; county behaviour enters through terrain and water history instead.</td></tr>
<tr><td><code>month</code></td><td>Discrete, cyclical 1–12</td><td>sin/cos pair</td><td>Preserves Dec–Jan adjacency; an ordinal 1–12 code would place them 11 units apart.</td></tr>
<tr><td>Ordinal variables</td><td>None present</td><td>OrdinalEncoder not applicable</td><td>No variable carries an inherent rank; forcing one would invent order.</td></tr>
</tbody></table>
<h3>C1a. Numerical scaling matched to distributions and algorithms</h3>
<table><thead><tr><th>Feature group</th><th>Distribution (B2)</th><th>Transform → scaler</th><th>Reason</th></tr></thead><tbody>
<tr><td>Water fractions</td><td>Extreme right skew (≈{wf.loc['water_frac', 'skew']:.0f})</td><td><code>log1p(100·x)</code> → StandardScaler</td><td>Log compresses the flood tail; MinMax alone would squash 99% of values near 0.</td></tr>
<tr><td>County area, elevation mean/std, slope, HAND, water occurrence</td><td>|skew| &gt; 1</td><td><code>log1p</code> → StandardScaler</td><td>Multiplicative scales; log makes them closer to symmetric before standardisation.</td></tr>
<tr><td>Rainfall totals and anomaly, TWI, scene count</td><td>|skew| ≤ 1</td><td>StandardScaler (after capping)</td><td>Near-symmetric; mean 0 / sd 1 required by the logistic-regression baseline (L2 penalty) and by PCA (Fig 5.1).</td></tr>
<tr><td><code>obs_frac</code>, <code>hand_lt5m_frac</code></td><td>Bounded [0, 1]</td><td>MinMaxScaler</td><td>Known physical bounds; preserves exact 0 and 1.</td></tr>
</tbody></table>
<p>The primary model (gradient-boosted trees) is invariant to monotone scaling, so scaling is not needed for it; it is still applied for the logistic-regression
baseline, PCA and any distance-based comparison. <code>build_preprocessor(scale_tree_inputs=False)</code> skips it for trees.</p>

<h3>C2. Train / validation / test partitioning and leakage prevention</h3>
<p><b>(a) Ratios.</b> Temporal blocks of whole flood seasons: train = issue dates &lt; 2024-01-01, validation = 2024-01-01 – 2025-06-30, test = 2025-07-01 – 2026-08-31.
Of {tot_lab:,} labelled rows this gives <b>{R['split_pct_of_labelled'].get('train')}% / {R['split_pct_of_labelled'].get('validation')}% / {R['split_pct_of_labelled'].get('test')}%</b>.
Seven training seasons (2017–2023) cover wet and dry years; the held-out test year contains the full 2025 flood season named in the proposal. With N ≈ {R['N'] // 1000}k
an 80/20 random split would be statistically ample, but randomness is the wrong criterion for a forecasting task (see d).</p>
<p><b>(b) Stratification.</b> <code>StratifiedShuffleSplit</code> is <b>rejected</b>: it mixes future windows into training (look-ahead leakage) and, because
neighbouring windows are autocorrelated, inflates scores. Class proportions are instead <i>reported</i> per split (A4) and per CV fold, and imbalance is handled by
class weights and validation-tuned thresholds.</p>
<p><b>(c) Grouping.</b> Every county appears in every split <i>by design</i>: the deployed system forecasts the same 79 counties forward in time, so the
operational question is temporal generalisation. A supplementary <code>GroupKFold</code> by county (within the training period) will measure spatial
generalisation to unseen counties as a robustness check.</p>
<p><b>(d) Temporal ordering.</b> All splits move forward in time. Inside training, hyper-parameters are tuned with <code>TimeSeriesSplit(n_splits=5,
gap=2 windows × 79 rows)</code>: the 2-window gap purges rows whose target window (<i>w</i>+2) would overlap the next validation fold. The last two training windows
are likewise purged before validation.</p>
<p><b>(e) Leakage-prevention architecture.</b></p>
<table><thead><tr><th>Leakage path</th><th>Control</th></tr></thead><tbody>
<tr><td>Target threshold sees the future</td><td>County P75 computed on training windows only (<code>data/raw/county_thresholds_train_only.csv</code>).</td></tr>
<tr><td>Rainfall from after the issue date</td><td>Sums end on the issue date; anomaly climatology from 2001–2016, strictly before the study period.</td></tr>
<tr><td>Future satellite data</td><td>Only windows <i>w</i>, <i>w</i>−1, <i>w</i>−2 enter X; window <i>w</i>+1 and the target window <i>w</i>+2 never do.</td></tr>
<tr><td>Imputation / capping / scaling / encoding statistics</td><td>All live in one scikit-learn <code>Pipeline</code> (<code>src/pipeline_builder.py</code>): <code>GroupMedianImputer → OutlierCapper(Z) → OutlierCapper(IQR) → ColumnTransformer → estimator</code>.
<code>fit</code> is called on <code>X_train</code> only and re-fitted inside every CV fold; validation/test are only <code>transform</code>ed.</td></tr>
<tr><td>Feature selection</td><td>Collinearity and MI statistics computed on training rows only.</td></tr>
<tr><td>Same-time proxies of Y</td><td>Excluded: no same-window flood/population layers (e.g. UNOSAT "Pop exposed"); UNOSAT is used only for label validation.</td></tr>
</tbody></table>

<h2>Reproducibility</h2>
<p>All code is in <code>src/</code>: <code>gee_extract.py</code> (Earth Engine extraction, cached and resumable) → <code>build_panel.py</code> (panel and target) →
<code>audit.py</code> (every number and Figs 1.1–1.2) → <code>report_phase1.py</code> (this report). Environment: <code>requirements.txt</code>. Audit tables: <code>phase1_strategy/audit/</code>.</p>
</body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True)
    print("wrote", OUT_PDF, OUT_PDF.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
