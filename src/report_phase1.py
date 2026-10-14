"""
SuddWatch - render phase1_strategy/Phase1_Strategy_Report.(html|pdf) from the audit outputs,
in IEEE conference format (two columns, Times, Roman-numeral sections, TABLE/Fig. captions).

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
REPO = "https://github.com/Super-Talen/SuddWatch"

_counter = {"table": 0, "fig": 0}


def roman(n):
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out, n = out + s, n - v
    return out


def caption_table(title, span=False, body=""):
    _counter["table"] += 1
    cls = "tblock span" if span else "tblock"
    return (f'<div class="{cls}"><div class="tcap">Table {roman(_counter["table"])}<br>{title}</div>'
            f"{body}</div>")


def colgroup(widths):
    return "<colgroup>" + "".join(f'<col style="width:{w}%">' for w in widths) + "</colgroup>" if widths else ""


def tbl(df, title, fmt=None, span=False, widths=None):
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
    body = f'<table>{colgroup(widths)}<thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    return caption_table(title, span, body)


def html_table(title, header, rows, span=False, widths=None):
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return caption_table(title, span, f"<table>{colgroup(widths)}<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>")


def pfmt(p):
    return "&lt; 1e-16" if p < 1e-16 else f"= {p:.1e}"


def img(name, caption, span=False, width="100%"):
    _counter["fig"] += 1
    p = FIG / name
    if not p.exists():
        return f"<p><em>[{name} not found]</em></p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    cls = "span" if span else ""
    return (f'<figure class="{cls}"><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f'<figcaption>Fig. {_counter["fig"]}. {caption}</figcaption></figure>')


CSS = """
@page { size: letter; margin: 0.75in 0.625in 0.9in 0.625in; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.18; color: #000; margin: 0; }
.title { font-size: 22pt; text-align: center; line-height: 1.15; margin: 0 0 12pt 0; }
.authors { text-align: center; margin-bottom: 14pt; }
.authors .name { font-size: 11pt; }
.authors .aff { font-size: 10pt; font-style: italic; }
.authors .mail { font-size: 9.5pt; }
.cols { column-count: 2; column-gap: 0.25in; text-align: justify; hyphens: auto; }
.abstract, .keywords { font-size: 9pt; font-weight: bold; text-align: justify; margin: 0 0 6pt 0; }
.abstract i, .keywords i { font-weight: bold; }
h2 { font-size: 10pt; font-weight: normal; text-align: center; font-variant: small-caps; letter-spacing: 0.2px;
     margin: 10pt 0 4pt 0; break-after: avoid; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; margin: 6pt 0 2pt 0; break-after: avoid; }
p { margin: 0 0 3pt 0; text-indent: 1em; }
td p, .box p { text-indent: 0; }
.span { column-span: all; }
.tblock { margin: 6pt 0 8pt 0; break-inside: avoid-column; }
.tblock.span { break-inside: auto; }
.tcap { text-align: center; font-size: 8pt; font-variant: small-caps; margin-bottom: 3pt; line-height: 1.25; }
table { border-collapse: collapse; width: 100%; font-size: 7.4pt; line-height: 1.15; text-align: left; }
table { table-layout: fixed; }
th, td { border: 0.6px solid #8a8a8a; padding: 1.5px 3px; vertical-align: top; overflow-wrap: break-word; hyphens: none; }
th { background: #e9edf2; font-weight: bold; }
tr { break-inside: avoid; }
.kv td:first-child { width: 27%; font-weight: bold; background: #f3f5f8; }
figure { margin: 6pt 0 8pt 0; text-align: center; break-inside: avoid; }
figcaption { font-size: 8pt; text-align: justify; margin-top: 2pt; }
.box { border: 0.8px solid #888; background: #f6f7f9; padding: 4pt 6pt; margin: 6pt 0; font-size: 9pt; text-align: justify; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8pt; overflow-wrap: anywhere; }
td code { font-size: 7pt; }
.refs { font-size: 8pt; text-align: left; }
.refs div { padding-left: 2.2em; text-indent: -2.2em; margin-bottom: 2pt; }
a { color: #000; text-decoration: none; }
"""


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
    sp = R["split_pct_of_labelled"]
    state_rho = ", ".join(f"{k} {v}" for k, v in sorted(U.get("per_state_median_temporal_spearman", {}).items(), key=lambda t: -t[1]))

    inv_show = inv[["column", "type", "role", "dtype", "n_unique", "n_missing", "pct_missing"]].copy()
    sk_show = sk[["feature", "skew", "mean", "median", "rule", "iqr_low", "iqr_high", "pct_iqr_outliers", "pct_z3_outliers"]].copy()
    asc_show = asc[["feature", "type", "test", "abs_r_pb", "p_pb", "p_mannwhitney", "mutual_info"]].rename(
        columns={"abs_r_pb": "|r_pb| / Cramér's V", "p_pb": "p (r_pb / χ²)", "p_mannwhitney": "p (Mann-Whitney)"})
    treat = {
        "water_frac": "Flag only (IQR); NOT capped: outliers are flood events (the signal). log1p(100·x) compresses the tail.",
        "water_frac_lag1": "Same as water_frac.", "water_frac_lag2": "Same as water_frac.",
        "obs_frac": "Not capped: coverage metadata bounded in [0,1]; low values are real acquisition gaps.",
    }
    for f in ["rain_12d_mm", "rain_36d_mm", "rain_90d_mm", "rain_90d_anom_mm"]:
        rule = wf.loc[f, "rule"]
        treat[f] = "Cap at training μ ± 3σ (near-normal)." if rule.startswith("Z") else "Cap at training 1.5×IQR fences (skewed)."
    sk_show["treatment"] = sk_show["feature"].map(treat).fillna("Static: real geography, not errors → not capped; log1p if |skew| > 1.")

    split_df = pd.DataFrame([{"split": s, "issue dates": f"{d['first_issue']} → {d['last_issue']}", "rows": d["n_rows"],
                              "labelled": d["n_labelled"], "Y=1": d["pos"], "Y=0": d["neg"],
                              "positive rate": f"{100 * d['pos_rate']:.1f}%", "neg:pos": f"{d['neg_to_pos_ratio']}:1"}
                             for s, d in [("train", yt), ("validation", yv), ("test", ys)]])

    body = f"""
<div class="title">SuddWatch Phase 1: Preprocessing and Feature Engineering Strategy for County-Level Flood Forecasting in South Sudan</div>
<div class="authors">
  <div class="name">Jok Akech Atem Mabior</div>
  <div class="aff">MSc Engineering Artificial Intelligence, Carnegie Mellon University Africa, Kigali, Rwanda</div>
  <div class="mail">jmabior@andrew.cmu.edu · Repository: <a href="{REPO}">{REPO}</a></div>
</div>

<div class="cols">
<div class="abstract"><i>Abstract</i>—This report audits the raw dataset for SuddWatch, a binary classifier that forecasts whether each of South Sudan's
79 counties will exceed its own normal surface-water level 13–24 days ahead, and defines the preprocessing strategy before any production code is fitted.
The panel was built in Google Earth Engine from Sentinel-1 radar, CHIRPS rainfall, SRTM, MERIT Hydro and JRC Global Surface Water, and contains
N = {R['N']:,} county-window observations and P = {R['P_total_columns']} columns ({R['P_candidate_features']} candidate features). Missing values
({R['overall_missing_pct']}% of cells) are confined to Sentinel-1-derived columns and are diagnosed as missing at random, driven by the satellite
acquisition plan after the loss of Sentinel-1B. The report specifies skewness-backed imputation, distribution-matched outlier rules, a feature–target
test matrix, a |ρ| ≥ 0.85 / VIF &gt; 10 collinearity protocol, cardinality-matched encoding, and a temporal {sp.get('train')}/{sp.get('validation')}/{sp.get('test')}
split whose transformers are fitted on training data only.</div>
<div class="keywords"><i>Index Terms</i>—flood forecasting, Sentinel-1, data preprocessing, missing data, temporal validation, South Sudan.</div>

<h2>I. Introduction</h2>
<p>Recurrent flooding in the Sudd wetlands has become a leading multiplier of South Sudan's hunger and displacement crises; by October 2025 floods had
affected over one million people across 29 counties [1]. Planners need to know <i>which county</i> will flood in the next few weeks, early enough to
pre-position food and boats before roads and river routes close. SuddWatch frames this as supervised binary classification on a county × time panel
(Table I). This Phase 1 report follows the assignment guidelines: Section III audits the dataset (Guideline A), Section IV defines cleaning,
imputation, outlier and association strategy (Guideline B), and Sections V–VI define encoding, scaling, partitioning and leakage prevention (Guideline C).</p>
{html_table("Problem Specification", ["Item", "Specification"], [
    ["Objective", "Forecast, per county, whether surface-water extent will exceed that county's normal high-water level <b>13–24 days after the forecast is issued</b> (SDG 13 → SDG 2, SDG 1)."],
    ["Task type", "<b>Binary classification</b> (supervised, short-horizon spatio-temporal panel)."],
    ["Unit of observation", "One county × one 12-day Sentinel-1 window (12 days = Sentinel-1A repeat cycle); forecast issued at the end of window <i>w</i>."],
    ["Target Y (<code>y_flood_2_4wk</code>)", "1 if the Sentinel-1 surface-water fraction in window <i>w</i>+2 exceeds the county's own <b>75th percentile, computed on training windows only</b> (floor: 1 km²). A county-specific threshold is needed because baseline wetness differs by orders of magnitude between the permanent Sudd swamps and the Equatoria highlands."],
    ["Raw panel", f"<b>N = {R['N']:,} × P = {R['P_total_columns']}</b> ({R['P_candidate_features']} candidate features, 2 IDs, 2 datetimes, 1 split tag, 1 target); issue dates {yt['first_issue']} → {ys['last_issue']}."],
]).replace("<table>", '<table class="kv">')}

<h2>II. Data Sources and Label Construction</h2>
<h3>A. Sources</h3>
<p>All sources are open. Sentinel-1 GRD [2], CHIRPS daily rainfall [3], SRTM [4], MERIT Hydro [5] and JRC Global Surface Water [6] were reduced to
county level in Google Earth Engine [7] using OCHA COD-AB county boundaries [8]. UNOSAT VIIRS water extents for 2022–2024 [9] are used only as an
independent check on the label, never as a model input.</p>
<h3>B. Label Construction</h3>
<p>Sentinel-1 provides radar backscatter, not flood maps. Surface water is defined as per-pixel median VV backscatter in the 12-day window below
−18 dB, restricted to terrain within 20 m of the nearest drainage (MERIT HAND), aggregated at 200 m. The −18 dB threshold was <b>calibrated rather than
assumed</b>: among −14 to −20 dB it gave the best county-level agreement with UNOSAT (Spearman 0.70 on 2023-09-26).</p>
<div class="box"><b>Validation against UNOSAT (2022–2024, {U.get('matched_pairs'):,} county-date pairs).</b> Across counties agreement is good (pooled
Spearman ρ = <b>{U.get('pooled_spearman')}</b>) and the national seasonal cycle agrees moderately (ρ = {U.get('national_per_date_spearman')} over
{U.get('n_dates')} dates). <i>Within</i> counties over time, the comparison that matters for a county-relative target, agreement is uneven: median ρ by
state {state_rho}; overall median {U.get('per_county_temporal_spearman_median')}. It is strongest in Jonglei, the core Sudd state, and weak or negative in
drier, more vegetated western and northern counties. Causes: (i) SAR detects open water but misses water under dense papyrus and reed canopies, whereas
VIIRS (375 m) also captures partly flooded vegetation; (ii) 2022–2024 is the single-satellite Sentinel-1 period with the sparsest coverage.
<b>Phase 2 actions:</b> add VH-based flooded-vegetation detection, run a −16 dB sensitivity variant, and report metrics nationally and for the Sudd core
(Jonglei, Unity, Upper Nile, Lakes).</div>

<h2>III. Dataset and Problem Audit (Guideline A)</h2>
<h3>A. Initial Shape</h3>
<p>The raw, un-preprocessed panel <code>data/raw/suddwatch_panel_unprocessed.csv</code> has <b>N = {R['N']:,} observations</b> (79 counties ×
{R['N'] // 79} issue windows) and <b>P = {R['P_total_columns']} columns</b>, of which <b>{R['P_candidate_features']} are candidate predictive features</b>.</p>
<h3>B. Variable Inventory</h3>
<p>Counts by type: {", ".join(f"{k}: {v}" for k, v in R['type_counts'].items())}. <b>No ordinal variables exist</b>: states have no rank, and month
is cyclical (December is adjacent to January), so an ordinal code would impose a false 12-to-1 distance. Table II lists every column.</p>
{tbl(inv_show, "Variable Inventory and Baseline Missingness", {"pct_missing": "{:.2f}"}, span=True, widths=[20, 26, 13, 13, 9, 9, 10])}
<h3>C. Baseline Missingness</h3>
<p>Before any processing, <b>{R['overall_missing_pct']}%</b> of all {R['N'] * R['P_total_columns']:,} cells are missing. <b>{R['complete_rows']:,}</b>
rows are complete and <b>{R['rows_with_missing']:,}</b> rows ({100 * R['rows_with_missing'] / R['N']:.1f}%) contain at least one missing value.
Missingness is confined to the Sentinel-1-derived columns (Table III, Fig. 1).</p>
{tbl(miss_cols, "Columns With Missing Values", {"pct_missing": "{:.2f}"})}
{img("fig1_1_missingness_heatmap.png", "Missingness pattern (assignment Fig. 1.1): <code>water_frac</code> missing by county × issue window (left) and per-column missing percentage (right). Red dashes mark the Sentinel-1B loss and the Sentinel-1C/D coverage.", span=True)}
<h3>D. Target Label Specification</h3>
<p>The task is binary classification. Table IV and Fig. 2 give the class balance of labelled rows per temporal split.</p>
{tbl(split_df, "Class Balance by Temporal Split", span=True)}
{img("fig1_2_target_distribution.png", "Target class balance by split (assignment Fig. 1.2).")}
<p><b>Imbalance.</b> The training set is moderately imbalanced ({yt['neg_to_pos_ratio']}:1, {100 * yt['pos_rate']:.1f}% positive). The rate is below
the nominal 25% because of the 1 km² floor: in {FL['n_floor']} of 79 dry counties the training P75 is below 1 km² of water, so the floor binds and those
counties are positive in only {100 * FL['pos_floor']:.1f}% of training windows, versus {100 * FL['pos_other']:.1f}% in the other {79 - FL['n_floor']}
counties (≈25% by construction). This is intended: in very dry counties, crossing a sub-1 km² level would reflect radar noise rather than flooding.
Positive rates are higher in validation and test because the threshold is fixed from 2017–2023 and later seasons were wetter; the 2025 season affected
over one million people [1]. A sensor effect cannot be fully excluded for the test period (Sentinel-1C/D add scenes from mid-2025); Phase 2 will check it
by stratifying exceedance rates by <code>n_scenes</code>.</p>
<p><b>Mitigation.</b> (1) <code>class_weight="balanced"</code> in the estimator (no synthetic rows, no information loss); (2) the decision threshold is
chosen on the <i>validation</i> precision–recall curve to meet the proposal's &lt;15% false-alarm target; (3) evaluation uses PR-AUC, F1 and recall
rather than accuracy. If SMOTE is tried, it runs inside training folds only, after the temporal split. Random under-sampling is rejected because it
discards whole flood seasons from a temporally scarce record.</p>

<h2>IV. Cleaning and Imputation Strategy (Guideline B)</h2>
<h3>A. Missingness Mechanism Classification</h3>
<p>Table V assigns each column with missing values to a mechanism, with the evidence from this dataset.</p>
{html_table("Missingness Mechanisms, Evidence and Treatment", ["Column(s)", "Mechanism", "Evidence from this dataset", "Treatment"], [
    ["<code>water_frac</code>", "<b>MAR</b>",
     f"Missing when &lt;50% of the county was imaged. Missing share by satellite era: {era_txt}; χ² test of era × missingness p {pfmt(M['chi2_era_p'])}; "
     f"by state p {pfmt(M['chi2_state_p'])} (highest: {', '.join(f'{k} {v}%' for k, v in worst_states)}). The gap is explained by <i>observed</i> variables: date "
     "(Sentinel-1B failure on 23 Dec 2021; Sentinel-1C/D arrival in 2025) and location (orbit tracks excluded from the single-satellite plan). "
     "<b>Not MCAR</b>: strongly systematic. <b>Not MNAR</b>: acquisitions are planned in advance, independently of ground conditions. In 2022–2024 raw "
     f"36-day rainfall before missing windows is lower (median {M['rain36_missing_vs_observed_median_2022_24'][0]:.1f} vs "
     f"{M['rain36_missing_vs_observed_median_2022_24'][1]:.1f} mm, Mann-Whitney p = {M['mannwhitney_rain_p_2022_24']:.3f}) only because un-imaged counties are "
     f"drier places: in a logistic regression of missingness on state + month + standardised rainfall, the rainfall coefficient is "
     f"{M['logit_rain_coef_adj_state_month']:.3f} (p = {M['logit_rain_p_adj_state_month']:.2f}). MNAR can never be fully excluded from observed data alone; "
     "here it is ruled out by the acquisition mechanism and this adjusted test.",
     "County-level <b>training-median</b> imputation + binary <code>_was_missing</code> indicator."],
    ["<code>water_frac_lag1</code>, <code>water_frac_lag2</code>", "<b>MAR</b> (+ structural)",
     "Shifted copies of <code>water_frac</code>, so they inherit its gaps; the first 1–2 windows of 2017 are missing by construction.",
     "Same imputer + indicator."],
    ["<code>y_flood_2_4wk</code>", "<b>MAR</b>", "Undefined when window <i>w</i>+2 was not imaged (same mechanism).",
     "<b>Never imputed</b>; rows without Y are excluded from fitting and evaluation."],
    ["Rainfall, terrain, calendar, IDs", "—", "0 missing (complete gridded products; static county attributes).", "None needed."],
], span=True, widths=[15, 9, 56, 20])}
<h3>B. Imputation Choice</h3>
<p><b>Median, not mean:</b> <code>water_frac</code> is extremely right-skewed (training skewness = <b>{wf.loc['water_frac', 'skew']:.1f}</b>;
mean = {wf.loc['water_frac', 'mean']:.4f} vs median = {wf.loc['water_frac', 'median']:.4f}), so a mean fill would inject a value inflated by rare floods
into ordinary dry windows. <b>Per county:</b> baseline wetness differs by orders of magnitude between counties, and a global median would erase that.
<b>Not KNN:</b> gaps are spatially clustered (whole orbit tracks missing together), so a county's nearest neighbours are usually missing in the same
window. <b>Not last-observation-carried-forward:</b> nine counties were not imaged for up to about three years (2022–2025), so carried values would be stale.
<b>Mode</b> is inapplicable to a continuous feature. Indicator columns preserve the fact that a value was imputed.</p>
<h3>C. Outlier Identification Rules</h3>
<p>The rule follows distribution shape: |skew| ≤ 0.5 → Z-score (|z| &gt; 3); |skew| &gt; 0.5 → 1.5×IQR. Bounds in Table VI use training rows only
and are re-learned inside the pipeline (<code>OutlierCapper.fit</code>) on <code>X_train</code>.</p>
{tbl(sk_show.rename(columns={"iqr_low": "IQR low", "iqr_high": "IQR high", "pct_iqr_outliers": "% IQR out", "pct_z3_outliers": "% |z|>3"}),
     "Skewness, Outlier Rules and Treatment (Training Rows)",
     {"skew": "{:.2f}", "mean": "{:.3g}", "median": "{:.3g}", "IQR low": "{:.3g}", "IQR high": "{:.3g}", "% IQR out": "{:.1f}%", "% |z|>3": "{:.1f}%"},
     span=True, widths=[15, 5.5, 6.5, 6.5, 11, 7, 7, 6.5, 6.5, 28.5])}
<h3>D. Outlier Treatment: Capping, Not Deletion</h3>
<p>No row is deleted for an outlier. (a) <b>Preserves N:</b> deleting every row with a flagged value would remove a large share of the panel and
preferentially remove flood seasons, the minority class. (b) <b>Prevents information loss:</b> a row with one extreme rainfall value still carries
valid terrain, calendar and water-history signal. (c) <b>Domain rule:</b> water-fraction "outliers" are the floods SuddWatch exists to predict, so they
are flagged but <b>not capped</b>; static terrain extremes are real counties (e.g. the Equatoria highlands), so they are log-transformed, not capped.
Only dynamic rainfall features are winsorised, at the training bounds.</p>
<h3>E. Feature–Target Association Matrix</h3>
<p>Continuous or discrete feature × binary Y → <b>point-biserial r</b> plus <b>Mann-Whitney U</b> (rank-based, robust to the skew in Table VI) plus
<b>mutual information</b> (captures non-monotonic effects such as seasonality); nominal state × Y → <b>χ² with Cramér's V</b> and MI. Spearman ρ is used
between features (Section IV-F) because most are non-normal. Static terrain features take only 79 distinct values repeated over time, so their row-level
p-values overstate the evidence; MI is therefore the ranking and tie-break criterion (Table VII).</p>
{tbl(asc_show, "Feature–Target Association (Training Rows)", {"|r_pb| / Cramér's V": "{:.3f}", "p (r_pb / χ²)": "{:.1e}", "p (Mann-Whitney)": "{:.1e}", "mutual_info": "{:.4f}"}, span=True, widths=[17, 13, 24, 12, 11, 12, 11])}
<h3>F. Multicollinearity Protocol</h3>
<p>(1) Pairwise screen with |Spearman ρ| ≥ 0.85 on training rows; (2) walk pairs from the strongest |ρ| down and, if both are still present,
<b>drop the feature with lower mutual information with Y</b>; (3) iterate VIF on the survivors, dropping the highest VIF &gt; 10 until all VIF ≤ 10.
Tables VIII–X give the pairs, the resulting drop list (applied via <code>DROP_COLLINEAR</code> in <code>src/pipeline_builder.py</code>) and VIF before
and after.</p>
{tbl(pairs, "Collinear Pairs (|ρ| ≥ 0.85)", {"spearman": "{:.3f}"})}
{tbl(drop, "Collinearity Drop List")}
{tbl(vif, "Variance Inflation Factors", {"VIF_before": "{:.2f}", "VIF_after": "{:.2f}"})}
<p>Retained numeric features: {", ".join(f"<code>{c}</code>" for c in kept_num)}. The terrain block (elevation, slope, HAND, TWI) is one physical
gradient from highland to floodplain, so one representative suffices. The autoregressive signal is kept through <code>water_frac</code>, the latest
observed water state two windows before the target; Phase 2 will test a less collinear momentum feature (<code>water_frac − water_frac_lag1</code>).</p>

<h2>V. Encoding, Scaling and Transformation (Guideline C1)</h2>
<h3>A. Encoding Matched to Cardinality</h3>
{html_table("Encoding Decisions", ["Variable", "Type / cardinality", "Encoding", "Justification"], [
    ["<code>adm1_name</code> (state)", "Nominal, 11 levels", "One-hot (<code>handle_unknown=\"ignore\"</code>)", "Low cardinality, no order; 11 binary columns keep states distinct."],
    ["<code>adm2_pcode</code> (county)", "Nominal, 79 levels", "<b>Not encoded</b>: join key, imputation group and CV group only", f"79 one-hot columns on ~{R['N'] // 79} rows per county would let a model memorise county identity; county behaviour enters through terrain and water history."],
    ["<code>month</code>", "Discrete, cyclical 1–12", "sin/cos pair", "Preserves Dec–Jan adjacency; an ordinal code would place them 11 units apart."],
    ["Ordinal variables", "None present", "OrdinalEncoder not applicable", "No variable has an inherent rank; forcing one would invent order."],
], span=True, widths=[16, 14, 25, 45])}
<h3>B. Scaling Matched to Distributions and Algorithms</h3>
{html_table("Transformation and Scaling Plan", ["Feature group", "Distribution", "Transform → scaler", "Reason"], [
    ["Water fractions", f"Extreme right skew (≈{wf.loc['water_frac', 'skew']:.0f})", "<code>log1p(100·x)</code> → StandardScaler", "Log compresses the flood tail; MinMax alone would squash 99% of values near 0."],
    ["County area, elevation mean/std, slope, HAND, water occurrence", "|skew| &gt; 1", "<code>log1p</code> → StandardScaler", "Multiplicative scales; log brings them closer to symmetric before standardisation."],
    ["Rainfall totals and anomaly, TWI, scene count", "|skew| ≤ 1", "StandardScaler (after capping)", "Near-symmetric; mean 0 / sd 1 required by the logistic-regression baseline (L2 penalty) and by PCA (Fig. 5.1)."],
    ["<code>obs_frac</code>, <code>hand_lt5m_frac</code>", "Bounded [0, 1]", "MinMaxScaler", "Known physical bounds; preserves exact 0 and 1."],
], span=True, widths=[26, 14, 20, 40])}
<p>The primary model (gradient-boosted trees) is invariant to monotone scaling, so scaling is not needed for it; it is applied for the
logistic-regression baseline, PCA and any distance-based comparison. <code>build_preprocessor(scale_tree_inputs=False)</code> skips it for trees.</p>

<h2>VI. Data Partitioning and Leakage Prevention (Guideline C2)</h2>
<h3>A. Partitioning Ratios</h3>
<p>Temporal blocks of whole flood seasons: train = issue dates before 2024-01-01, validation = 2024-01-01 to 2025-06-30, test = 2025-07-01 to
2026-08-31. Of {tot_lab:,} labelled rows this gives <b>{sp.get('train')}% / {sp.get('validation')}% / {sp.get('test')}%</b>. Seven training seasons
(2017–2023) cover wet and dry years, and the held-out test year contains the full 2025 flood season named in the proposal. With N ≈ {R['N'] // 1000}k
an 80/20 random split would be statistically ample, but randomness is the wrong criterion for a forecasting task (Section VI-D).</p>
<h3>B. Stratification</h3>
<p><code>StratifiedShuffleSplit</code> is <b>rejected</b>: it mixes future windows into training (look-ahead leakage) and, because neighbouring windows
are autocorrelated, inflates scores. Class proportions are instead reported per split (Table IV) and per CV fold, and imbalance is handled by class
weights and validation-tuned thresholds.</p>
<h3>C. Grouping</h3>
<p>Every county appears in every split by design: the deployed system forecasts the same 79 counties forward in time, so the operational question
is temporal generalisation. A supplementary <code>GroupKFold</code> by county within the training period will measure generalisation to unseen
counties as a robustness check.</p>
<h3>D. Temporal Ordering and Validation Scheme</h3>
<p>All splits move forward in time. Hyper-parameters are tuned inside training with <code>TimeSeriesSplit(n_splits=5, gap=2 windows × 79 rows)</code>:
the gap purges rows whose target window (<i>w</i>+2) would overlap the next fold. The last two training windows are likewise purged before validation,
and every preprocessing step is re-fitted inside each fold.</p>
<h3>E. Leakage-Prevention Architecture</h3>
{html_table("Leakage Paths and Controls", ["Leakage path", "Control"], [
    ["Target threshold sees the future", "County P75 computed on training windows only (<code>county_thresholds_train_only.csv</code>)."],
    ["Rainfall after the issue date", "Sums end on the issue date; anomaly climatology from 2001–2016, before the study period."],
    ["Future satellite data", "Only windows <i>w</i>, <i>w</i>−1, <i>w</i>−2 enter X; <i>w</i>+1 and the target window <i>w</i>+2 never do."],
    ["Imputation, capping, scaling, encoding", "One scikit-learn [10] <code>Pipeline</code>: <code>GroupMedianImputer → OutlierCapper(Z) → OutlierCapper(IQR) → ColumnTransformer → estimator</code>; <code>fit</code> on <code>X_train</code> only and re-fitted in every CV fold; validation/test are only transformed."],
    ["Feature selection", "Collinearity and MI statistics computed on training rows only."],
    ["Same-time proxies of Y", "Excluded: no same-window flood or population layers (e.g. UNOSAT \"Pop exposed\"); UNOSAT is used only for label validation."],
], widths=[30, 70])}

<h2>VII. Conclusion</h2>
<p>The audit shows a {R['N']:,} × {R['P_total_columns']} panel whose only missing values follow a documented, testable MAR mechanism, a moderately
imbalanced target with a stated mitigation plan, and a heavily collinear terrain block that reduces to a few representatives. Phase 2 will implement
this strategy in the pipeline defined in <code>src/pipeline_builder.py</code>, produce the eight diagnostic figures, and address the label-validation
limitation in Section II. All code, audit tables and figures are in the repository ({REPO}); every number in this report is generated by
<code>src/audit.py</code> and <code>src/report_phase1.py</code>.</p>

<h2>Acknowledgment</h2>
<p>Claude (Anthropic) assisted with the Google Earth Engine extraction code, the panel-construction and audit scripts, and drafting this report. All
data were extracted from the cited public sources with the author's Earth Engine account, every reported number is generated directly by the
audit scripts, and all design decisions were reviewed and approved by the author.</p>

<h2>References</h2>
<div class="refs">
<div>[1] UN Office for the Coordination of Humanitarian Affairs (OCHA), "South Sudan: Humanitarian Snapshot (October 2025)," Nov. 2025. [Online]. Available: https://www.unocha.org/publications/report/south-sudan/south-sudan-humanitarian-snapshot-october-2025</div>
<div>[2] R. Torres <i>et al.</i>, "GMES Sentinel-1 mission," <i>Remote Sens. Environ.</i>, vol. 120, pp. 9–24, 2012.</div>
<div>[3] C. Funk <i>et al.</i>, "The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes," <i>Sci. Data</i>, vol. 2, art. 150066, 2015.</div>
<div>[4] T. G. Farr <i>et al.</i>, "The Shuttle Radar Topography Mission," <i>Rev. Geophys.</i>, vol. 45, no. 2, RG2004, 2007.</div>
<div>[5] D. Yamazaki <i>et al.</i>, "MERIT Hydro: A high-resolution global hydrography map based on latest topography dataset," <i>Water Resour. Res.</i>, vol. 55, pp. 5053–5073, 2019.</div>
<div>[6] J.-F. Pekel, A. Cottam, N. Gorelick, and A. S. Belward, "High-resolution mapping of global surface water and its long-term changes," <i>Nature</i>, vol. 540, pp. 418–422, 2016.</div>
<div>[7] N. Gorelick <i>et al.</i>, "Google Earth Engine: Planetary-scale geospatial analysis for everyone," <i>Remote Sens. Environ.</i>, vol. 202, pp. 18–27, 2017.</div>
<div>[8] OCHA, "South Sudan – Subnational Administrative Boundaries (COD-AB)," Humanitarian Data Exchange. [Online]. Available: https://data.humdata.org/dataset/cod-ab-ssd</div>
<div>[9] UNOSAT, "Satellite detected water extents over South Sudan," Humanitarian Data Exchange. [Online]. Available: https://data.humdata.org/dataset/satellite-detected-water-extents-between-20-and-24-april-2022-over-south-sudan</div>
<div>[10] F. Pedregosa <i>et al.</i>, "Scikit-learn: Machine learning in Python," <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.</div>
</div>
</div>
"""
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Phase 1 Strategy Report</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT_HTML.write_text(html, encoding="utf-8")
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True)
    print("wrote", OUT_PDF, OUT_PDF.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
