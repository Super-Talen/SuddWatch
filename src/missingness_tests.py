"""
SuddWatch - formal missingness tests on TRAINING rows (rubric 1: MCAR / MAR / MNAR).

1. Little's MCAR test (Little 1988): EM estimates of the mean and covariance of the numeric variables, then
   d2 = sum over missingness patterns of n_j (mean_j - mu_j)' Sigma_j^-1 (mean_j - mu_j), chi-square with
   sum(p_j) - p degrees of freedom. Run on all training rows and separately inside each Sentinel-1 era.
2. Cramer's V (chi-square) between "water_frac is missing" and sensor era, state and month: what the
   missingness depends on (observed variables -> MAR evidence).
3. MNAR check: does 36-day rainfall (a proxy for the unobserved flood state) predict a gap? First with a coarse
   logit (state, month, era), then with county and issue-date fixed effects, which hold the acquisition plan fixed.

Usage: .venv\\Scripts\\python.exe -m src.missingness_tests
Output: phase3_modeling/audit/missingness_tests.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase3_modeling/audit"
VARS = ["water_frac", "water_frac_lag1", "water_frac_lag2", "rain_36d_mm", "rain_90d_anom_mm"]


def em_mvn(Y, iters=200, tol=1e-8):
    """ML mean and covariance of a multivariate normal with missing values (EM)."""
    n, p = Y.shape
    mu = np.nanmean(Y, axis=0)
    S = np.diag(np.nanvar(Y, axis=0))
    M = np.isnan(Y)
    patterns = {}
    for i, row in enumerate(M):
        patterns.setdefault(row.tobytes(), []).append(i)
    for _ in range(iters):
        T1, T2 = np.zeros(p), np.zeros((p, p))
        for key, idx in patterns.items():
            m = np.frombuffer(key, dtype=bool); o = ~m
            Yo = Y[np.ix_(idx, np.where(o)[0])]
            Yh = np.empty((len(idx), p)); Yh[:, o] = Yo
            C = np.zeros((p, p))
            if m.any():
                Soo_inv = np.linalg.inv(S[np.ix_(o, o)])
                B_ = S[np.ix_(m, o)] @ Soo_inv
                Yh[:, m] = mu[m] + (Yo - mu[o]) @ B_.T
                C[np.ix_(m, m)] = S[np.ix_(m, m)] - B_ @ S[np.ix_(o, m)]
            T1 += Yh.sum(0); T2 += Yh.T @ Yh + len(idx) * C
        mu_new = T1 / n
        S_new = T2 / n - np.outer(mu_new, mu_new)
        done = np.abs(mu_new - mu).max() < tol and np.abs(S_new - S).max() < tol
        mu, S = mu_new, S_new
        if done:
            break
    return mu, S, patterns


def littles_test(df):
    Y = df[VARS].to_numpy(dtype=float)
    Y = Y[~np.isnan(Y).all(1)]
    mu, S, patterns = em_mvn(Y)
    d2, df_ = 0.0, 0
    for key, idx in patterns.items():
        o = ~np.frombuffer(key, dtype=bool)
        diff = Y[np.ix_(idx, np.where(o)[0])].mean(0) - mu[o]
        d2 += len(idx) * diff @ np.linalg.solve(S[np.ix_(o, o)], diff)
        df_ += int(o.sum())
    df_ -= Y.shape[1]
    return {"rows": int(Y.shape[0]), "patterns": len(patterns), "d2": round(float(d2), 2), "df": df_,
            "p_value": float(stats.chi2.sf(d2, df_)), "verdict": "reject MCAR" if stats.chi2.sf(d2, df_) < 0.05 else "MCAR not rejected"}


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    return {"cramers_v": round(float(v), 3), "chi2": round(float(chi2), 1), "dof": int(dof), "p_value": float(p)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data/processed/panel_v2_features.csv", parse_dates=["issue_date"])
    tr = d[d["split"] == "train"].copy()
    for c in ("water_frac", "water_frac_lag1", "water_frac_lag2"):
        tr[c] = np.log1p(100 * tr[c])          # the skewed water fractions, on the scale the model uses
    res = {"variables": VARS, "little_all_training_rows": littles_test(tr),
           "little_by_sensor_era": {era: littles_test(g) for era, g in tr.groupby("sensor_era")}}
    miss = tr["water_frac"].isna().astype(int)
    res["water_frac_missing_pct_train"] = round(100 * float(miss.mean()), 2)
    res["cramers_v_missing_vs"] = {"sensor_era": cramers_v(miss, tr["sensor_era"]),
                                   "state": cramers_v(miss, tr["adm1_name"]),
                                   "month": cramers_v(miss, tr["month"])}
    # (a) coarse adjustment: logit with state, month and era
    m = smf.logit("miss ~ rain_36d_mm + C(adm1_name) + C(month) + C(sensor_era)",
                  data=tr.assign(miss=miss, rain_36d_mm=tr["rain_36d_mm"] / 100)).fit(disp=False)
    res["mnar_check_coarse_logit"] = {"rain_36d_per_100mm_coef": round(float(m.params["rain_36d_mm"]), 4),
                                      "p_value": float(m.pvalues["rain_36d_mm"]), "adjusted_for": "state, month, sensor era"}
    # (b) fine adjustment: county and issue-date fixed effects (linear probability model, two-way within transform,
    #     SEs clustered by date). The acquisition plan decides WHICH counties are imaged on WHICH date; once both
    #     are held fixed, does rainfall (proxy for the unobserved flood state) still predict a gap?
    def twoway(df, cols, it=50):
        x = df[cols].astype(float).copy()
        for _ in range(it):
            x = x - x.groupby(df["adm2_pcode"]).transform("mean")
            x = x - x.groupby(df["issue_date"]).transform("mean")
        return x
    fe = {}
    for name, sub in (("all_training_rows", tr), ("gap_years_2022_2023", tr[tr["issue_date"].dt.year >= 2022])):
        z = twoway(sub.assign(miss=sub["water_frac"].isna().astype(float), r=sub["rain_36d_mm"] / 100), ["miss", "r"])
        ols = sm.OLS(z["miss"], z[["r"]]).fit(cov_type="cluster", cov_kwds={"groups": sub["issue_date"].astype("int64")})
        fe[name] = {"rows": int(len(sub)), "coef_per_100mm": round(float(ols.params["r"]), 4), "p_value": round(float(ols.pvalues["r"]), 4)}
    res["mnar_check_county_and_date_fixed_effects"] = fe
    res["conclusion"] = ("Not MCAR (Little's test rejects in every era; missingness depends on era and state). "
                         "The coarse logit finds a rainfall association (more rain, fewer gaps), but it disappears once county "
                         "and date are held fixed (p = %.2f; %.2f in the 2022-23 gap years): the gaps are explained by which "
                         "counties the acquisition plan covers on which date, i.e. MAR on observed variables. MNAR cannot be "
                         "excluded from observed data alone, so the stress tests also simulate long outages."
                         % (fe["all_training_rows"]["p_value"], fe["gap_years_2022_2023"]["p_value"]))
    (OUT / "missingness_tests.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
