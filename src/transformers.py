"""
SuddWatch - custom scikit-learn transformers.

Every statistic (medians, IQR bounds) is learned in fit() from X_train only and re-used
unchanged in transform(), so the same objects can sit inside a Pipeline / CV loop without
validation or test leakage. Planned in Phase 1; exercised in Phase 2.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def log1p_percent(x):
    """log(1 + 100*x): spreads water fractions (skew ~ 29, mostly < 1 %) before scaling."""
    return np.log1p(100 * np.asarray(x, dtype=float))


def signed_log1p_percent(x):
    """sign(x) * log(1 + 100|x|): the same compression for changes/anomalies that can be negative."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(100 * np.abs(x))


class SeasonalBlendImputer(BaseEstimator, TransformerMixin):
    """Time-aware, past-only gap filling for Sentinel-1 water features (Phase 2 feedback, rec. A2).

    A missing value at issue date t is filled as
        w * last_observed + (1 - w) * seasonal_median,   w = exp(-days_since_observation / tau_days)
    so a short gap keeps the county's most recent observed level and a long, multi-year gap
    relaxes to that county's typical level for the calendar month. last_observed and its age come
    from src/features_v2.py and use only data up to t; the seasonal (county x month) medians are
    learned in fit() from X_train only. Linear/spline interpolation is deliberately NOT used: it
    would read the value AFTER the gap, which is not yet known when a forecast is issued.
    """

    def __init__(self, columns, group_col="adm2_pcode", month_col="month", tau_days=60.0,
                 last_suffix="_last", days_suffix="_days", add_indicator=True):
        self.columns = columns
        self.group_col = group_col
        self.month_col = month_col
        self.tau_days = tau_days
        self.last_suffix = last_suffix
        self.days_suffix = days_suffix
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.seasonal_ = {c: X.groupby([self.group_col, self.month_col])[c].median() for c in self.columns}
        self.county_ = {c: X.groupby(self.group_col)[c].median() for c in self.columns}
        self.global_ = {c: X[c].median() for c in self.columns}
        return self

    def _seasonal(self, X, c):
        key = pd.MultiIndex.from_arrays([X[self.group_col], X[self.month_col]])
        s = pd.Series(self.seasonal_[c].reindex(key).to_numpy(), index=X.index)
        s = s.fillna(X[self.group_col].map(self.county_[c]))
        return s.fillna(self.global_[c])

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns:
            miss = X[c].isna()
            if self.add_indicator:
                X[f"{c}_was_missing"] = miss.astype(int)
            last = X[c + self.last_suffix]
            w = np.exp(-X[c + self.days_suffix].astype(float) / self.tau_days).where(last.notna(), 0.0)
            fill = w * last.fillna(0.0) + (1 - w) * self._seasonal(X, c)
            X.loc[miss, c] = fill[miss]
        return X

    def get_feature_names_out(self, input_features=None):
        extra = [f"{c}_was_missing" for c in self.columns] if self.add_indicator else []
        return np.asarray(list(input_features) + extra, dtype=object)


class AnomalyFeature(BaseEstimator, TransformerMixin):
    """water_frac_anomaly = water_frac - water_frac_12m_mean (Phase 2 feedback, rec. A1).

    Stateless; the 12-month mean is trailing (past-only, src/features_v2.py), so the anomaly measures
    how unusual the current water level is relative to the county's own recent year, which tracks
    persistent shifts such as the post-2022 inundation around Bentiu/Rubkona.
    """

    def __init__(self, current="water_frac", baseline="water_frac_12m_mean", name="water_frac_anomaly"):
        self.current = current
        self.baseline = baseline
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.name] = X[self.current] - X[self.baseline]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + [self.name], dtype=object)


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill NaNs with the per-county TRAINING median, falling back to the global training median.

    Why median: water-fraction features are strongly right-skewed (see audit), so the mean is
    pulled by rare extreme floods. Why per county: counties differ by orders of magnitude in
    baseline wetness (Sudd vs. Equatoria). Missingness is MAR (Sentinel-1 acquisition plan),
    so an explicit indicator column is added to let the model learn any residual pattern.
    """

    def __init__(self, columns, group_col="adm2_pcode", add_indicator=True):
        self.columns = columns
        self.group_col = group_col
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.group_medians_ = X.groupby(self.group_col)[self.columns].median()
        self.global_medians_ = X[self.columns].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.columns:
            miss = X[c].isna()
            if self.add_indicator:
                X[f"{c}_was_missing"] = miss.astype(int)
            fill = X[self.group_col].map(self.group_medians_[c]).fillna(self.global_medians_[c])
            X.loc[miss, c] = fill[miss]
        return X

    def get_feature_names_out(self, input_features=None):
        extra = [f"{c}_was_missing" for c in self.columns] if self.add_indicator else []
        return np.asarray(list(input_features) + extra, dtype=object)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Winsorise features to bounds learned on X_train.

    method="iqr": [Q1 - k*IQR, Q3 + k*IQR], k = 1.5   (skewed / non-normal features)
    method="z"  : [mean - z*sd, mean + z*sd], z = 3.0  (near-normal features, |skew| < 0.5)

    Capping (not row deletion) keeps N and the other valid features in that row. It is applied
    to dynamic rainfall features only: water-fraction outliers ARE the flood events we predict,
    and static terrain outliers are real counties (e.g. the Equatoria highlands), not errors.
    """

    def __init__(self, columns, method="iqr", k=1.5, z=3.0):
        self.columns = columns
        self.method = method
        self.k = k
        self.z = z

    def fit(self, X, y=None):
        X = pd.DataFrame(X)[self.columns]
        if self.method == "iqr":
            q1, q3 = X.quantile(0.25), X.quantile(0.75)
            self.lower_, self.upper_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        else:
            mu, sd = X.mean(), X.std(ddof=0)
            self.lower_, self.upper_ = mu - self.z * sd, mu + self.z * sd
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.columns] = X[self.columns].clip(self.lower_, self.upper_, axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class MomentumFeature(BaseEstimator, TransformerMixin):
    """Add water_frac_delta = water_frac - water_frac_lag1 (change since the previous window).

    Stateless and row-wise (no fitted statistics), placed after imputation so it never sees NaNs.
    Phase 2 replacement for the raw lags dropped by the collinearity rule: Spearman |rho| <= 0.20
    with every retained feature on training rows, while carrying the lag's signal (MI ~ 0.096).
    """

    def __init__(self, current="water_frac", previous="water_frac_lag1", name="water_frac_delta"):
        self.current = current
        self.previous = previous
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.name] = X[self.current] - X[self.previous]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + [self.name], dtype=object)


class ThresholdRatioFeature(BaseEstimator, TransformerMixin):
    """water_ratio = water_frac / county threshold: how far the county is from its own flood definition.

    The target is "water two windows ahead exceeds THIS county's threshold"; without this ratio the model
    must re-learn 79 thresholds from state and terrain. The thresholds come from training years only.
    """

    def __init__(self, water="water_frac", threshold="y_threshold_frac", name="water_ratio"):
        self.water = water
        self.threshold = threshold
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X[self.name] = X[self.water] / X[self.threshold]
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(input_features) + [self.name], dtype=object)


class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """Encode a periodic integer (month 1-12) as sin/cos so December and January are adjacent."""

    def __init__(self, period=12):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        v = np.asarray(X, dtype=float).reshape(-1, 1)
        ang = 2 * np.pi * (v - 1) / self.period
        return np.hstack([np.sin(ang), np.cos(ang)])

    def get_feature_names_out(self, input_features=None):
        base = input_features[0] if input_features is not None else "x"
        return np.asarray([f"{base}_sin", f"{base}_cos"], dtype=object)
