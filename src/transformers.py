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
