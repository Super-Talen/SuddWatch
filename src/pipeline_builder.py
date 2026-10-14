"""
SuddWatch - modular ColumnTransformer / Pipeline definitions.

Order of operations (all fitted on X_train only, re-fitted inside every CV fold):
  1. GroupMedianImputer  - county-median fill (+ missing indicators) for Sentinel-1 water features
  2. OutlierCapper      - winsorise rainfall features to training Z (|z|>3) or 1.5xIQR fences
  3. ColumnTransformer   - log1p + scaling for skewed features, MinMax for bounded [0,1] fractions,
                           StandardScaler for near-symmetric features, one-hot for state,
                           sin/cos for month
  4. estimator           - gradient-boosted trees (primary) or logistic regression (baseline)

Splitting: temporal (train < 2024-01-01 <= validation < 2025-07-01 <= test), and TimeSeriesSplit
with a 2-window gap inside train, because the target looks 2 windows ahead.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

from .transformers import CyclicalEncoder, GroupMedianImputer, OutlierCapper, log1p_percent

TARGET = "y_flood_2_4wk"
GROUP = "adm2_pcode"
WATER = ["water_frac", "water_frac_lag1", "water_frac_lag2"]             # MAR gaps -> imputed
SKEWED = ["county_area_km2", "elev_mean_m",
          "elev_std_m", "slope_mean_deg", "hand_mean_m", "perm_water_frac",
          "jrc_water_occurrence_pct"]                                    # |skew| > 1 -> log1p
BOUNDED = ["obs_frac", "hand_lt5m_frac"]                                 # fractions in [0, 1]
SYMMETRIC = ["rain_12d_mm", "rain_36d_mm", "rain_90d_mm", "rain_90d_anom_mm",
             "twi_mean", "n_scenes"]                        # |skew| <= 1 -> StandardScaler
NOMINAL = ["adm1_name"]                                                  # 11 states -> one-hot
CYCLICAL = ["month"]
CAP_Z = ["rain_36d_mm", "rain_90d_mm", "rain_90d_anom_mm"]   # |skew| <= 0.5 -> Z-score rule (|z| > 3)
CAP_IQR = ["rain_12d_mm"]                                    # |skew| > 0.5 -> 1.5 x IQR rule
# From the Phase 1 collinearity screen (phase1_strategy/audit/C_collinearity_drop_list.csv)
DROP_COLLINEAR = ['twi_mean', 'hand_mean_m', 'water_frac_lag1', 'water_frac_lag2', 'elev_std_m', 'rain_12d_mm', 'slope_mean_deg', 'perm_water_frac', 'elev_mean_m', 'rain_90d_mm']


def _keep(cols):
    return [c for c in cols if c not in DROP_COLLINEAR]


def build_preprocessor(scale_tree_inputs=True):
    """ColumnTransformer; trees are scale-invariant, so scaling can be switched off for them."""
    log_scale = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                              StandardScaler() if scale_tree_inputs else "passthrough")
    water_scale = make_pipeline(FunctionTransformer(log1p_percent, feature_names_out="one-to-one"),
                                StandardScaler() if scale_tree_inputs else "passthrough")
    ct = ColumnTransformer([
        ("water_logpct_std", water_scale, _keep(WATER)),
        ("skewed_log_std", log_scale, _keep(SKEWED)),
        ("bounded_minmax", MinMaxScaler() if scale_tree_inputs else "passthrough", _keep(BOUNDED)),
        ("symmetric_std", StandardScaler() if scale_tree_inputs else "passthrough", _keep(SYMMETRIC)),
        ("state_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL),
        ("month_cyclic", CyclicalEncoder(12), CYCLICAL),
        ("missing_flags", "passthrough", [f"{c}_was_missing" for c in _keep(WATER)]),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("impute", GroupMedianImputer(WATER, group_col=GROUP)),
        ("cap_z", OutlierCapper(_keep(CAP_Z), method="z")),
        ("cap_iqr", OutlierCapper(_keep(CAP_IQR), method="iqr")),
        ("columns", ct),
    ])


def build_pipeline(estimator=None, scale_tree_inputs=True):
    est = estimator or LogisticRegression(class_weight="balanced", max_iter=2000)
    pipe = build_preprocessor(scale_tree_inputs)
    pipe.steps.append(("model", est))
    return pipe


def temporal_cv(n_splits=5, gap_windows=2, n_counties=79):
    """Expanding-window CV over time-sorted rows; the gap purges rows whose target overlaps the fold."""
    return TimeSeriesSplit(n_splits=n_splits, gap=gap_windows * n_counties)
