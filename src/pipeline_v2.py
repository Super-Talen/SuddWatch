"""
SuddWatch Phase 3 - preprocessing pipeline v2 (Phase 2 pipeline + the Phase 2 feedback).

Order (every fitted statistic learned on X_train only, re-fitted inside every CV fold):
  1. SeasonalBlendImputer  water_frac, water_frac_lag1: last observation fading (tau = 120 d) toward the
                           county x month training median - past-only, replaces static medians (rec. A2)
  2. GroupMedianImputer    water_frac_12m_mean for the first months of 2017 (no 8 observations yet)
  3. MomentumFeature       water_frac_delta = water_frac - water_frac_lag1
  4. AnomalyFeature        water_frac_anomaly = water_frac - water_frac_12m_mean (rec. A1)
  4b. ThresholdRatioFeature water_ratio = water_frac / county threshold (training-only P75; Phase 3 run 2)
  5. OutlierCapper         rainfall, unchanged Phase 1 rules
  6. ColumnTransformer     as Phase 2, plus: signed-log anomaly/delta, log days-since-observation,
                           and ONE terrain index (PC1 of the six terrain metrics) replacing the unstable
                           MI tie-break (weakness 4). The 12-month mean is dropped: it is an exact linear
                           function of water_frac and the anomaly (VIF infinite).
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, StandardScaler

from .transformers import (AnomalyFeature, CyclicalEncoder, GroupMedianImputer, MomentumFeature,
                           OutlierCapper, SeasonalBlendImputer, ThresholdRatioFeature, log1p_percent,
                           signed_log1p_percent)

TARGET = "y_flood_2_4wk"
GROUP = "adm2_pcode"
TAU_DAYS = 120          # chosen by the hidden-year imputation experiment (phase3_modeling/feedback)
TERRAIN = ["elev_mean_m", "elev_std_m", "slope_mean_deg", "hand_mean_m", "hand_lt5m_frac", "twi_mean"]
CAP_Z = ["rain_36d_mm", "rain_90d_anom_mm"]

# raw columns the pipeline reads (the API validates requests against this list)
INPUT_COLUMNS = ["adm2_pcode", "adm1_name", "month", "n_scenes", "obs_frac",
                 "water_frac", "water_frac_last", "water_frac_days",
                 "water_frac_lag1", "water_frac_lag1_last", "water_frac_lag1_days",
                 "water_frac_12m_mean", "rain_36d_mm", "rain_90d_anom_mm",
                 "county_area_km2", "jrc_water_occurrence_pct", "y_threshold_frac", *TERRAIN]


def _fn(f):
    return FunctionTransformer(f, feature_names_out="one-to-one")


def _terrain_name(transformer, input_features):
    return np.asarray(["terrain_index"], dtype=object)


def build_preprocessor_v2(vh=False):
    water = make_pipeline(_fn(log1p_percent), StandardScaler())
    signed = make_pipeline(_fn(signed_log1p_percent), StandardScaler())
    terrain = make_pipeline(_fn(np.log1p), StandardScaler(), PCA(n_components=1),
                            FunctionTransformer(feature_names_out=_terrain_name))
    blocks = [
        ("water_logpct", water, ["water_frac"]),
        ("change_signedlog", signed, ["water_frac_delta", "water_frac_anomaly"]),
        ("gap_age_log", make_pipeline(_fn(np.log1p), StandardScaler()), ["water_frac_days"]),
        ("ratio_log", make_pipeline(_fn(np.log1p), StandardScaler()), ["water_ratio"]),
        ("terrain_index", terrain, TERRAIN),
        ("static_log", make_pipeline(_fn(np.log1p), StandardScaler()), ["county_area_km2", "jrc_water_occurrence_pct"]),
        ("bounded_minmax", MinMaxScaler(), ["obs_frac"]),
        ("symmetric_std", StandardScaler(), ["rain_36d_mm", "rain_90d_anom_mm", "n_scenes"]),
        ("state_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["adm1_name"]),
        ("month_cyclic", CyclicalEncoder(12), ["month"]),
        ("missing_flags", "passthrough", ["water_frac_was_missing"]),
    ]
    if vh:      # VH -26 dB water fraction as an extra input; left NaN where un-imaged (trees handle it)
        blocks.append(("vh26_logpct", _fn(log1p_percent), ["vh26_frac"]))
    ct = ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([
        ("impute_water", SeasonalBlendImputer(["water_frac", "water_frac_lag1"], group_col=GROUP, tau_days=TAU_DAYS)),
        ("impute_baseline", GroupMedianImputer(["water_frac_12m_mean"], group_col=GROUP, add_indicator=False)),
        ("momentum", MomentumFeature()),
        ("anomaly", AnomalyFeature()),
        ("ratio", ThresholdRatioFeature()),
        ("cap_z", OutlierCapper(CAP_Z, method="z")),
        ("columns", ct),
    ])


def build_pipeline_v2(estimator, vh=False):
    pipe = build_preprocessor_v2(vh=vh)
    pipe.steps.append(("model", estimator))
    return pipe
