"""
explainability.py
==============================================================================
AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts (Day 1-10)
Explainable AI (XAI) module -- NCMRWF prototype

Loads the artifact produced by ``train_model.py`` (models/bust_detector.pkl)
and provides:

  1. ``global_feature_importance`` -- mean |SHAP value| across a sample of
     grid points, i.e. which features matter most to the model OVERALL, with
     a bar-chart PNG for a slide/demo.

  2. ``explain_bust`` -- a LOCAL explanation for a single grid point / lead
     time: computes SHAP (TreeExplainer) attributions for that one
     prediction and converts the top contributors into a plain-English
     meteorological rationale, e.g.:

         "Confidence degraded to 18.4% (Severe bust risk) primarily due to
          a sharp 1-day swing in maximum temperature (-4.2 C) and strong
          850 hPa convergence (-3.1e-5 /s), during an active monsoon trough
          over the Indo-Gangetic Plain."

Feature names match ``generate_data_and_features_real.py`` (real NCMRWF
IMDAA reanalysis). Explainers for the older synthetic-dataset features
(MSLP, spread_*, ...) are kept, so artifacts trained on either dataset work.

The artifact is entirely self-contained (model + feature schema + category
vocabulary + imputation values) -- this file has NO dependency on
``train_model.py`` and can be shipped/deployed independently of it.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python explainability.py
    python explainability.py --model-path models/bust_detector.pkl \
        --data data/processed_features_real.parquet

    # or, programmatically, from another script:
    from explainability import load_artifact, explain_bust
    artifact = load_artifact("models/bust_detector.pkl")
    result = explain_bust({"lead_day": 5, "region": "Indo_Gangetic_Plain", "T2m_tendency_1d": -4.2, ...}, artifact=artifact)
    print(result["narrative"])

Dependencies: numpy, pandas, shap, matplotlib, and whichever of
xgboost/lightgbm was used to train the loaded model.
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bust_detection.explain")

CATEGORICAL_COLS_DEFAULT = ["regime", "region", "season", "lead_day_bucket"]

# ===========================================================================
# 1. HUMAN-READABLE LABEL DICTIONARIES
# ===========================================================================

REGION_LABELS = {
    "BoB_Coast": "the Bay of Bengal coastal belt",
    "Arabian_Sea_Coast": "the Arabian Sea coastal belt",
    "Himalaya": "the Himalayan foothills",
    "Indo_Gangetic_Plain": "the Indo-Gangetic Plain",
    "Thar_Desert": "the Thar Desert region",
    "Western_Ghats": "the Western Ghats",
    "Peninsular_Plateau": "the peninsular plateau",
    "Northeast_India": "Northeast India",
}

REGIME_LABELS = {
    "monsoon_trough": "an active monsoon trough",
    "monsoon_break": "a weak / break phase of the monsoon",
    "bay_of_bengal_low": "a head-Bay-of-Bengal monsoon low",
    "western_disturbance": "an active western disturbance",
    "bay_of_bengal_cyclone": "a cyclonic depression over the Bay of Bengal",
    "quiescent": "generally quiescent synoptic conditions",
}

SEASON_LABELS = {
    "monsoon": "the monsoon season",
    "pre_monsoon": "the pre-monsoon season",
    "post_monsoon": "the post-monsoon season",
    "winter": "winter",
}

# Groups of near-redundant features -- only the single top-ranked feature per
# family is used as a narrative reason, so e.g. spread_Rain and
# spread_composite don't both show up saying almost the same thing.
FEATURE_FAMILY = {
    "pressure_gradient_hpa_per_100km": "pressure",
    "pressure_deficit": "pressure",
    "MSLP": "pressure",
    "relative_vorticity_1e5_s": "dynamics",
    "wind_shear_proxy_ms_per_100km": "dynamics",
    "wind_speed10": "dynamics",
    "U10": "dynamics",
    "V10": "dynamics",
    "spread_T2m": "spread",
    "spread_Rain": "spread",
    "spread_wind": "spread",
    "spread_composite": "spread",
    "spread_anomaly_vs_climatology": "spread",
    "clim_hist_bust_rate": "climatology",
    "clim_hist_mean_error": "climatology",
    "clim_hist_mean_spread": "climatology",
    "clim_hist_std_spread": "climatology",
    "lead_day": "lead_time",
    "lead_day_norm": "lead_time",
    "predictability_decay_factor": "lead_time",
    "Rain_fcst": "precipitation",
    "T2m": "temperature",
    # --- real IMDAA dataset features ---
    "U850": "dynamics",
    "V850": "dynamics",
    "wind_speed850": "dynamics",
    "relative_vorticity_850_1e5_s": "dynamics",
    "divergence_850_1e5_s": "dynamics",
    "wind_speed_gradient_850_per_100km": "dynamics",
    "T2m_subgrid_std": "temperature_variability",
    "T2m_recent_std": "temperature_variability",
    "T2m_tendency_1d": "temperature_variability",
    "Rain_subgrid_std": "rain_variability",
    "Rain_recent_std": "rain_variability",
    "Rain_tendency_1d": "rain_variability",
    # --- anomaly / neighbourhood / regime-index features ---
    "T2m_anom_prev3": "temperature_variability",
    "T2m_anom_to_date": "temperature_variability",
    "T2m_nbr_range": "temperature_contrast",
    "T2m_gradient_per_100km": "temperature_contrast",
    "Rain_anom_prev3": "rain_variability",
    "Rain_wet_days_prev5": "rain_variability",
    "Rain_nbr_max": "rain_nearby",
    "Rain_nbr_mean": "rain_nearby",
    "idx_bob_vorticity": "regime_strength",
    "idx_monsoon_core_rain": "regime_strength",
    # --- S2S forecast dataset features ---
    "T850": "temperature",
    "T925": "temperature",
    "T500": "temperature",
    "T850_fcst_change_1d": "temperature_variability",
    "Rain_fcst_change_1d": "rain_variability",
    "lapse_850_500": "stability",
    "shear_850_500": "dynamics",
    "Z500": "pressure",
    "U10": "dynamics",
    "V10": "dynamics",
    "wind_speed10": "dynamics",
    "lagged_spread_Rain": "spread",
    "lagged_spread_T850": "spread",
    "lagged_spread_Z500": "spread",
    "lagged_n_members": "spread",
    "land_frac": "location",
    "orography_m": "location",
    "clim_hist_mean_abs_temp_error": "climatology",
    "clim_hist_mean_abs_rain_error": "climatology",
    "lat": "location",
    "lon": "location",
}


def _fmt_region(row: pd.Series) -> str:
    r = str(row.get("region", "") or "")
    return REGION_LABELS.get(r, r.replace("_", " ") if r else "the region")


def _fmt_regime(row: pd.Series) -> str:
    rg = str(row.get("regime", "") or "")
    return REGIME_LABELS.get(rg, rg.replace("_", " ") if rg else "the current synoptic pattern")


def _fmt_lead(row: pd.Series) -> str:
    lead = row.get("lead_day", None)
    try:
        return f"Day {int(lead)}"
    except (TypeError, ValueError):
        return "this lead time"


# ===========================================================================
# 2. PER-FEATURE PLAIN-ENGLISH EXPLAINERS
# ===========================================================================


def _explain_pressure_gradient(value, row):
    return f"a steep {_fmt_region(row)} pressure gradient (~{value:.1f} hPa/100km), often a precursor to rapid intensification"


def _explain_pressure_deficit(value, row):
    if value <= 0:
        return f"forecast pressure {abs(value):.1f} hPa above its surroundings near {_fmt_region(row)}"
    return f"a forecast low-pressure area near {_fmt_region(row)} ({value:.1f} hPa below its surroundings)"


def _explain_vorticity(value, row):
    sense = "cyclonic" if value > 0 else "anticyclonic"
    return f"a strong {sense} vorticity signature (~{value:.1f}e-5 /s) near {_fmt_region(row)}"


def _explain_wind_shear(value, row):
    return f"elevated local wind shear (~{value:.1f} m/s/100km), which can disorganise convection and blur rainfall placement"


def _make_spread_explainer(label: str) -> Callable:
    def _f(value, row):
        return f"elevated {_fmt_lead(row)} ensemble spread ({label}: {value:.2f}), showing the members disagree on outcome"

    return _f


def _explain_spread_anomaly(value, row):
    return f"{_fmt_lead(row)} ensemble spread running {value:+.1f} sigma vs. its historical climatological norm for this location"


def _explain_clim_bust_rate(value, row):
    return f"a historically elevated bust rate (~{value * 100:.0f}%) for {_fmt_region(row)} at this lead time"


def _explain_clim_mean_error(value, row):
    return f"a history of larger verification errors (~{value:.1f}) at this location and lead time"


def _explain_clim_mean_spread(value, row):
    return f"a historically wide typical ensemble spread (~{value:.2f}) for {_fmt_region(row)} at this lead time"


def _explain_clim_std_spread(value, row):
    return f"high historical variability in ensemble spread (~{value:.2f}) at this location and lead time"


def _explain_lead_day(value, row):
    lead = row.get("lead_day", None)
    try:
        lead_int = int(lead)
    except (TypeError, ValueError):
        lead_int = None
    if lead_int is not None and lead_int <= 3:
        return f"a small amount of residual forecast uncertainty even at the still-early {_fmt_lead(row)} lead time"
    return f"the extended {_fmt_lead(row)} lead time, where forecast skill has already decayed substantially"


def _explain_rain_fcst(value, row):
    if value >= 50:
        return f"a heavy forecast rainfall total ({value:.1f} mm/24h), which is very hard to place precisely day to day"
    return f"a forecast rainfall total of {value:.1f} mm/24h in an area where monsoon rain switches on and off quickly"


def _explain_mslp(value, row):
    return f"a deep forecast low (MSLP {value:.1f} hPa)"


def _explain_wind_speed(value, row):
    return f"strong forecast surface winds (~{value:.1f} m/s)"


def _explain_vorticity_850(value, row):
    sense = "cyclonic" if value > 0 else "anticyclonic"
    return f"{sense} 850 hPa vorticity (~{value:.1f}e-5 /s) near {_fmt_region(row)}"


def _explain_divergence_850(value, row):
    if value < 0:
        return f"strong 850 hPa convergence ({value:.1f}e-5 /s), which feeds convection that is hard to place day to day"
    return f"850 hPa divergence ({value:+.1f}e-5 /s), suppressing rainfall that could quickly reverse"


def _explain_speed_gradient_850(value, row):
    return f"a sharp 850 hPa wind-speed gradient (~{value:.1f} m/s per 100km), marking a nearby jet or trough edge"


def _explain_wind_speed_850(value, row):
    return f"strong 850 hPa monsoon flow (~{value:.1f} m/s)"


def _explain_t2m(value, row):
    return f"a current maximum temperature of {value:.1f} C that is liable to change sharply"


def _explain_t2m_tendency(value, row):
    return f"a sharp 1-day swing in maximum temperature ({value:+.1f} C), showing the air mass is changing"


def _explain_t2m_recent_std(value, row):
    return f"unsettled recent temperatures (day-to-day std {value:.1f} C)"


def _explain_t2m_subgrid(value, row):
    return f"large temperature contrasts within the grid cell (std {value:.1f} C), typical of terrain or a moving boundary"


def _explain_rain_tendency(value, row):
    return f"a large 1-day change in rainfall ({value:+.1f} mm), showing an active, shifting rain pattern"


def _explain_rain_recent_std(value, row):
    return f"erratic recent rainfall (day-to-day std {value:.1f} mm)"


def _explain_rain_subgrid(value, row):
    return f"patchy rainfall within the grid cell (std {value:.1f} mm), i.e. convective showers that are hard to place"


def _explain_clim_abs_temp_error(value, row):
    return f"a recent track record of large temperature misses (~{value:.1f} C) at this location"


def _explain_clim_abs_rain_error(value, row):
    return f"a recent track record of large rainfall misses (~{value:.1f} mm) at this location"


def _explain_location(value, row):
    return f"the location itself ({_fmt_region(row)}), where forecasts have been failing more often"


def _explain_t2m_anom(value, row):
    word = "hotter" if value > 0 else "cooler"
    return f"today running {abs(value):.1f} C {word} than recent days at this spot, a sign the air mass is changing"


def _explain_rain_anom(value, row):
    word = "wetter" if value > 0 else "drier"
    return f"today being {abs(value):.0f} mm {word} than recent days here, so the rain pattern is shifting"


def _explain_wet_days(value, row):
    return f"rain on {value * 100:.0f}% of the last five days, an on-off pattern that is hard to carry forward"


def _explain_rain_nbr_max(value, row):
    return f"heavy rain nearby (up to {value:.0f} mm within about 1 deg) that could move in by the valid day"


def _explain_rain_nbr_mean(value, row):
    return f"widespread rain in the surrounding area (~{value:.0f} mm on average within about 1 deg)"


def _explain_t2m_contrast(value, row):
    return f"a sharp temperature contrast nearby ({value:.1f} C across about 1 deg), typical of an advancing monsoon boundary"


def _explain_t2m_gradient(value, row):
    return f"a strong temperature gradient ({value:.1f} C per 100 km), marking a boundary between air masses"


def _explain_idx_bob(value, row):
    return f"the strength of circulation over the head Bay of Bengal ({value:.1f}e-5 /s), which steers monsoon lows inland"


def _explain_idx_core_rain(value, row):
    return f"how active the monsoon is across central India ({value:.1f} mm average rain)"


def _explain_u850(value, row):
    word = "westerly (monsoon)" if value > 0 else "easterly"
    return f"{abs(value):.1f} m/s {word} flow at 850 hPa, a key control on where monsoon rain sets up"


def _explain_v850(value, row):
    word = "southerly" if value > 0 else "northerly"
    return f"{abs(value):.1f} m/s {word} flow at 850 hPa, pulling in air of a different origin"


def _explain_t850(value, row):
    return f"a forecast 850 hPa temperature of {value:.1f} C, a level where the model's temperature errors are largest here"


def _explain_t850_change(value, row):
    return f"a sharp day-to-day swing in the forecast 850 hPa temperature ({value:+.1f} C), showing a changing air mass"


def _explain_rain_change(value, row):
    return f"forecast rainfall changing fast from one day to the next ({value:+.1f} mm), a sign of a shifting rain band"


def _explain_lapse(value, row):
    if value >= 25:
        return f"a weakly stable atmosphere (850-500 hPa temperature drop {value:.1f} C), favouring convection that is hard to place"
    return f"the forecast vertical temperature structure (850-500 hPa drop {value:.1f} C)"


def _explain_shear_850_500(value, row):
    return f"strong vertical wind shear between 850 and 500 hPa ({value:.1f} m/s), which disorganises convection"


def _explain_z500(value, row):
    return f"the forecast 500 hPa height pattern ({value:.0f} m) over {_fmt_region(row)}"


def _make_lagged_spread_explainer(label: str, unit: str) -> Callable:
    def _f(value, row):
        return (f"forecasts started on different days disagreeing on {label} ({value:.1f} {unit} spread for "
                f"{_fmt_lead(row)}), a sign the situation is hard to predict")

    return _f


def _explain_orography(value, row):
    return f"mountainous terrain (~{value:.0f} m), where coarse models struggle to place rain and temperature"


def _explain_land_frac(value, row):
    return "the land-sea boundary in this grid cell, where the model's coastal representation is crude" if 0.1 < value < 0.9 else f"the surface type (land fraction {value:.1f})"


def _explain_generic(feature_name, value, row):
    label = feature_name.replace("_", " ")
    try:
        return f"an elevated {label} ({float(value):.2f})"
    except (TypeError, ValueError):
        return f"an elevated {label} ({value})"


FEATURE_EXPLAINERS: Dict[str, Callable] = {
    "pressure_gradient_hpa_per_100km": _explain_pressure_gradient,
    "pressure_deficit": _explain_pressure_deficit,
    "relative_vorticity_1e5_s": _explain_vorticity,
    "wind_shear_proxy_ms_per_100km": _explain_wind_shear,
    "spread_T2m": _make_spread_explainer("temperature"),
    "spread_Rain": _make_spread_explainer("rainfall"),
    "spread_wind": _make_spread_explainer("wind"),
    "spread_composite": _make_spread_explainer("overall"),
    "spread_anomaly_vs_climatology": _explain_spread_anomaly,
    "clim_hist_bust_rate": _explain_clim_bust_rate,
    "clim_hist_mean_error": _explain_clim_mean_error,
    "clim_hist_mean_spread": _explain_clim_mean_spread,
    "clim_hist_std_spread": _explain_clim_std_spread,
    "lead_day": _explain_lead_day,
    "lead_day_norm": _explain_lead_day,
    "predictability_decay_factor": _explain_lead_day,
    "Rain_fcst": _explain_rain_fcst,
    "MSLP": _explain_mslp,
    "wind_speed10": _explain_wind_speed,
    # --- real IMDAA dataset features ---
    "relative_vorticity_850_1e5_s": _explain_vorticity_850,
    "divergence_850_1e5_s": _explain_divergence_850,
    "wind_speed_gradient_850_per_100km": _explain_speed_gradient_850,
    "wind_speed850": _explain_wind_speed_850,
    "T2m": _explain_t2m,
    "T2m_tendency_1d": _explain_t2m_tendency,
    "T2m_recent_std": _explain_t2m_recent_std,
    "T2m_subgrid_std": _explain_t2m_subgrid,
    "Rain_tendency_1d": _explain_rain_tendency,
    "Rain_recent_std": _explain_rain_recent_std,
    "Rain_subgrid_std": _explain_rain_subgrid,
    "clim_hist_mean_abs_temp_error": _explain_clim_abs_temp_error,
    "clim_hist_mean_abs_rain_error": _explain_clim_abs_rain_error,
    "lat": _explain_location,
    "lon": _explain_location,
    # --- anomaly / neighbourhood / regime-index features ---
    "U850": _explain_u850,
    "V850": _explain_v850,
    "T2m_anom_prev3": _explain_t2m_anom,
    "T2m_anom_to_date": _explain_t2m_anom,
    "Rain_anom_prev3": _explain_rain_anom,
    "Rain_wet_days_prev5": _explain_wet_days,
    "Rain_nbr_max": _explain_rain_nbr_max,
    "Rain_nbr_mean": _explain_rain_nbr_mean,
    "T2m_nbr_range": _explain_t2m_contrast,
    "T2m_gradient_per_100km": _explain_t2m_gradient,
    "idx_bob_vorticity": _explain_idx_bob,
    "idx_monsoon_core_rain": _explain_idx_core_rain,
    # --- S2S forecast dataset features ---
    "T850": _explain_t850,
    "T850_fcst_change_1d": _explain_t850_change,
    "Rain_fcst_change_1d": _explain_rain_change,
    "lapse_850_500": _explain_lapse,
    "shear_850_500": _explain_shear_850_500,
    "Z500": _explain_z500,
    "lagged_spread_Rain": _make_lagged_spread_explainer("rainfall", "mm"),
    "lagged_spread_T850": _make_lagged_spread_explainer("850 hPa temperature", "C"),
    "lagged_spread_Z500": _make_lagged_spread_explainer("the 500 hPa pattern", "m"),
    "orography_m": _explain_orography,
    "land_frac": _explain_land_frac,
}


def _risk_level(p: float) -> str:
    if p >= 0.6:
        return "Severe"
    if p >= 0.35:
        return "High"
    if p >= 0.15:
        return "Moderate"
    return "Low"


def _compose_narrative(confidence_pct: float, risk_level: str, reason_phrases: List[str], row: pd.Series) -> str:
    regime_phrase = _fmt_regime(row)
    region_phrase = _fmt_region(row)
    context = f", during {regime_phrase} over {region_phrase}" if (regime_phrase or region_phrase) else ""

    if not reason_phrases:
        return f"Confidence remains high at {confidence_pct:.1f}% ({risk_level} bust risk); no dominant bust-risk factors were identified{context}."

    if len(reason_phrases) == 1:
        reasons_txt = reason_phrases[0]
    else:
        reasons_txt = ", ".join(reason_phrases[:-1]) + " and " + reason_phrases[-1]

    if risk_level in ("High", "Severe"):
        return f"Confidence degraded to {confidence_pct:.1f}% ({risk_level} bust risk) primarily due to {reasons_txt}{context}."
    return f"Confidence stands at {confidence_pct:.1f}% ({risk_level} bust risk), influenced by {reasons_txt}{context}."


# ===========================================================================
# 3. ARTIFACT LOADING / ENCODING (self-contained -- no train_model import)
# ===========================================================================


def load_artifact(path: str = "models/bust_detector.pkl") -> dict:
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    required = {"model", "feature_cols", "categorical_cols", "category_levels", "numeric_fill_values", "categorical_fill_values"}
    missing = required - set(artifact.keys())
    if missing:
        raise ValueError(f"Model artifact at {path} is missing expected keys: {missing}")
    return artifact


def encode_dataframe(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Vectorised encoding for many rows (used for global importance)."""
    feature_cols = artifact["feature_cols"]
    categorical_cols = artifact["categorical_cols"]
    category_levels = artifact["category_levels"]
    X = df[feature_cols].copy()
    for c in categorical_cols:
        mapping = {cat: i for i, cat in enumerate(category_levels[c])}
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    return X


def _build_feature_row(grid_point_features: dict, artifact: dict) -> pd.Series:
    """Fill any features the caller didn't supply with training-time
    medians (numeric) / modes (categorical), so ``explain_bust`` works
    gracefully with a partial feature dict, not just a full model row."""
    feature_cols = artifact["feature_cols"]
    categorical_cols = artifact["categorical_cols"]
    numeric_fill = artifact["numeric_fill_values"]
    categorical_fill = artifact["categorical_fill_values"]

    data, missing = {}, []
    for c in feature_cols:
        if c in grid_point_features and grid_point_features[c] is not None:
            data[c] = grid_point_features[c]
        else:
            missing.append(c)
            data[c] = categorical_fill.get(c, "unknown") if c in categorical_cols else numeric_fill.get(c, 0.0)
    if missing:
        logger.debug("explain_bust: filled %d unsupplied feature(s) with training fill values: %s", len(missing), missing)
    return pd.Series(data)


def _encode_row(row: pd.Series, artifact: dict) -> pd.DataFrame:
    """Encode a single raw feature row into the exact numeric matrix shape
    (column order + categorical codes) the model was trained on."""
    feature_cols = artifact["feature_cols"]
    categorical_cols = artifact["categorical_cols"]
    category_levels = artifact["category_levels"]
    X = pd.DataFrame([row[feature_cols]])
    for c in categorical_cols:
        mapping = {cat: i for i, cat in enumerate(category_levels[c])}
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in feature_cols:
        if c not in categorical_cols:
            # NaN is kept as-is: the real dataset legitimately has NaN (e.g. no
            # verification history yet on the first day) and XGBoost/LightGBM
            # were trained with those NaN handled natively. Filling them here
            # would make single-point predictions disagree with batch ones.
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X[feature_cols]


# ===========================================================================
# 4. SHAP EXPLAINER (cached) + version-robust output extraction
# ===========================================================================

_EXPLAINER_CACHE: Dict[int, "shap.TreeExplainer"] = {}


def _get_explainer(artifact: dict):
    if not _HAS_SHAP:
        raise ImportError("shap is not installed. Install with: pip install shap")
    key = id(artifact["model"])
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(artifact["model"])
    return _EXPLAINER_CACHE[key]


def _extract_positive_class_shap(shap_output, expected_value):
    """SHAP's return shape/type for binary classifiers has varied across
    versions (plain ndarray vs. a per-class list vs. a 3-D ndarray). Handle
    all three so this module works regardless of the installed shap version."""
    if isinstance(shap_output, list):
        sv = shap_output[1] if len(shap_output) > 1 else shap_output[0]
        ev = expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.atleast_1d(ev)
            ev = ev[1] if len(ev) > 1 else ev[0]
    else:
        sv = np.asarray(shap_output)
        if sv.ndim == 3:  # (n_rows, n_features, n_classes)
            sv = sv[:, :, 1]
        ev = expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.atleast_1d(ev)
            ev = ev[1] if len(ev) > 1 else ev[0]
    return sv, float(ev)


# ===========================================================================
# 5. GLOBAL EXPLANATION
# ===========================================================================


def compute_shap_matrix(artifact: dict, X: pd.DataFrame):
    """Public helper: batch SHAP attribution for many rows at once (e.g. to
    precompute a per-grid-point dominant risk factor for an operational API
    cache), returning (shap_matrix [n_rows, n_features] in log-odds space,
    base_value). Exposed separately from ``global_feature_importance`` so
    callers that need the raw matrix (not just the aggregated importances)
    don't have to reach into the underscore-prefixed internals."""
    explainer = _get_explainer(artifact)
    raw_shap = explainer.shap_values(X)
    return _extract_positive_class_shap(raw_shap, explainer.expected_value)


def global_feature_importance(
    artifact: dict, X_sample: pd.DataFrame, max_display: int = 15, save_path: Optional[str] = "models/shap_global_importance.png"
) -> pd.DataFrame:
    """Mean |SHAP value| across a sample of encoded rows -- overall feature
    importance for the trained bust detector, with an optional bar chart."""
    shap_vals, _ = compute_shap_matrix(artifact, X_sample)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    imp_df = (
        pd.DataFrame({"feature": artifact["feature_cols"], "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    if save_path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = imp_df.head(max_display).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1.5))
        ax.barh(top["feature"], top["mean_abs_shap"], color="#2b6cb0")
        ax.set_xlabel("mean |SHAP value|  (impact on bust-probability log-odds)")
        ax.set_title("Global Feature Importance -- BustGuard (NCMRWF IMDAA data)")
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Global SHAP importance plot saved to %s", save_path)

    return imp_df


# ===========================================================================
# 6. LOCAL EXPLANATION -- explain_bust()
# ===========================================================================


def explain_bust(
    grid_point_features: dict,
    artifact: Optional[dict] = None,
    artifact_path: str = "models/bust_detector.pkl",
    top_k: int = 2,
) -> dict:
    """Explain a single grid-point / lead-time forecast-bust prediction.

    Parameters
    ----------
    grid_point_features : dict
        Feature name -> value for (any subset of) the model's predictor
        columns, e.g. {"lead_day": 7, "region": "BoB_Coast",
        "regime": "bay_of_bengal_cyclone", "pressure_gradient_hpa_per_100km":
        9.4, "spread_Rain": 2.3, ...}. Any feature not supplied is filled
        with its training-time median (numeric) or mode (categorical). Note:
        imputed features are filled independently of whatever you DID
        supply, so hand-setting one strongly-correlated feature (e.g. a
        large pressure_deficit) without its correlated peers (e.g. MSLP)
        can look internally inconsistent -- for the most physically
        coherent explanation, prefer a full, mutually consistent feature
        row (e.g. one pulled straight from the processed features dataset).
    artifact : dict, optional
        A pre-loaded artifact (from ``load_artifact``); avoids re-reading
        the pickle file on every call in a batch/loop.
    artifact_path : str
        Used only if ``artifact`` is not supplied.
    top_k : int
        Number of distinct feature "families" to cite as the primary
        drivers in the narrative (default 2, matching the target style:
        "... primarily due to X and Y ...").

    Returns
    -------
    dict with keys: bust_probability, confidence_pct, risk_level,
    top_contributing_factors (list of {feature, shap_value, value}, sorted
    by |SHAP| descending), narrative (str).
    """
    if artifact is None:
        artifact = load_artifact(artifact_path)

    row = _build_feature_row(grid_point_features, artifact)
    X = _encode_row(row, artifact)

    model = artifact["model"]
    bust_probability = float(model.predict_proba(X)[:, 1][0])
    confidence_pct = round((1.0 - bust_probability) * 100.0, 1)
    risk_level = _risk_level(bust_probability)

    explainer = _get_explainer(artifact)
    raw_shap = explainer.shap_values(X)
    shap_vals, base_log_odds = _extract_positive_class_shap(raw_shap, explainer.expected_value)
    shap_row = shap_vals[0]

    contributions = [
        {"feature": f, "shap_value": float(s), "value": row[f]}
        for f, s in zip(artifact["feature_cols"], shap_row)
    ]
    contributions.sort(key=lambda d: abs(d["shap_value"]), reverse=True)

    categorical_cols = artifact["categorical_cols"]
    # Materiality floor: at very extreme (near-0 or near-1) predictions, EVERY
    # SHAP value is tiny, and the top-ranked one is often just noise -- don't
    # let it masquerade as a meaningful "reason". Require a contribution to be
    # both non-trivial in absolute log-odds terms AND a decent fraction of the
    # single largest contribution for this row before citing it.
    max_abs_shap = float(np.max(np.abs(shap_row))) if len(shap_row) else 0.0
    materiality_floor = max(0.05, 0.15 * max_abs_shap)

    used_families, candidate_reasons = set(), []
    for c in contributions:
        if c["feature"] in categorical_cols or c["shap_value"] <= materiality_floor:
            continue
        try:
            if pd.isna(c["value"]):  # e.g. no history yet on the first day -- nothing meaningful to say
                continue
        except (TypeError, ValueError):
            pass
        family = FEATURE_FAMILY.get(c["feature"], c["feature"])
        if family in used_families:
            continue
        used_families.add(family)
        candidate_reasons.append(c)
        if len(candidate_reasons) >= top_k:
            break

    reason_phrases = [
        FEATURE_EXPLAINERS.get(c["feature"], lambda v, r, fn=c["feature"]: _explain_generic(fn, v, r))(c["value"], row)
        for c in candidate_reasons
    ]
    narrative = _compose_narrative(confidence_pct, risk_level, reason_phrases, row)

    return {
        "bust_probability": bust_probability,
        "confidence_pct": confidence_pct,
        "risk_level": risk_level,
        "base_rate_log_odds": base_log_odds,
        "top_contributing_factors": contributions[: max(top_k, 5)],
        "narrative": narrative,
    }


# ===========================================================================
# 7. CLI DEMO
# ===========================================================================


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP-based explainability demo for the NCMRWF forecast-bust detector.")
    p.add_argument("--model-path", type=str, default="models/bust_detector.pkl")
    p.add_argument("--data", type=str, default="data/processed_features_real.parquet", help="Optional: dataset to pull demo examples / a global-importance sample from.")
    p.add_argument("--sample-size", type=int, default=5000, help="Rows sampled for global feature importance.")
    p.add_argument("--top-k", type=int, default=2)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not _HAS_SHAP:
        raise ImportError("shap is not installed. Install with: pip install shap")

    logger.info("Loading model artifact from %s ...", args.model_path)
    artifact = load_artifact(args.model_path)
    logger.info("Loaded %s model with %d features.", artifact["model_type"], len(artifact["feature_cols"]))

    data_path = Path(args.data)
    if data_path.exists():
        logger.info("Loading dataset from %s for demo examples + global importance sample ...", data_path)
        df = pd.read_parquet(data_path)

        rng = np.random.default_rng(0)
        sample_n = min(args.sample_size, len(df))
        sample_idx = rng.choice(len(df), size=sample_n, replace=False)
        sample_df = df.iloc[sample_idx].reset_index(drop=True)
        X_sample = encode_dataframe(sample_df, artifact)

        logger.info("Computing global SHAP feature importance on a sample of %d rows ...", sample_n)
        imp_df = global_feature_importance(artifact, X_sample)
        print("=" * 74)
        print("GLOBAL FEATURE IMPORTANCE (mean |SHAP value|)")
        print("=" * 74)
        print(imp_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # Demo: explain the model's clearest correctly-flagged bust and its
        # clearest correctly-confident reliable case within the sample (a
        # random draw could otherwise land on a borderline / mis-scored row,
        # which makes for a confusing demo).
        if "is_bust" in sample_df.columns:
            sample_df = sample_df.assign(_bust_prob=artifact["model"].predict_proba(X_sample)[:, 1])
            demo_rows = []
            bust_rows = sample_df[sample_df["is_bust"] == 1]
            reliable_rows = sample_df[sample_df["is_bust"] == 0]
            if len(bust_rows):
                demo_rows.append(("Example BUST case (model correctly flags elevated risk)", bust_rows.loc[bust_rows["_bust_prob"].idxmax()]))
            if len(reliable_rows):
                demo_rows.append(("Example RELIABLE case (model correctly stays confident)", reliable_rows.loc[reliable_rows["_bust_prob"].idxmin()]))

            for label, demo_row in demo_rows:
                features = {c: demo_row[c] for c in artifact["feature_cols"]}
                result = explain_bust(features, artifact=artifact, top_k=args.top_k)
                print("=" * 74)
                print(f"{label}  (region={demo_row.get('region')}, regime={demo_row.get('regime')}, lead_day={demo_row.get('lead_day')})")
                print(f"  bust_probability = {result['bust_probability']:.3f}   confidence = {result['confidence_pct']}%   risk = {result['risk_level']}")
                print(f"  narrative: {result['narrative']}")
    else:
        logger.info("Dataset not found at %s -- running explain_bust() on a hand-crafted example instead.", data_path)
        example = {
            "lead_day": 5,
            "lead_day_norm": 0.5,
            "lead_day_bucket": "day_4_7",
            "region": "Indo_Gangetic_Plain",
            "regime": "monsoon_trough",
            "season": "monsoon",
            "lat": 26.5,
            "lon": 82.0,
            "T2m": 38.5,
            "T2m_tendency_1d": -4.2,
            "Rain_fcst": 12.0,
            "divergence_850_1e5_s": -3.1,
            "relative_vorticity_850_1e5_s": 4.0,
        }
        result = explain_bust(example, artifact=artifact, top_k=args.top_k)
        print("=" * 74)
        print("EXAMPLE (hand-crafted grid point)")
        print(f"  bust_probability = {result['bust_probability']:.3f}   confidence = {result['confidence_pct']}%   risk = {result['risk_level']}")
        print(f"  narrative: {result['narrative']}")

    print("=" * 74)


if __name__ == "__main__":
    main()