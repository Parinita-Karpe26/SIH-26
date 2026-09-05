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

         "Confidence degraded to 22.0% (Severe bust risk) primarily due to
          a steep the Bay of Bengal coastal belt pressure gradient
          (~9.4 hPa/100km), often a precursor to rapid intensification and
          elevated Day 7 ensemble spread in rainfall (2.31), showing the
          members disagree on outcome, during a cyclonic depression over the
          Bay of Bengal over the Bay of Bengal coastal belt."

The artifact is entirely self-contained (model + feature schema + category
vocabulary + imputation values) -- this file has NO dependency on
``train_model.py`` and can be shipped/deployed independently of it.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python explainability.py
    python explainability.py --model-path models/bust_detector.pkl \
        --data data/processed_features.parquet

    # or, programmatically, from another script:
    from explainability import load_artifact, explain_bust
    artifact = load_artifact("models/bust_detector.pkl")
    result = explain_bust({"lead_day": 7, "pressure_gradient_hpa_per_100km": 9.4, ...}, artifact=artifact)
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
    return f"an active low-pressure system near {_fmt_region(row)} ({value:.1f} hPa below the climatological background)"


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
    return f"a heavy forecast rainfall total ({value:.1f} mm/24h), which ensembles historically struggle to place precisely"


def _explain_mslp(value, row):
    return f"a deep forecast low (MSLP {value:.1f} hPa)"


def _explain_wind_speed(value, row):
    return f"strong forecast surface winds (~{value:.1f} m/s)"


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
    numeric_fill = artifact["numeric_fill_values"]

    X = pd.DataFrame([row[feature_cols]])
    for c in categorical_cols:
        mapping = {cat: i for i, cat in enumerate(category_levels[c])}
        X[c] = X[c].astype(str).map(mapping).fillna(-1).astype(int)
    for c in feature_cols:
        if c not in categorical_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(numeric_fill.get(c, 0.0))
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
        ax.set_title("Global Feature Importance -- NCMRWF Forecast Bust Detector")
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
    p.add_argument("--data", type=str, default="data/processed_features.parquet", help="Optional: dataset to pull demo examples / a global-importance sample from.")
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
            "lead_day": 7,
            "lead_day_norm": 0.7,
            "region": "BoB_Coast",
            "regime": "bay_of_bengal_cyclone",
            "season": "post_monsoon",
            "pressure_gradient_hpa_per_100km": 9.4,
            "pressure_deficit": 22.0,
            "relative_vorticity_1e5_s": 18.0,
            "spread_Rain": 2.3,
            "spread_composite": 1.4,
        }
        result = explain_bust(example, artifact=artifact, top_k=args.top_k)
        print("=" * 74)
        print("EXAMPLE (hand-crafted grid point)")
        print(f"  bust_probability = {result['bust_probability']:.3f}   confidence = {result['confidence_pct']}%   risk = {result['risk_level']}")
        print(f"  narrative: {result['narrative']}")

    print("=" * 74)


if __name__ == "__main__":
    main()
