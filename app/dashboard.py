"""
app/dashboard.py
==============================================================================
AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts (Day 1-10)
Operational Streamlit dashboard -- NCMRWF prototype (SIH 2026)

A duty-officer-facing view over the same model artifact and explainability
logic used by ``app/api.py``: a country-wide bust-risk map, regional KPIs, a
lead-time confidence-decay chart, and a SHAP-based "why is confidence low
here" drill-down with a plain-English guidance summary.

------------------------------------------------------------------------------
DATA / CACHING STRATEGY
------------------------------------------------------------------------------
Unlike ``app/api.py`` (which only ever serves the single most recent forecast
cycle), this dashboard lets a duty officer explore DIFFERENT archived
scenarios via the "Synoptic Event Filter" (Monsoon Depression / Western
Disturbance / Heat Wave / Normal). That needs predictions across the WHOLE
dataset, not just the latest cycle -- so the caching is split in two tiers:

  1. ``load_model_and_data()`` (``st.cache_resource``, runs once per server
     process): loads the model, loads the full processed-features dataset,
     and runs ONE batched (SHAP-free, hence fast even at ~1.4M rows)
     ``predict_proba`` call to get bust_probability/confidence for every
     row. This is what lets the event filter and lead-time-decay chart
     span the entire dataset instantly.

  2. ``compute_dominant_factors_for_case()`` (``st.cache_data``, keyed by a
     small ``(init_time, lead_day)`` string): SHAP is comparatively
     expensive, so it is only ever run over the ~4,000-row slice currently
     on screen, and only recomputed when the user actually moves to a new
     lead day or scenario -- revisiting a previous selection is instant.

This keeps every slider/dropdown interaction fast: Streamlit reruns this
whole script on every widget change, but the two cache tiers above mean that
rerun almost never repeats real work.

Reuses (does not duplicate) logic from ``app/api.py`` and
``explainability.py``: model/artifact loading, categorical encoding, the
macro-zone geography bucketing, the SHAP dominant-risk-factor reduction, and
the plain-English ``explain_bust`` narrative generator.

------------------------------------------------------------------------------
RUNNING
------------------------------------------------------------------------------
From the project root (the folder containing app/, data/, models/):

    pip install streamlit streamlit-folium folium plotly
    streamlit run app/dashboard.py

Environment overrides (same as app/api.py):
    BUST_DATA_PATH   -- path to processed_features.parquet
    BUST_MODEL_PATH  -- path to bust_detector.pkl
==============================================================================
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
import shapely
from shapely.geometry import shape as shapely_shape
from folium.raster_layers import ImageOverlay
from scipy.ndimage import gaussian_filter, zoom as ndi_zoom
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Make sibling modules (api.py, explainability.py) importable regardless of
# the working directory `streamlit run` happens to be launched from.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
for _p in (APP_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import api as bust_api  # noqa: E402  (import after sys.path setup, by design)
import explainability as xai  # noqa: E402

# ===========================================================================
# CONFIGURATION / CONSTANTS
# ===========================================================================

DATA_PATH = bust_api.DATA_PATH
MODEL_PATH = bust_api.MODEL_PATH
INDIA_BOUNDARY_PATH = APP_DIR / "assets" / "india_boundary.geojson"

REGION_DISPLAY_NAMES: Dict[str, str] = {
    "BoB_Coast": "Odisha & Andhra Coast (Bay of Bengal)",
    "Arabian_Sea_Coast": "Konkan & Gujarat Coast (Arabian Sea)",
    "Himalaya": "Himalayan Foothills",
    "Indo_Gangetic_Plain": "Indo-Gangetic Plain",
    "Thar_Desert": "Northwest Plains (Thar Desert)",
    "Western_Ghats": "Western Ghats",
    "Peninsular_Plateau": "Deccan Plateau (Peninsular)",
    "Northeast_India": "Northeast India",
}
REGION_DISPLAY_TO_CODE = {v: k for k, v in REGION_DISPLAY_NAMES.items()}

REGIME_DISPLAY_SHORT: Dict[str, str] = {
    "monsoon_trough": "Monsoon Trough",
    "western_disturbance": "Western Disturbance",
    "bay_of_bengal_cyclone": "Bay of Bengal Depression / Cyclone",
    "quiescent": "Quiescent / Normal Conditions",
}

EVENT_FILTER_OPTIONS = [
    "Live Forecast Cycle (current)",
    "Monsoon Depression",
    "Western Disturbance",
    "Heat Wave (derived)",
    "Normal / Quiescent",
]

RISK_LEVEL_COLORS = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#ef6c00", "Severe": "#c62828"}
MACRO_ZONES = ["North", "South", "East", "West", "Central"]

# ===========================================================================
# STYLING
# ===========================================================================

CUSTOM_CSS = """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1400px; }

.ncmrwf-header {
    background: linear-gradient(135deg, #0b2545 0%, #13315c 100%);
    border-radius: 12px;
    padding: 18px 26px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.18);
}
.ncmrwf-header-accent {
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
    margin: 8px 0 18px 0;
}
.ncmrwf-title { color: #ffffff; font-size: 1.55rem; font-weight: 700; letter-spacing: 0.2px; margin: 0; }
.ncmrwf-subtitle { color: #cfe0ff; font-size: 0.92rem; margin-top: 3px; }
.ncmrwf-subtitle-small { color: #93a9cf; font-size: 0.76rem; margin-top: 3px; }
.ncmrwf-header-right { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; max-width: 46%; }
.status-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.20);
    color: #e8eefc;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    white-space: nowrap;
}
.status-ok { background: rgba(19,136,8,0.28); border-color: rgba(19,136,8,0.6); color: #c9f5c1; font-weight: 600; }

.scenario-banner {
    background: #eef4ff;
    border: 1px solid #d3e3ff;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #1d2939;
    margin: 4px 0 16px 0;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e6e9ef;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    height: 100%;
}
.kpi-label { font-size: 0.74rem; color: #667085; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #0b2545; margin-top: 3px; line-height: 1.1; }
.kpi-sub { font-size: 0.76rem; color: #98a2b3; margin-top: 4px; }

.section-title { font-size: 1.15rem; font-weight: 700; color: #0b2545; margin: 6px 0 2px 0; }
.section-caption { font-size: 0.82rem; color: #667085; margin-bottom: 10px; }

.legend-row { display: flex; align-items: center; gap: 10px; font-size: 0.78rem; color: #475467; margin: 4px 0 14px 2px; }
.legend-swatch { display: inline-block; width: 26px; height: 10px; border-radius: 3px; vertical-align: middle; }
.legend-gradient { width: 160px; height: 10px; border-radius: 5px; background: linear-gradient(90deg, #2e7d32 0%, #f9c74f 50%, #c62828 100%); }

.guidance-box { background: #f8fafc; border-radius: 10px; padding: 16px 20px; margin-top: 4px; }
.guidance-box-title { font-weight: 700; font-size: 1.0rem; color: #0b2545; margin-bottom: 4px; }
.guidance-box-meta { font-size: 0.82rem; color: #475467; margin-bottom: 8px; }
.guidance-box-text { font-size: 0.98rem; color: #1d2939; line-height: 1.55; }

.footer-note { font-size: 0.74rem; color: #98a2b3; text-align: center; margin-top: 28px; }
</style>
"""

# ===========================================================================
# DATA LOADING (cached across the whole app session)
# ===========================================================================


@st.cache_resource(show_spinner=False)
def load_india_geometry():
    """Load India's national boundary (mainland + island territories) from a
    locally-shipped, pre-simplified GeoJSON asset -- no runtime network
    dependency. Source: Natural-Earth-derived country boundaries, simplified
    with Shapely (tolerance 0.03 deg) to a resolution appropriate for our
    0.5 deg data grid. Used to mask both the map visual and every "% of
    India" style statistic to the country's actual territory, since the
    raw data grid is a rectangular lat/lon bounding box that -- as observed
    directly on this dataset -- is only ~28% India by area (the rest is
    Pakistan, Afghanistan, China, Nepal, Bangladesh, Myanmar, Sri Lanka, and
    open ocean)."""
    import json

    with open(INDIA_BOUNDARY_PATH) as f:
        geojson = json.load(f)
    return shapely_shape(geojson)


@st.cache_resource(show_spinner="Loading model artifact and forecast dataset ...")
def load_model_and_data():
    artifact = xai.load_artifact(str(MODEL_PATH))
    df = pd.read_parquet(DATA_PATH)

    X_all = xai.encode_dataframe(df, artifact)
    bust_probability = artifact["model"].predict_proba(X_all)[:, 1]
    df = df.copy()
    df["bust_probability"] = bust_probability
    df["confidence_pct"] = (1.0 - bust_probability) * 100.0
    df["macro_zone"] = bust_api.assign_macro_zone(df["lat"].to_numpy(dtype=float), df["lon"].to_numpy(dtype=float))
    df["region_display"] = df["region"].astype(str).map(REGION_DISPLAY_NAMES).fillna(df["region"].astype(str))

    india_geom = load_india_geometry()
    df["is_india"] = shapely.contains_xy(india_geom, df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float))

    # Every statistic (case-scenario matching, KPIs, region confidence,
    # drill-down defaults, decay chart) is computed from India-only points;
    # the full (unmasked) df is kept only so the map's raster field has
    # complete spatial coverage to interpolate from before being visually
    # clipped to the real border -- see build_probability_raster.
    df_india = df[df["is_india"]]

    case_meta = (
        df_india.groupby("init_time")
        .agg(regime=("regime", "first"), max_T2m=("T2m", "max"), mean_T2m=("T2m", "mean"))
        .reset_index()
        .sort_values("init_time")
    )
    heatwave_threshold = float(df_india["T2m"].quantile(0.90))

    return artifact, df, case_meta, heatwave_threshold, india_geom


@st.cache_data(show_spinner="Computing SHAP-based risk drivers for this forecast view ...")
def compute_dominant_factors_for_case(_artifact: dict, _case_df: pd.DataFrame, cache_key: str) -> list:
    X = xai.encode_dataframe(_case_df, _artifact)
    return bust_api.compute_dominant_risk_factor(_artifact, X)


def select_case_init_time(case_meta: pd.DataFrame, event_filter: str, heatwave_threshold: float) -> Optional[pd.Timestamp]:
    """Pick which archived forecast case (init_time) to display for the
    selected scenario filter -- the most recent case whose regime (and, for
    the Heat Wave proxy, peak temperature) matches. ``case_meta`` is
    computed from India-only grid points, so e.g. the Heat Wave threshold
    reflects India's own temperatures, not a neighboring desert's."""
    if event_filter == "Live Forecast Cycle (current)":
        return case_meta["init_time"].max()
    if event_filter == "Monsoon Depression":
        candidates = case_meta.loc[case_meta["regime"] == "bay_of_bengal_cyclone", "init_time"]
    elif event_filter == "Western Disturbance":
        candidates = case_meta.loc[case_meta["regime"] == "western_disturbance", "init_time"]
    elif event_filter == "Heat Wave (derived)":
        candidates = case_meta.loc[(case_meta["regime"] == "quiescent") & (case_meta["max_T2m"] >= heatwave_threshold), "init_time"]
    elif event_filter == "Normal / Quiescent":
        candidates = case_meta.loc[(case_meta["regime"] == "quiescent") & (case_meta["max_T2m"] < heatwave_threshold), "init_time"]
    else:
        candidates = case_meta["init_time"]
    if len(candidates) == 0:
        return None
    return candidates.max()


def bust_prob_to_hex(p: float, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Green (low risk) -> amber -> red (critical) 3-stop gradient, applied
    to p after normalising against [vmin, vmax] -- see
    ``compute_color_scale_range`` for why this may not always be the literal
    0-1 probability scale."""
    norm = 0.0 if vmax <= vmin else (p - vmin) / (vmax - vmin)
    norm = float(np.clip(norm, 0.0, 1.0))
    stops = [(0.0, (46, 125, 50)), (0.5, (249, 199, 79)), (1.0, (198, 40, 40))]
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if p0 <= norm <= p1:
            t = (norm - p0) / (p1 - p0) if p1 > p0 else 0.0
            rgb = tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3))
            return "#{:02x}{:02x}{:02x}".format(*rgb)
    return "#c62828"


def compute_color_scale_range(lead_df: pd.DataFrame, scale_mode: str) -> Tuple[float, float]:
    """Decide what [vmin, vmax] the green->red gradient should span for this
    view.

    "adaptive" stretches to the 2nd-98th percentile of the CURRENT view's own
    bust_probability values. This matters a lot in practice: at short lead
    times, probability is often genuinely low and tightly clustered (e.g.
    0.0005-0.007 at Day 1-3 in this dataset) -- under a fixed 0-1 scale that
    entire range renders as visually-indistinguishable green, hiding real
    (if small) spatial risk variation. Adaptive scaling always shows that
    relative pattern, at the cost of the colour scale's meaning shifting
    between views -- which is exactly why the legend discloses the actual
    numeric range being shown, rather than leaving "red" ambiguous.
    "fixed" always maps the literal 0.0-1.0 probability range, so colours
    stay directly comparable across different lead days/views.
    """
    values = lead_df["bust_probability"].to_numpy(dtype=float)
    if scale_mode == "adaptive":
        vmin = float(np.percentile(values, 2))
        vmax = float(np.percentile(values, 98))
        if vmax - vmin < 1e-6:
            vmin, vmax = float(values.min()), float(values.max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    return vmin, vmax


def _apply_risk_gradient(normalized: np.ndarray) -> np.ndarray:
    """Map values already normalised to [0, 1] through the green->amber->red
    gradient, returning an (H, W, 4) uint8 RGBA array."""
    stops = [(0.0, (46, 125, 50)), (0.5, (249, 199, 79)), (1.0, (198, 40, 40))]
    rgba = np.zeros((*normalized.shape, 4), dtype=np.uint8)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        mask = (normalized >= p0) & (normalized <= p1)
        denom = (p1 - p0) if p1 > p0 else 1.0
        t = (normalized[mask] - p0) / denom
        for ch in range(3):
            rgba[..., ch][mask] = (c0[ch] + (c1[ch] - c0[ch]) * t).astype(np.uint8)
    rgba[..., 3] = 210  # semi-opaque, so basemap labels remain legible underneath
    return rgba


def build_probability_raster(
    lead_df: pd.DataFrame, vmin: float, vmax: float, india_geom, upsample: int = 4
) -> Tuple[np.ndarray, List[List[float]]]:
    """Rasterise bust_probability into a smooth, georeferenced RGBA image for
    a folium ImageOverlay, using the given [vmin, vmax] color-scale range
    (see ``compute_color_scale_range``), clipped to India's actual national
    boundary (not the full rectangular data grid).

    The grid is a REGULAR 0.5-degree lattice, not scattered points -- so a
    point-based heat-blob layer (folium.plugins.HeatMap) is the wrong tool:
    at country-scale zoom its fixed pixel radius doesn't fully bridge the
    gaps between grid columns/rows, producing visible banding artifacts.
    Rendering our own raster avoids that entirely and looks smooth at any
    zoom level, since Leaflet stretches a single image rather than
    accumulating thousands of individual blobs.

    ``lead_df`` should be the FULL (unmasked) grid, not just the India-only
    subset: the smooth field is interpolated from complete rectangular
    coverage first, then clipped to the real border as a final step -- this
    avoids interpolation holes/artifacts right at the coastline that would
    appear if the input itself already had India-external points removed.
    """
    lats = np.sort(lead_df["lat"].unique())
    lons = np.sort(lead_df["lon"].unique())
    grid = lead_df.pivot_table(index="lat", columns="lon", values="bust_probability", aggfunc="mean")
    grid = grid.reindex(index=lats, columns=lons)
    grid_vals = np.nan_to_num(grid.to_numpy(dtype=float), nan=0.0)

    # Row 0 of an image is its TOP (north) edge; our grid was built with lat
    # ascending (south -> north), so flip vertically to match.
    grid_vals = np.flipud(grid_vals)

    normalized = np.clip((grid_vals - vmin) / (vmax - vmin), 0.0, 1.0)

    # Upsample + lightly smooth so the coarse 0.5 deg lattice reads as a
    # continuous risk surface rather than a blocky pixel grid.
    smooth = ndi_zoom(normalized, upsample, order=3)
    smooth = gaussian_filter(smooth, sigma=1.0)
    smooth = np.clip(smooth, 0.0, 1.0)

    rgba = _apply_risk_gradient(smooth)

    # Clip to India's real border: build a lat/lon coordinate for every pixel
    # of the upsampled image and hide (alpha=0) anything outside the polygon.
    lat_min, lat_max, lon_min, lon_max = float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max())
    out_h, out_w = smooth.shape
    pixel_lats = np.linspace(lat_max, lat_min, out_h)  # row 0 = north, matching the flip above
    pixel_lons = np.linspace(lon_min, lon_max, out_w)
    lon_grid, lat_grid = np.meshgrid(pixel_lons, pixel_lats)
    inside_india = shapely.contains_xy(india_geom, lon_grid, lat_grid)
    rgba[..., 3] = np.where(inside_india, rgba[..., 3], 0)

    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    return rgba, bounds


def find_nearest_grid_point(lead_df: pd.DataFrame, lat: float, lon: float) -> Optional[pd.Series]:
    if len(lead_df) == 0:
        return None
    d2 = (lead_df["lat"] - lat) ** 2 + (lead_df["lon"] - lon) ** 2
    return lead_df.loc[d2.idxmin()]


def representative_point_for_region(lead_df: pd.DataFrame, region_code: str) -> Optional[pd.Series]:
    """The single highest-bust-probability point in the region -- the point
    a duty officer would actually want to look at first."""
    rdf = lead_df[lead_df["region"] == region_code]
    if len(rdf) == 0:
        return None
    return rdf.loc[rdf["bust_probability"].idxmax()]


# ===========================================================================
# RENDER: HEADER
# ===========================================================================


def render_header(artifact: dict, forecast_init_time) -> None:
    st.markdown(
        f"""
        <div class="ncmrwf-header">
          <div>
            <p class="ncmrwf-title">NCMRWF Forecast Bust Detection System</p>
            <p class="ncmrwf-subtitle">Ministry of Earth Sciences (MoES) &middot; National Centre for Medium-Range Weather Forecasting</p>
            <p class="ncmrwf-subtitle-small">AI-Based Forecast Verification &middot; Day 1&ndash;10 Medium-Range Guidance &middot; Prototype build for Smart India Hackathon 2026</p>
          </div>
          <div class="ncmrwf-header-right">
            <span class="status-pill status-ok">&#9679; SYSTEM OPERATIONAL</span>
            <span class="status-pill">Model: {artifact['model_type'].upper()}</span>
            <span class="status-pill">Forecast Cycle: {forecast_init_time}</span>
            <span class="status-pill">Server: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</span>
          </div>
        </div>
        <div class="ncmrwf-header-accent"></div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# RENDER: KPI CARDS
# ===========================================================================


def kpi_card(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_kpi_cards(lead_df: pd.DataFrame, alert_threshold: float) -> None:
    high_risk_pct = float((lead_df["bust_probability"] >= alert_threshold).mean() * 100.0)
    avg_confidence = float(lead_df["confidence_pct"].mean())

    zone_alert_frac = lead_df.groupby("macro_zone")["bust_probability"].apply(lambda s: (s >= alert_threshold).mean())
    alert_regions = sorted(zone_alert_frac[zone_alert_frac > 0.01].index.tolist())

    driver_counts = lead_df.loc[lead_df["dominant_risk_factor"] != bust_api.NO_RISK_LABEL, "dominant_risk_factor"].value_counts()
    top_driver = driver_counts.index[0] if len(driver_counts) else bust_api.NO_RISK_LABEL

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "High-Risk Area (% of India)", f"{high_risk_pct:.1f}%", f"P(bust) &ge; {alert_threshold:.2f}")
    kpi_card(c2, "Avg. System Confidence", f"{avg_confidence:.1f}%", "Mean across all cached grid points")
    kpi_card(c3, "Alert Regions", f"{len(alert_regions)} / 5", ", ".join(alert_regions) if alert_regions else "None currently flagged")
    kpi_card(c4, "Dominant Risk Driver", top_driver, "Most common cause among flagged points")


# ===========================================================================
# RENDER: MAP
# ===========================================================================


def render_map(lead_df_full: pd.DataFrame, lead_df_india: pd.DataFrame, map_mode: str, vmin: float, vmax: float, india_geom, key: str) -> dict:
    """``lead_df_full`` is the FULL (unmasked) grid, needed only as the
    raster's interpolation input (see build_probability_raster).
    ``lead_df_india`` is the India-only subset (with dominant_risk_factor
    already attached by the caller) used for the point-marker mode."""
    center_lat, center_lon = float(lead_df_india["lat"].mean()), float(lead_df_india["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="OpenStreetMap", control_scale=True)

    if map_mode == "Heatmap (recommended)":
        rgba, bounds = build_probability_raster(lead_df_full, vmin, vmax, india_geom)
        ImageOverlay(image=rgba, bounds=bounds, opacity=0.78, interactive=False, cross_origin=False).add_to(m)
    else:
        # Decimated grid points (India-only) so the browser only has to
        # render a few hundred vector markers (with hover tooltips) instead
        # of thousands, and never a marker sitting in a neighbouring country.
        lats = np.sort(lead_df_india["lat"].unique())
        lons = np.sort(lead_df_india["lon"].unique())
        keep_lat = set(lats[::2])
        keep_lon = set(lons[::2])
        sampled = lead_df_india[lead_df_india["lat"].isin(keep_lat) & lead_df_india["lon"].isin(keep_lon)]
        for row in sampled.itertuples():
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=5,
                color=None,
                fill=True,
                fill_color=bust_prob_to_hex(row.bust_probability, vmin, vmax),
                fill_opacity=0.8,
                weight=0,
                tooltip=(
                    f"({row.lat:.1f}, {row.lon:.1f})<br>"
                    f"Bust probability: {row.bust_probability:.3f}<br>"
                    f"Confidence: {row.confidence_pct:.0f}%<br>"
                    f"Driver: {row.dominant_risk_factor}<br>"
                    f"Region: {row.region_display}"
                ),
            ).add_to(m)

    map_data = st_folium(m, height=560, use_container_width=True, key=key, returned_objects=["last_clicked"])
    return map_data


def render_map_legend(vmin: float, vmax: float, scale_mode: str) -> None:
    if scale_mode == "adaptive":
        caption = f"Adaptive scale for this view: {vmin:.3f} (green) &rarr; {vmax:.3f} (red)"
    else:
        caption = f"Fixed scale: {vmin:.2f} (green) &rarr; {vmax:.2f} (red)"
    st.markdown(
        f"""<div class="legend-row">
            <span>{caption}</span><span class="legend-gradient"></span>
        </div>""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# RENDER: SHAP WATERFALL / BAR CHART
# ===========================================================================


def build_shap_waterfall_figure(result: dict) -> go.Figure:
    factors = result["top_contributing_factors"]
    base = result["base_rate_log_odds"]
    p = np.clip(result["bust_probability"], 1e-6, 1 - 1e-6)
    final_log_odds = float(np.log(p / (1 - p)))
    other = final_log_odds - base - sum(f["shap_value"] for f in factors)

    labels = ["Base rate"] + [f["feature"].replace("_", " ") for f in factors] + ["Other features", "Final prediction"]
    values = [base] + [f["shap_value"] for f in factors] + [other, final_log_odds]
    measures = ["absolute"] + ["relative"] * len(factors) + ["relative", "total"]

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=measures,
            y=labels,
            x=values,
            connector={"line": {"color": "rgba(120,120,120,0.35)"}},
            decreasing={"marker": {"color": "#2e7d32"}},
            increasing={"marker": {"color": "#c62828"}},
            totals={"marker": {"color": "#0b2545"}},
        )
    )
    fig.update_layout(
        title="SHAP Contribution Waterfall (log-odds of forecast bust)",
        height=380, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
    )
    return fig


def build_shap_bar_figure(result: dict) -> go.Figure:
    factors = sorted(result["top_contributing_factors"], key=lambda f: abs(f["shap_value"]))
    labels = [f["feature"].replace("_", " ") for f in factors]
    values = [f["shap_value"] for f in factors]
    colors = ["#c62828" if v > 0 else "#2e7d32" for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(
        title="Top Feature Impact on Bust Probability (SHAP, log-odds)",
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="SHAP value  (red = increases bust risk, green = reduces it)",
    )
    return fig


def render_guidance_box(result: dict, point_row: pd.Series) -> None:
    color = RISK_LEVEL_COLORS.get(result["risk_level"], "#455a64")
    st.markdown(
        f"""<div class="guidance-box" style="border-left: 6px solid {color};">
            <div class="guidance-box-title">Meteorologist Guidance Summary &mdash; Duty Officer Briefing</div>
            <div class="guidance-box-meta">
                Region: <b>{point_row['region_display']}</b> &nbsp;|&nbsp;
                Lead Time: <b>Day {int(point_row['lead_day'])}</b> &nbsp;|&nbsp;
                Confidence: <b>{result['confidence_pct']}%</b> &nbsp;|&nbsp;
                Risk Level: <b style="color:{color};">{result['risk_level']}</b>
            </div>
            <div class="guidance-box-text">{result['narrative']}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_drill_down(artifact: dict, lead_df: pd.DataFrame, map_click: Optional[dict]) -> None:
    st.markdown('<p class="section-title">Drill-Down &amp; Explainable AI (XAI) Inspector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-caption">Pick a region below, or click anywhere on the map above, to see the SHAP-based '
        "rationale behind the model's confidence at that location.</p>",
        unsafe_allow_html=True,
    )

    region_options = sorted(lead_df["region_display"].unique().tolist())
    default_region = lead_df.groupby("region_display")["confidence_pct"].mean().idxmin()
    default_idx = region_options.index(default_region) if default_region in region_options else 0

    col_sel, col_note = st.columns([1, 2])
    with col_sel:
        selected_display = st.selectbox(
            "Select region for detailed inspection", region_options, index=default_idx,
            help="Defaults to the region with the lowest average confidence in the current view.",
        )
    selected_code = REGION_DISPLAY_TO_CODE.get(selected_display, selected_display)

    point_row = None
    with col_note:
        if map_click is not None:
            point_row = find_nearest_grid_point(lead_df, map_click["lat"], map_click["lon"])
            if point_row is not None:
                st.caption(f"📍 Showing the grid point nearest your map click: ({point_row['lat']:.2f}\u00b0N, {point_row['lon']:.2f}\u00b0E) -- overrides the region dropdown.")
        if point_row is None:
            point_row = representative_point_for_region(lead_df, selected_code)
            if point_row is not None:
                st.caption(f"Showing the highest-risk grid point within {selected_display}: ({point_row['lat']:.2f}\u00b0N, {point_row['lon']:.2f}\u00b0E).")

    if point_row is None:
        st.warning("No cached grid points found for this selection.")
        return

    features = {c: point_row[c] for c in artifact["feature_cols"]}
    result = xai.explain_bust(features, artifact=artifact, top_k=2)

    m1, m2, m3 = st.columns(3)
    kpi_card(m1, "Bust Probability", f"{result['bust_probability']:.2f}")
    kpi_card(m2, "Confidence", f"{result['confidence_pct']:.1f}%")
    kpi_card(m3, "Risk Level", result["risk_level"])

    st.write("")
    tab_waterfall, tab_bar = st.tabs(["Waterfall View", "Bar Chart View"])
    with tab_waterfall:
        st.plotly_chart(build_shap_waterfall_figure(result), width='stretch')
    with tab_bar:
        st.plotly_chart(build_shap_bar_figure(result), width='stretch')

    st.write("")
    render_guidance_box(result, point_row)


# ===========================================================================
# RENDER: LEAD-TIME DECAY GRAPH
# ===========================================================================


def render_lead_time_decay(case_df: pd.DataFrame, selected_lead_day: int) -> None:
    st.markdown('<p class="section-title">Lead-Time Confidence Decay</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-caption">Average forecast confidence by region as lead time extends from Day 1 to Day 10, '
        "for the currently displayed scenario.</p>",
        unsafe_allow_html=True,
    )

    decay = case_df.groupby(["lead_day", "macro_zone"], observed=True)["confidence_pct"].mean().reset_index()
    fig = px.line(
        decay, x="lead_day", y="confidence_pct", color="macro_zone", markers=True,
        labels={"lead_day": "Forecast Lead Day", "confidence_pct": "Avg. Confidence (%)", "macro_zone": "Zone"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_range=[0, 100], height=380, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="Zone")
    fig.add_vline(x=selected_lead_day, line_dash="dash", line_color="#98a2b3")
    fig.add_annotation(x=selected_lead_day, y=102, text="Selected", showarrow=False, font=dict(size=11, color="#667085"), yref="y")
    st.plotly_chart(fig, width='stretch')


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> None:
    st.set_page_config(page_title="NCMRWF Forecast Bust Detection", page_icon="\U0001F327", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    try:
        artifact, df, case_meta, heatwave_threshold, india_geom = load_model_and_data()
    except FileNotFoundError as exc:
        st.error(
            f"Could not load a required file: {exc}\n\n"
            "Run `generate_data_and_features.py` then `train_model.py` first, "
            "or set the BUST_DATA_PATH / BUST_MODEL_PATH environment variables."
        )
        st.stop()

    latest_init_time = df["init_time"].max()
    render_header(artifact, latest_init_time)

    # --- Sidebar controls ---
    with st.sidebar:
        st.markdown("### 🎛️ Forecast Controls")
        lead_day = st.slider("Lead Time (Forecast Day)", min_value=1, max_value=10, value=3, step=1)
        event_filter = st.selectbox("Synoptic Event Filter", EVENT_FILTER_OPTIONS, index=0)

        st.markdown("### ⚠️ Alerting")
        alert_threshold = st.slider(
            "Alert Threshold (flag as bust if P \u2265 threshold)", min_value=0.0, max_value=1.0, value=0.65, step=0.01,
        )

        st.markdown("### 🗺️ Map Display")
        map_mode = st.radio("Layer style", ["Heatmap (recommended)", "Grid Points (sampled)"], index=0)
        scale_mode_label = st.radio(
            "Colour scale", ["Adaptive (relative to this view)", "Fixed (0.00 - 1.00 absolute)"], index=0,
            help="Adaptive stretches green->red to this view's own risk range -- useful because short "
                 "lead times are often uniformly low-risk and would otherwise look flat green. Fixed keeps "
                 "colours directly comparable across different lead days.",
        )
        scale_mode = "adaptive" if scale_mode_label.startswith("Adaptive") else "fixed"

        st.markdown("---")
        with st.expander("ℹ️ About this prototype"):
            st.write(
                "This dashboard is served from a fully **synthetic** NWP + ensemble-spread dataset "
                "generated for the SIH 2026 prototype (not live NCMRWF operational data). "
                "The 'Synoptic Event Filter' selects among archived synthetic forecast cases whose "
                "dominant regime matches the chosen scenario; 'Heat Wave' is a derived proxy "
                "(quiescent regime + T2m above the 90th percentile) since heat waves are not a "
                "separately modelled synoptic regime in the generator.\n\n"
                "The underlying data grid is a rectangular lat/lon box (6-38N, 68-98E) that is only "
                "~28% India by area -- the map and every statistic below are clipped/filtered to India's "
                "actual national boundary (mainland + island territories), not the full rectangle."
            )

    # --- Resolve which archived case to show (India-only case statistics) ---
    chosen_init_time = select_case_init_time(case_meta, event_filter, heatwave_threshold)
    if chosen_init_time is None:
        st.warning(f"No archived case matches '{event_filter}' in the current dataset -- showing the live forecast cycle instead.")
        chosen_init_time = latest_init_time

    case_df = df[df["init_time"] == chosen_init_time]
    case_df_india = case_df[case_df["is_india"]]
    case_regime = case_df["regime"].iloc[0] if len(case_df) else "quiescent"

    lead_df = case_df[case_df["lead_day"] == lead_day].copy()  # full rectangle: raster interpolation input only
    lead_df_india = lead_df[lead_df["is_india"]].copy()  # every statistic below uses this

    cache_key = f"{chosen_init_time}_{lead_day}"
    lead_df_india["dominant_risk_factor"] = compute_dominant_factors_for_case(artifact, lead_df_india, cache_key)

    st.markdown(
        f"""<div class="scenario-banner">
            Showing forecast cycle initialized <b>{chosen_init_time}</b> &nbsp;|&nbsp;
            Dominant synoptic regime: <b>{REGIME_DISPLAY_SHORT.get(case_regime, case_regime)}</b> &nbsp;|&nbsp;
            Lead time: <b>Day {lead_day}</b> &nbsp;|&nbsp;
            Grid points in India: <b>{len(lead_df_india):,}</b>
        </div>""",
        unsafe_allow_html=True,
    )

    # --- KPI cards ---
    render_kpi_cards(lead_df_india, alert_threshold)
    st.write("")

    # --- Map ---
    st.markdown('<p class="section-title">Forecast Bust Probability &mdash; India</p>', unsafe_allow_html=True)
    vmin, vmax = compute_color_scale_range(lead_df_india, scale_mode)
    render_map_legend(vmin, vmax, scale_mode)
    map_data = render_map(lead_df, lead_df_india, map_mode, vmin, vmax, india_geom, key=f"map_{cache_key}_{map_mode}_{scale_mode}")

    map_click = None
    if map_data:
        lc = map_data.get("last_clicked")
        if lc and "lat" in lc and "lng" in lc:
            if (bust_api.DOMAIN_LAT_MIN <= lc["lat"] <= bust_api.DOMAIN_LAT_MAX) and (bust_api.DOMAIN_LON_MIN <= lc["lng"] <= bust_api.DOMAIN_LON_MAX):
                map_click = {"lat": lc["lat"], "lon": lc["lng"]}

    st.write("")

    # --- Drill-down / XAI (India-only, so a click in a neighbouring country
    # still snaps to the nearest genuine India grid point) ---
    render_drill_down(artifact, lead_df_india, map_click)
    st.write("")

    # --- Lead-time decay (India-only) ---
    render_lead_time_decay(case_df_india, lead_day)

    st.markdown(
        '<p class="footer-note">NCMRWF Forecast Bust Detection &mdash; Prototype for Smart India Hackathon 2026. '
        "All data shown is synthetically generated for demonstration and is not an official MoES/NCMRWF product.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
