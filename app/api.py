"""
app/api.py
==============================================================================
AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts (Day 1-10)
Operational FastAPI service -- NCMRWF prototype

Wraps the trained model artifact (``models/bust_detector.pkl``) and the
explainability logic (``explainability.py``) in a small, fast HTTP API
suitable for a mapping dashboard / operational forecaster desk.

------------------------------------------------------------------------------
STARTUP CACHING STRATEGY
------------------------------------------------------------------------------
Recomputing SHAP attributions or re-reading the parquet dataset on every
request would make the API far too slow for an interactive map. Instead,
ONCE at process startup:

  1. The model artifact is loaded (model + feature schema + category
     vocabulary -- see ``explainability.py`` for the schema).
  2. The processed-features dataset is loaded and ONE forecast cycle is
     cached. With the real IMDAA dataset (10 days, 1-10 July 2019) only
     the EARLIEST cycle (issued 2019-07-01) can be verified out to Day 9 --
     later cycles run out of observation days (the 07-09 cycle only has
     Day 1). So by default the API serves the most recent cycle among those
     with the MOST lead days available. Override with BUST_INIT_TIME.
  3. ``bust_probability`` is predicted for every cached grid point / lead
     day in one batched call.
  4. SHAP is run ONCE, in a single batched call, over that entire cached
     grid, and reduced to a short "dominant risk factor" label per point.
  5. A KD-tree over the (fixed) grid coordinates is built for O(log n)
     nearest-point lookups from arbitrary (lat, lon) queries.

Every GET endpoint then just filters/aggregates this pre-computed, in-memory
DataFrame -- no model inference or SHAP computation happens on those
requests. Only ``POST /explain/point`` recomputes anything per-request, and
even then only a single-row SHAP call (cheap) via ``explainability.explain_bust``.

------------------------------------------------------------------------------
ENDPOINTS
------------------------------------------------------------------------------
  GET  /health                                    -- API / model status
  GET  /forecast/summary?lead_day={1..10}         -- 5-zone (N/S/E/W/Central
                                                      India) regional stats
  GET  /forecast/grid?lead_day={1..10}            -- GeoJSON (default) or
                                                      flat JSON point array
  POST /explain/point  {lat, lon, lead_day}       -- SHAP-based plain-English
                                                      narrative for the
                                                      nearest cached grid point

------------------------------------------------------------------------------
RUNNING
------------------------------------------------------------------------------
From the project root (the folder containing app/, data/, models/):

    uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

Then open http://127.0.0.1:8000/docs for interactive OpenAPI docs.

Environment overrides (useful in containers):
    BUST_DATA_PATH   -- path to the dataset (default data/processed_features_real.parquet)
    BUST_MODEL_PATH  -- path to bust_detector.pkl
    BUST_INIT_TIME   -- optional, e.g. 2019-07-05, to serve a specific forecast cycle

Dependencies: fastapi, uvicorn, scipy, numpy, pandas, pydantic, plus whatever
explainability.py/train_model.py need (xgboost/lightgbm, shap).
==============================================================================
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Make the sibling modules (explainability.py) importable regardless of the
# working directory uvicorn happens to be launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import explainability as xai  # noqa: E402  (import after sys.path setup, by design)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bust_detection.api")

# ===========================================================================
# CONFIGURATION
# ===========================================================================

DATA_PATH = Path(os.environ.get("BUST_DATA_PATH", str(PROJECT_ROOT / "data" / "processed_features_real.parquet")))
MODEL_PATH = Path(os.environ.get("BUST_MODEL_PATH", str(PROJECT_ROOT / "models" / "bust_detector.pkl")))
INIT_TIME_OVERRIDE = os.environ.get("BUST_INIT_TIME")  # optional: serve a specific forecast cycle

# IMDAA download box used by generate_data_and_features_real.py.
DOMAIN_LAT_MIN, DOMAIN_LAT_MAX = 6.0, 38.0
DOMAIN_LON_MIN, DOMAIN_LON_MAX = 68.0, 98.0

KM_PER_DEG = 111.32  # rough degrees->km conversion for the nearest-point distance
ABS_SHAP_FLOOR = 0.05  # below this |SHAP| (log-odds), a "dominant risk factor" is noise, not signal

# Short, dashboard-chip-friendly labels (a compact counterpart to the full
# sentence-level explanations in explainability.FEATURE_EXPLAINERS).
SHORT_RISK_LABELS: Dict[str, str] = {
    "pressure_gradient_hpa_per_100km": "Steep pressure gradient",
    "pressure_deficit": "Active low-pressure system",
    "MSLP": "Deep forecast low",
    "relative_vorticity_1e5_s": "Strong vorticity signal",
    "wind_shear_proxy_ms_per_100km": "Elevated wind shear",
    "spread_T2m": "Temperature ensemble spread",
    "spread_Rain": "Rainfall ensemble spread",
    "spread_wind": "Wind ensemble spread",
    "spread_composite": "Elevated ensemble spread",
    "spread_anomaly_vs_climatology": "Spread above climatological norm",
    "clim_hist_bust_rate": "Historically bust-prone area",
    "clim_hist_mean_error": "History of large verification errors",
    "clim_hist_mean_spread": "Historically wide ensemble spread",
    "clim_hist_std_spread": "Variable historical spread",
    "lead_day": "Extended lead time",
    "lead_day_norm": "Extended lead time",
    "predictability_decay_factor": "Extended lead time",
    "Rain_fcst": "Heavy forecast rainfall",
    "wind_speed10": "Strong forecast winds",
    "U10": "Strong zonal wind",
    "V10": "Strong meridional wind",
    "T2m": "High current temperature",
    # --- real IMDAA dataset features ---
    "U850": "Strong 850 hPa zonal flow",
    "V850": "Strong 850 hPa meridional flow",
    "wind_speed850": "Strong 850 hPa monsoon flow",
    "relative_vorticity_850_1e5_s": "850 hPa vorticity signal",
    "divergence_850_1e5_s": "850 hPa convergence",
    "wind_speed_gradient_850_per_100km": "Sharp 850 hPa wind gradient",
    "T2m_subgrid_std": "Patchy temperatures in cell",
    "T2m_recent_std": "Unsettled recent temperatures",
    "T2m_tendency_1d": "Sharp temperature swing",
    "Rain_subgrid_std": "Patchy convective rain",
    "Rain_recent_std": "Erratic recent rainfall",
    "Rain_tendency_1d": "Rapidly changing rainfall",
    "clim_hist_mean_abs_temp_error": "Recent large temperature misses",
    "clim_hist_mean_abs_rain_error": "Recent large rainfall misses",
    "T2m_anom_prev3": "Unusual temperature",
    "T2m_anom_to_date": "Unusual temperature",
    "Rain_anom_prev3": "Unusual rainfall",
    "Rain_wet_days_prev5": "On-off rain pattern",
    "Rain_nbr_max": "Heavy rain nearby",
    "Rain_nbr_mean": "Widespread rain nearby",
    "T2m_nbr_range": "Sharp temperature boundary",
    "T2m_gradient_per_100km": "Sharp temperature boundary",
    "idx_bob_vorticity": "Bay of Bengal circulation",
    "idx_monsoon_core_rain": "Monsoon activity",
    # --- S2S forecast dataset features ---
    "T850": "850 hPa temperature",
    "T925": "925 hPa temperature",
    "T500": "500 hPa temperature",
    "T850_fcst_change_1d": "Forecast temperature swing",
    "Rain_fcst_change_1d": "Shifting forecast rain band",
    "lapse_850_500": "Weak atmospheric stability",
    "shear_850_500": "Strong vertical wind shear",
    "Z500": "500 hPa height pattern",
    "lagged_spread_Rain": "Forecasts disagree on rain",
    "lagged_spread_T850": "Forecasts disagree on temperature",
    "lagged_spread_Z500": "Forecasts disagree on pattern",
    "lagged_n_members": "Forecast agreement",
    "orography_m": "Mountainous terrain",
    "land_frac": "Coastline effect",
    "lat": "Bust-prone location",
    "lon": "Bust-prone location",
}
NO_RISK_LABEL = "No significant risk factor"


def assign_macro_zone(
    lat: np.ndarray, lon: np.ndarray, lat_mid: float = 22.0, lon_mid: float = 83.0, central_radius_deg: float = 6.0
) -> np.ndarray:
    """Bucket grid points into 5 dashboard-style macro-zones (North / South /
    East / West / Central India). This is a coarser, purely geographic split
    -- distinct from the finer physiographic ``region`` column (Himalaya,
    Western Ghats, etc.) already in the dataset -- built around the domain
    centroid, with a Central band and N/S/E/W assigned by whichever axis a
    point deviates from that centroid more strongly."""
    dlat = lat - lat_mid
    dlon = lon - lon_mid
    dist = np.sqrt(dlat ** 2 + dlon ** 2)
    is_central = dist <= central_radius_deg
    is_ns_dominant = np.abs(dlat) >= np.abs(dlon)
    return np.where(
        is_central,
        "Central",
        np.where(is_ns_dominant, np.where(dlat > 0, "North", "South"), np.where(dlon > 0, "East", "West")),
    )


# ===========================================================================
# IN-MEMORY MODEL / PREDICTION STORE
# ===========================================================================


@dataclass
class ModelStore:
    artifact: dict
    grid_df: pd.DataFrame  # one row per (lat, lon, lead_day) for the latest forecast cycle, w/ cached predictions
    row_lookup: Dict[Tuple[float, float, int], int]  # (lat, lon, lead_day) -> row position in grid_df
    kdtree: cKDTree
    tree_coords: np.ndarray  # (N, 2) unique (lat, lon), same ordering as kdtree indices
    grid_resolution_deg: float
    forecast_init_time: str
    lead_days_available: List[int]
    loaded_at: str


def compute_dominant_risk_factor(artifact: dict, X: pd.DataFrame) -> List[str]:
    """Vectorised, single-pass reduction of a full SHAP matrix down to one
    short human-readable "dominant risk factor" label per row -- only among
    POSITIVE (bust-probability-increasing) non-categorical contributors,
    and only if that contribution clears an absolute materiality floor
    (mirrors the logic in ``explainability.explain_bust``, so the per-point
    cached label and the on-demand narrative never disagree in spirit)."""
    if len(X) == 0:
        return []
    shap_matrix, _ = xai.compute_shap_matrix(artifact, X)

    feature_cols = artifact["feature_cols"]
    categorical_cols = set(artifact["categorical_cols"])
    noncat_idx = [i for i, c in enumerate(feature_cols) if c not in categorical_cols]

    shap_sub = shap_matrix[:, noncat_idx]
    shap_sub_pos = np.where(shap_sub > 0, shap_sub, -np.inf)
    top_idx_local = np.argmax(shap_sub_pos, axis=1)
    n = shap_sub_pos.shape[0]
    top_val = shap_sub_pos[np.arange(n), top_idx_local]
    dominant_feature = [feature_cols[noncat_idx[i]] for i in top_idx_local]

    return [
        SHORT_RISK_LABELS.get(fn, fn.replace("_", " ").title()) if v > ABS_SHAP_FLOOR else NO_RISK_LABEL
        for fn, v in zip(dominant_feature, top_val)
    ]


def choose_serving_init_time(df: pd.DataFrame, override: Optional[str] = None) -> pd.Timestamp:
    """Pick which forecast cycle to serve. An explicit override wins;
    otherwise the most recent cycle among those with the MOST lead days
    (on the real 10-day dataset that is 2019-07-01, Day 1-9). Simply taking
    max(init_time) would serve a cycle with only Day 1 data."""
    inits = pd.to_datetime(df["init_time"])
    if override:
        wanted = pd.Timestamp(override)
        if wanted not in set(inits.unique()):
            raise ValueError(f"BUST_INIT_TIME={override} not in dataset. Available: {sorted(str(d.date()) for d in inits.unique())}")
        return wanted
    n_leads = df.groupby("init_time")["lead_day"].nunique()
    best = n_leads[n_leads == n_leads.max()]
    return pd.Timestamp(best.index.max())


def build_model_store(data_path: Path, model_path: Path) -> ModelStore:
    t0 = time.perf_counter()

    logger.info("Loading model artifact from %s ...", model_path)
    artifact = xai.load_artifact(str(model_path))
    logger.info("Loaded %s model with %d features.", artifact["model_type"], len(artifact["feature_cols"]))

    logger.info("Loading dataset from %s ...", data_path)
    df = pd.read_parquet(data_path)
    latest_init_time = choose_serving_init_time(df, INIT_TIME_OVERRIDE)
    grid_df = df[df["init_time"] == latest_init_time].reset_index(drop=True).copy()
    n_unique_points = grid_df[["lat", "lon"]].drop_duplicates().shape[0]
    logger.info(
        "Caching the latest forecast cycle: init_time=%s  (%d grid points x %d lead days = %d rows)",
        latest_init_time, n_unique_points, grid_df["lead_day"].nunique(), len(grid_df),
    )

    logger.info("Running batched model inference over the cached grid ...")
    X = xai.encode_dataframe(grid_df, artifact)
    bust_probability = artifact["model"].predict_proba(X)[:, 1]
    grid_df["bust_probability"] = bust_probability
    grid_df["confidence_pct"] = (1.0 - bust_probability) * 100.0

    logger.info("Running one batched SHAP pass over the cached grid to derive dominant risk factors ...")
    grid_df["dominant_risk_factor"] = compute_dominant_risk_factor(artifact, X)

    grid_df["macro_zone"] = assign_macro_zone(grid_df["lat"].to_numpy(dtype=float), grid_df["lon"].to_numpy(dtype=float))

    # Round coordinates once, consistently, so the KD-tree, the row_lookup
    # dict keys, and later query snapping all agree exactly on float values.
    grid_df["lat"] = grid_df["lat"].astype(float).round(5)
    grid_df["lon"] = grid_df["lon"].astype(float).round(5)

    unique_coords = grid_df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    tree_coords = unique_coords.to_numpy(dtype=float)
    kdtree = cKDTree(tree_coords)

    row_lookup = {
        (lat, lon, int(ld)): pos
        for pos, (lat, lon, ld) in enumerate(zip(grid_df["lat"], grid_df["lon"], grid_df["lead_day"]))
    }

    lat_sorted = np.sort(unique_coords["lat"].unique())
    grid_resolution_deg = float(np.min(np.diff(lat_sorted))) if len(lat_sorted) > 1 else 0.0
    lead_days_available = sorted(int(x) for x in grid_df["lead_day"].unique())

    logger.info(
        "Model store ready in %.1f s  (%d cached rows, %d unique grid points, %d lead days).",
        time.perf_counter() - t0, len(grid_df), len(tree_coords), len(lead_days_available),
    )

    return ModelStore(
        artifact=artifact,
        grid_df=grid_df,
        row_lookup=row_lookup,
        kdtree=kdtree,
        tree_coords=tree_coords,
        grid_resolution_deg=grid_resolution_deg,
        forecast_init_time=str(latest_init_time),
        lead_days_available=lead_days_available,
        loaded_at=datetime.now(timezone.utc).isoformat(),
    )


# ===========================================================================
# FASTAPI APP + LIFESPAN (load once at startup, reuse for every request)
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.store = build_model_store(DATA_PATH, MODEL_PATH)
    except FileNotFoundError as exc:
        logger.error(
            "Startup failed -- could not find a required file (%s). "
            "Run generate_data_and_features_real.py and train_model.py first, "
            "or set BUST_DATA_PATH / BUST_MODEL_PATH.", exc,
        )
        raise
    yield
    app.state.store = None


app = FastAPI(
    title="NCMRWF Forecast Bust Detection API",
    description="Operational serving layer for the AI-based medium-range (Day 1-10) forecast-bust detector.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)  # permissive for a hackathon demo dashboard; scope this down for real deployment


# ===========================================================================
# RESPONSE / REQUEST SCHEMAS
# ===========================================================================


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_loaded: bool
    forecast_init_time: str
    lead_days_available: List[int]
    n_grid_points: int
    grid_resolution_deg: float
    domain: Dict[str, float]
    loaded_at_utc: str
    server_time_utc: str


class RegionSummary(BaseModel):
    region: str
    n_grid_points: int
    avg_confidence_pct: float
    bust_risk_pct: float = Field(..., description="% of grid points in this zone classified as a likely bust at the model's operating threshold")
    avg_bust_probability: float
    primary_risk_driver: str


class ForecastSummaryResponse(BaseModel):
    lead_day: int
    forecast_init_time: str
    operating_threshold: float
    regions: List[RegionSummary]


class ExplainPointRequest(BaseModel):
    lat: float = Field(..., ge=DOMAIN_LAT_MIN, le=DOMAIN_LAT_MAX, description="Latitude, degrees N")
    lon: float = Field(..., ge=DOMAIN_LON_MIN, le=DOMAIN_LON_MAX, description="Longitude, degrees E")
    lead_day: int = Field(..., ge=1, le=10, description="Forecast lead day, 1-10")


def _dump_model(m: BaseModel) -> dict:
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None  # NaN (e.g. no verification history yet) is not valid JSON
        return round(float(value), 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# ===========================================================================
# ENDPOINTS
# ===========================================================================


@app.get("/", include_in_schema=False)
def root():
    return {"service": "NCMRWF Forecast Bust Detection API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    store: ModelStore = app.state.store
    if store is None:
        raise HTTPException(status_code=503, detail="Model store is not loaded yet.")
    return HealthResponse(
        status="ok",
        model_type=store.artifact["model_type"],
        model_loaded=True,
        forecast_init_time=store.forecast_init_time,
        lead_days_available=store.lead_days_available,
        n_grid_points=int(len(store.tree_coords)),
        grid_resolution_deg=store.grid_resolution_deg,
        domain={"lat_min": DOMAIN_LAT_MIN, "lat_max": DOMAIN_LAT_MAX, "lon_min": DOMAIN_LON_MIN, "lon_max": DOMAIN_LON_MAX},
        loaded_at_utc=store.loaded_at,
        server_time_utc=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/forecast/summary", response_model=ForecastSummaryResponse, tags=["forecast"])
def forecast_summary(
    lead_day: int = Query(..., ge=1, le=10, description="Forecast lead day, 1-10"),
) -> ForecastSummaryResponse:
    """Regional stats across North / South / East / West / Central India:
    average confidence, % of the zone's grid points flagged as a likely
    bust, and that zone's most common dominant risk driver."""
    store: ModelStore = app.state.store
    if lead_day not in store.lead_days_available:
        raise HTTPException(status_code=404, detail=f"lead_day={lead_day} not available. Available: {store.lead_days_available}")

    sub = store.grid_df[store.grid_df["lead_day"] == lead_day]
    threshold = float(store.artifact.get("operating_threshold", 0.5))

    regions: List[RegionSummary] = []
    for zone in ["North", "South", "East", "West", "Central"]:
        zdf = sub[sub["macro_zone"] == zone]
        if len(zdf) == 0:
            continue
        driver_mode = zdf["dominant_risk_factor"].mode()
        regions.append(
            RegionSummary(
                region=zone,
                n_grid_points=int(len(zdf)),
                avg_confidence_pct=round(float(zdf["confidence_pct"].mean()), 2),
                bust_risk_pct=round(float((zdf["bust_probability"] >= threshold).mean() * 100.0), 2),
                avg_bust_probability=round(float(zdf["bust_probability"].mean()), 4),
                primary_risk_driver=driver_mode.iloc[0] if len(driver_mode) else NO_RISK_LABEL,
            )
        )

    return ForecastSummaryResponse(
        lead_day=lead_day, forecast_init_time=store.forecast_init_time,
        operating_threshold=round(threshold, 3), regions=regions,
    )


@app.get("/forecast/grid", tags=["forecast"])
def forecast_grid(
    lead_day: int = Query(..., ge=1, le=10, description="Forecast lead day, 1-10"),
    format: str = Query("geojson", pattern="^(geojson|json)$", description="'geojson' (default, for map libraries) or 'json' (flat point array)"),
    max_confidence: Optional[float] = Query(None, ge=0, le=100, description="Only return points with confidence_score <= this value, e.g. 60 to see just the risky zones"),
    stride: int = Query(1, ge=1, le=20, description="Return every Nth cached grid point, for a lighter payload on slower connections"),
):
    """Per-grid-point [lat, lon, bust_probability, confidence_score,
    dominant_risk_factor] for mapping, as GeoJSON (default) or a flat JSON
    point array."""
    store: ModelStore = app.state.store
    if lead_day not in store.lead_days_available:
        raise HTTPException(status_code=404, detail=f"lead_day={lead_day} not available. Available: {store.lead_days_available}")

    sub = store.grid_df[store.grid_df["lead_day"] == lead_day]
    if max_confidence is not None:
        sub = sub[sub["confidence_pct"] <= max_confidence]
    if stride > 1:
        sub = sub.iloc[::stride]

    if format == "geojson":
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},  # GeoJSON order: [lon, lat]
                "properties": {
                    "bust_probability": round(float(r.bust_probability), 4),
                    "confidence_score": round(float(r.confidence_pct), 2),
                    "dominant_risk_factor": r.dominant_risk_factor,
                    "region": str(r.region),
                    "regime": str(r.regime),
                },
            }
            for r in sub.itertuples()
        ]
        return {
            "type": "FeatureCollection", "lead_day": lead_day, "forecast_init_time": store.forecast_init_time,
            "n_points": len(features), "features": features,
        }

    points = [
        {
            "lat": float(r.lat), "lon": float(r.lon),
            "bust_probability": round(float(r.bust_probability), 4),
            "confidence_score": round(float(r.confidence_pct), 2),
            "dominant_risk_factor": r.dominant_risk_factor,
        }
        for r in sub.itertuples()
    ]
    return {"lead_day": lead_day, "forecast_init_time": store.forecast_init_time, "n_points": len(points), "points": points}


@app.post("/explain/point", tags=["explain"])
def explain_point(req: ExplainPointRequest):
    """SHAP-based plain-English rationale for the nearest cached grid point
    to the requested (lat, lon) at the requested lead day -- e.g. why the
    model's confidence there is low."""
    store: ModelStore = app.state.store
    if req.lead_day not in store.lead_days_available:
        raise HTTPException(status_code=404, detail=f"lead_day={req.lead_day} not available. Available: {store.lead_days_available}")

    query_pt = np.array([[round(req.lat, 5), round(req.lon, 5)]])
    dist_deg, idx = store.kdtree.query(query_pt, k=1)
    matched_lat, matched_lon = (float(v) for v in store.tree_coords[int(idx[0])])
    distance_km = float(dist_deg[0]) * KM_PER_DEG

    key = (round(matched_lat, 5), round(matched_lon, 5), req.lead_day)
    row_pos = store.row_lookup.get(key)
    if row_pos is None:
        raise HTTPException(status_code=404, detail="No cached forecast for the nearest grid point at this lead day.")
    row = store.grid_df.iloc[row_pos]

    features = {c: row[c] for c in store.artifact["feature_cols"]}
    result = xai.explain_bust(features, artifact=store.artifact, top_k=2)

    return {
        "query": _dump_model(req),
        "matched_grid_point": {"lat": matched_lat, "lon": matched_lon, "distance_km": round(distance_km, 1)},
        "region": str(row["region"]),
        "regime": str(row["regime"]),
        "season": str(row["season"]),
        "bust_probability": round(result["bust_probability"], 4),
        "confidence_pct": result["confidence_pct"],
        "risk_level": result["risk_level"],
        "narrative": result["narrative"],
        "top_contributing_factors": [
            {"feature": c["feature"], "shap_value": round(c["shap_value"], 4), "value": _jsonable(c["value"])}
            for c in result["top_contributing_factors"]
        ],
    }


# ===========================================================================
# LOCAL DEV ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)