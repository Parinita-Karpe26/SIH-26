"""
generate_data_and_features.py
==============================================================================
AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts (Day 1-10)
Prototype data pipeline -- NCMRWF

This standalone module generates a physically-motivated SYNTHETIC gridded NWP
ensemble dataset over the Indian subcontinent, simulates ground-truth
observations (including deliberate "forecast bust" events caused by rapid
system intensification the model fails to capture), engineers a set of
verification-relevant predictor features, and exports a tidy, ML-ready
dataset to ``data/processed_features.parquet``.

------------------------------------------------------------------------------
PIPELINE STAGES
------------------------------------------------------------------------------
1. Grid & case setup
   - 0.5 deg lat/lon grid, 6.0N-38.0N x 68.0E-98.0E.
   - N synthetic forecast initializations spread across a full annual cycle,
     each initialization stochastically assigned a dominant synoptic regime
     (monsoon trough / western disturbance / Bay of Bengal cyclonic
     depression / quiescent) with season-appropriate probabilities.

2. Raw NWP + ensemble field simulation (``build_dataset``)
   - T2m, U10, V10, MSLP, 24h Rain_fcst for lead times Day 1 .. Day 10.
   - Ensemble spread (std-dev proxy) for T2m / Rain / wind, growing with lead
     time and with local synoptic activity (pressure deficit).
   NOTE: ensemble spread is simulated directly as the operational EPS "spread"
   product (as centres like NCMRWF/ECMWF distribute it) rather than by
   generating and differencing individual perturbed members -- this is a
   deliberate, documented simplification for a 48-hour prototype.

3. Dynamical feature engineering (``add_dynamical_features``)
   - Spatial gradient of MSLP  -> pressure-gradient-force proxy.
   - Relative vorticity from U10/V10 spatial gradients.
   - Horizontal wind-shear proxy (gradient of wind-speed magnitude); given
     the dataset is single-level (10 m), this stands in for true vertical
     shear and is documented as such.
   All computed fully vectorised via ``numpy.gradient`` over the entire
   (init_time, lead_day, lat, lon) array in a single call per axis.

4. Ground truth + forecast-bust simulation (``simulate_ground_truth_and_bust``)
   - Ambient random observation noise (scaled by ensemble spread) applied to
     ALL grid points/lead times.
   - PLUS a set of deliberately injected severe "bust" events representing
     rapid, model-missed intensification: forced rain error > 50 mm or
     temperature error > 5 degC. These are concentrated (via a risk score)
     at longer lead times and in the vicinity of active synoptic systems --
     not spread uniformly at random -- to mimic how real busts cluster.
   - ``is_bust`` (binary) and ``error_magnitude`` (continuous severity, >=1
     implies a bust) are derived AFTER errors are computed, directly from the
     stated physical thresholds.

5. Historical climatology + lead-time features
   (``build_climatology_lookup`` / ``add_climatology_features`` /
   ``add_lead_time_features`` / ``add_contextual_features``)
   - A synthetic "archived verification statistics" lookup table (regional /
     seasonal / lead-time dependent historical bust rate & mean error) is
     built INDEPENDENTLY of the current sample's simulated truth and merged
     in by (lat_bin, lon_bin, season, lead_day) -- exactly like joining a
     real climatological verification database. This avoids leaking the
     current instance's own ground-truth error into its own features.
   - A genuine "anomaly" feature is then computed from CURRENT ensemble
     spread vs. its historical climatological norm for that place/lead time.
   - Lead-time scaling features (normalised lead day, range bucket,
     exponential predictability-decay factor).

6. Export + summary (``export_dataset`` / ``print_summary``)
   - Writes ``data/processed_features.parquet``.
   - Prints class balance and top feature correlations with ``is_bust``.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python generate_data_and_features.py
    python generate_data_and_features.py --n-init-dates 60 --seed 7 \
        --output data/processed_features.parquet

Dependencies: numpy, pandas, xarray, pyarrow (for parquet export).
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bust_detection.datagen")

EARTH_M_PER_DEG_LAT = 111_320.0  # metres per degree latitude (~constant)

# Columns that require the simulated ground truth and are therefore ONLY
# available post-hoc (after verification) -- these are the modelling
# TARGETS / diagnostics and must never be used as classifier inputs.
TARGET_AND_DIAGNOSTIC_COLUMNS = [
    "Obs_T2m", "Obs_Rain", "temp_error", "rain_error", "error_magnitude", "is_bust",
]

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================


@dataclass
class GridConfig:
    """Spatial grid definition over the Indian subcontinent."""

    lat_min: float = 6.0
    lat_max: float = 38.0
    lon_min: float = 68.0
    lon_max: float = 98.0
    resolution: float = 0.5


@dataclass
class SimConfig:
    """Simulation / labelling / export configuration."""

    n_lead_days: int = 10
    n_ensemble_members: int = 21  # documented EPS size assumption (see module docstring)
    n_init_dates: int = 36
    random_seed: int = 42
    bust_fraction: float = 0.06  # target share of deliberately-injected severe busts
    output_path: str = "data/processed_features.parquet"


# ===========================================================================
# 2. GRID / REGIME HELPERS
# ===========================================================================


def build_grid(cfg: GridConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 1D lat/lon arrays and their 2D meshgrid (indexing='ij')."""
    lat = np.arange(cfg.lat_min, cfg.lat_max + 1e-6, cfg.resolution)
    lon = np.arange(cfg.lon_min, cfg.lon_max + 1e-6, cfg.resolution)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    return lat, lon, lat2d, lon2d


def build_init_dates(n_init_dates: int) -> List[pd.Timestamp]:
    """Spread N forecast initializations evenly across one full annual cycle
    so that all major synoptic regimes (monsoon, WD, post-monsoon cyclones)
    are represented in proportion to their real seasonal likelihood."""
    start = pd.Timestamp("2023-01-01")
    offsets = np.linspace(0, 364, n_init_dates).astype(int)
    return [start + pd.Timedelta(days=int(o)) for o in offsets]


def pick_regime(month: int, rng: np.random.Generator) -> Tuple[str, dict]:
    """Stochastically assign a dominant synoptic regime for a forecast case,
    with season-appropriate probabilities, plus regime-specific random
    parameters (e.g. cyclone track/intensity, WD starting longitude)."""
    if month in (6, 7, 8, 9):  # SW monsoon season
        regimes = ["monsoon_trough", "bay_of_bengal_cyclone", "quiescent"]
        weights = [0.65, 0.10, 0.25]
    elif month in (10, 11, 12):  # post-monsoon: peak BoB cyclone season
        regimes = ["bay_of_bengal_cyclone", "western_disturbance", "monsoon_trough", "quiescent"]
        weights = [0.45, 0.20, 0.05, 0.30]
    elif month in (1, 2, 3):  # winter: western disturbances dominate
        regimes = ["western_disturbance", "bay_of_bengal_cyclone", "quiescent"]
        weights = [0.55, 0.05, 0.40]
    else:  # 4, 5 -- pre-monsoon
        regimes = ["bay_of_bengal_cyclone", "western_disturbance", "monsoon_trough", "quiescent"]
        weights = [0.30, 0.10, 0.05, 0.55]

    regime = str(rng.choice(regimes, p=weights))
    params: dict = {}
    if regime == "monsoon_trough":
        params["phase0"] = rng.uniform(0.0, 360.0)
    elif regime == "western_disturbance":
        params["lon0"] = rng.uniform(60.0, 68.0)
    elif regime == "bay_of_bengal_cyclone":
        params["center0_lat"] = rng.uniform(10.0, 18.0)
        params["center0_lon"] = rng.uniform(85.0, 95.0)
        params["peak_day"] = rng.uniform(3.0, 6.0)
        params["base_intensity"] = rng.uniform(15.0, 45.0)  # hPa central deficit
    return regime, params


def month_to_season_str(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "pre_monsoon"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


def assign_season(months: np.ndarray) -> np.ndarray:
    """Vectorised IMD-convention season assignment from calendar month."""
    conditions = [
        np.isin(months, [12, 1, 2]),
        np.isin(months, [3, 4, 5]),
        np.isin(months, [6, 7, 8, 9]),
        np.isin(months, [10, 11]),
    ]
    choices = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
    return np.select(conditions, choices, default="unknown")


def assign_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Coarse physiographic region labelling for interpretability."""
    conditions = [
        lat > 30.0,
        (lat >= 24) & (lat <= 30) & (lon >= 68) & (lon < 75),
        (lat >= 24) & (lat <= 30) & (lon >= 75) & (lon < 88),
        (lat >= 26) & (lon >= 90),
        (lat >= 8) & (lat <= 21) & (lon >= 73) & (lon < 77),
        (lat >= 8) & (lat <= 22) & (lon > 82),
        (lat >= 8) & (lat <= 24) & (lon < 74),
    ]
    choices = [
        "Himalaya",
        "Thar_Desert",
        "Indo_Gangetic_Plain",
        "Northeast_India",
        "Western_Ghats",
        "BoB_Coast",
        "Arabian_Sea_Coast",
    ]
    return np.select(conditions, choices, default="Peninsular_Plateau")


# ===========================================================================
# 3. RAW METEOROLOGICAL FIELD SIMULATION
# ===========================================================================


def _base_climatology(lat2d: np.ndarray, lon2d: np.ndarray, day_of_year: int) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth background T2m / MSLP climatology (no active systems)."""
    season = np.cos(2 * np.pi * (day_of_year - 173) / 365.25)  # +1 midsummer, -1 midwinter
    lat_effect = -0.55 * (lat2d - 15.0)
    t2m = 27.0 + lat_effect + 8.0 * season

    himalaya_mask = lat2d > 30.0
    t2m = np.where(himalaya_mask, t2m - 12.0 * ((lat2d - 30.0) / 8.0), t2m)

    thar_mask = (lat2d >= 24) & (lat2d <= 30) & (lon2d >= 68) & (lon2d < 75)
    t2m = np.where(thar_mask & (season > 0), t2m + 4.0 * season, t2m)

    mslp = np.full_like(lat2d, 1010.0, dtype=np.float64)
    return t2m.astype(np.float64), mslp


def _monsoon_trough_perturbation(
    lat3d: np.ndarray, lon3d: np.ndarray, lead_r: np.ndarray, rng: np.random.Generator, phase0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Monsoon trough: an elongated NW-SE tilting low-pressure band across
    the Indo-Gangetic Plain, with the main rain band on its southern flank."""
    n_lead = lead_r.shape[0]
    noise = rng.normal(0, 0.3, size=(n_lead, 1, 1))
    phase = phase0 + 0.6 * lead_r + noise
    trough_lat = 26.0 - 0.06 * (lon3d - 68.0) + 1.5 * np.sin(np.radians(phase * 10.0))
    sigma_lat = 3.0
    deficit = 10.0 * np.exp(-((lat3d - trough_lat) ** 2) / (2 * sigma_lat ** 2))
    mslp_pert = -deficit

    rain_axis_lat = trough_lat - 2.0
    rain = 25.0 * np.exp(-((lat3d - rain_axis_lat) ** 2) / (2 * (sigma_lat * 1.2) ** 2))
    banding = 1.0 + 0.3 * np.sin(np.radians(lon3d * 3.0 + phase))
    rain = np.clip(rain * banding, 0, None)
    return mslp_pert, rain


def _western_disturbance_perturbation(
    lat3d: np.ndarray, lon3d: np.ndarray, lead_r: np.ndarray, rng: np.random.Generator, lon0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Western disturbance: a mid-latitude trough embedded in the westerlies,
    propagating eastward across northern India / the western Himalaya."""
    n_lead = lead_r.shape[0]
    noise = rng.normal(0, 0.4, size=(n_lead, 1, 1))
    trough_lon = lon0 + 6.0 * lead_r + noise  # ~6 deg/day eastward propagation
    sigma_lon = 6.0
    lat_band = lat3d > 28.0

    deficit_full = 8.0 * np.exp(-((lon3d - trough_lon) ** 2) / (2 * sigma_lon ** 2))
    mslp_pert = np.where(lat_band, -deficit_full, -deficit_full * 0.15)

    rain_full = 15.0 * np.exp(-((lon3d - trough_lon) ** 2) / (2 * (sigma_lon * 0.8) ** 2))
    rain = np.where(lat_band, rain_full, rain_full * 0.1)
    return mslp_pert, rain


def _bob_cyclone_perturbation(
    lat3d: np.ndarray,
    lon3d: np.ndarray,
    lead_r: np.ndarray,
    rng: np.random.Generator,
    center0_lat: float,
    center0_lon: float,
    peak_day: float,
    base_intensity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bay of Bengal cyclonic depression: Gaussian pressure deficit + Rankine-
    like tangential wind field, tracking WNW and intensifying/decaying with a
    Gaussian lifecycle peaking near ``peak_day``, weakening sharply after
    (simulated) landfall west of 82E."""
    n_lead = lead_r.shape[0]
    jitter_lat = rng.normal(0, 0.05, size=(n_lead, 1, 1))
    jitter_lon = rng.normal(0, 0.05, size=(n_lead, 1, 1))
    center_lat = center0_lat + 0.15 * lead_r + jitter_lat
    center_lon = center0_lon - 0.35 * lead_r + jitter_lon

    intensity = base_intensity * np.exp(-((lead_r - peak_day) ** 2) / (2 * 2.0 ** 2))
    landfall_factor = np.where(center_lon < 82.0, 0.4, 1.0)
    intensity = intensity * landfall_factor

    dy = (lat3d - center_lat) * EARTH_M_PER_DEG_LAT / 1000.0  # km
    dx = (lon3d - center_lon) * EARTH_M_PER_DEG_LAT / 1000.0 * np.cos(np.radians(center_lat))
    r = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 1.0)  # km, radius from system centre

    deficit = intensity * np.exp(-r / 150.0)
    mslp_pert = -deficit

    r_max = 50.0  # km, radius of maximum wind
    v_tan = np.where(r < r_max, intensity * 0.9 * (r / r_max), intensity * 0.9 * (r_max / r))
    theta = np.arctan2(dy, dx)
    u_cyc = -v_tan * np.sin(theta)  # cyclonic (counter-clockwise, N. Hemisphere)
    v_cyc = v_tan * np.cos(theta)

    rain = intensity * 1.2 * np.exp(-r / 80.0)  # heaviest rain near the eyewall
    return mslp_pert, rain, u_cyc, v_cyc


def simulate_case(
    init_date: pd.Timestamp,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    lead_days: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Simulate one full forecast case (all lead days, full grid) for a
    single initialization date, returning raw NWP + ensemble-spread fields."""
    n_lead, nlat, nlon = len(lead_days), lat2d.shape[0], lat2d.shape[1]
    lat3d = lat2d[None, :, :]
    lon3d = lon2d[None, :, :]
    lead_r = lead_days.reshape(-1, 1, 1).astype(np.float64)

    t2m_clim0, mslp_clim0 = _base_climatology(lat2d, lon2d, int(init_date.dayofyear))
    t2m_clim0 = t2m_clim0[None, :, :]
    mslp_clim0 = mslp_clim0[None, :, :]

    regime, params = pick_regime(int(init_date.month), rng)

    mslp_pert = np.zeros((n_lead, nlat, nlon))
    rain = 1.5 * rng.random((n_lead, nlat, nlon))  # ambient mesoscale/background rain
    u_extra = np.zeros((n_lead, nlat, nlon))
    v_extra = np.zeros((n_lead, nlat, nlon))

    if regime == "monsoon_trough":
        mslp_pert, rain_regime = _monsoon_trough_perturbation(lat3d, lon3d, lead_r, rng, params["phase0"])
        rain = rain + rain_regime
        u_extra = u_extra + 5.0  # background SW monsoon flow
        v_extra = v_extra + 4.0
    elif regime == "western_disturbance":
        mslp_pert, rain_regime = _western_disturbance_perturbation(lat3d, lon3d, lead_r, rng, params["lon0"])
        rain = rain + rain_regime
        lat_band = lat3d > 28.0
        u_extra = np.where(lat_band, u_extra + 10.0, u_extra + 3.0)  # strong sfc westerlies under WD
    elif regime == "bay_of_bengal_cyclone":
        mslp_pert, rain_regime, u_cyc, v_cyc = _bob_cyclone_perturbation(
            lat3d,
            lon3d,
            lead_r,
            rng,
            params["center0_lat"],
            params["center0_lon"],
            params["peak_day"],
            params["base_intensity"],
        )
        rain = rain + rain_regime
        u_extra = u_extra + u_cyc
        v_extra = v_extra + v_cyc
    # else: "quiescent" -- background fields only

    mslp = mslp_clim0 + mslp_pert + rng.normal(0, 0.4, size=(n_lead, nlat, nlon))
    pressure_deficit = np.clip(mslp_clim0 - mslp, 0, None)  # proxy for local system intensity

    t_anom = np.clip(-0.04 * rain, -6.0, 0.5)  # cloud/rain cooling effect
    t2m = t2m_clim0 + t_anom + rng.normal(0, 0.6, size=(n_lead, nlat, nlon))

    u10 = u_extra + rng.normal(0, 1.0, size=(n_lead, nlat, nlon))
    v10 = v_extra + rng.normal(0, 1.0, size=(n_lead, nlat, nlon))

    rain_fcst = np.clip(rain + rng.normal(0, 1.0, size=(n_lead, nlat, nlon)), 0, None)

    # Ensemble spread grows with lead time and with local synoptic activity.
    spread_t2m = np.clip(
        0.3 + 0.12 * lead_r + 0.015 * pressure_deficit + rng.normal(0, 0.05, size=(n_lead, nlat, nlon)), 0.1, None
    )
    spread_rain = np.clip(
        1.0 + 0.8 * lead_r + 0.25 * pressure_deficit + rng.normal(0, 0.3, size=(n_lead, nlat, nlon)), 0.2, None
    )
    spread_wind = np.clip(
        0.4 + 0.15 * lead_r + 0.01 * pressure_deficit + rng.normal(0, 0.05, size=(n_lead, nlat, nlon)), 0.1, None
    )

    return {
        "regime": regime,
        "T2m": t2m.astype(np.float32),
        "U10": u10.astype(np.float32),
        "V10": v10.astype(np.float32),
        "MSLP": mslp.astype(np.float32),
        "Rain_fcst": rain_fcst.astype(np.float32),
        "pressure_deficit": pressure_deficit.astype(np.float32),
        "spread_T2m": spread_t2m.astype(np.float32),
        "spread_Rain": spread_rain.astype(np.float32),
        "spread_wind": spread_wind.astype(np.float32),
    }


def build_dataset(grid_cfg: GridConfig, sim_cfg: SimConfig) -> xr.Dataset:
    """Simulate every forecast case and assemble a single 4-D xarray Dataset
    with dims (init_time, lead_day, lat, lon)."""
    lat, lon, lat2d, lon2d = build_grid(grid_cfg)
    lead_days = np.arange(1, sim_cfg.n_lead_days + 1)
    init_dates = build_init_dates(sim_cfg.n_init_dates)

    var_names = [
        "T2m", "U10", "V10", "MSLP", "Rain_fcst",
        "pressure_deficit", "spread_T2m", "spread_Rain", "spread_wind",
    ]
    collected: Dict[str, List[np.ndarray]] = {v: [] for v in var_names}
    regimes: List[str] = []

    for i, init_date in enumerate(init_dates):
        case_rng = np.random.default_rng(sim_cfg.random_seed + 1000 + i)
        case_out = simulate_case(init_date, lat2d, lon2d, lead_days, case_rng)
        regimes.append(case_out["regime"])
        for v in var_names:
            collected[v].append(case_out[v])
        if (i + 1) % 10 == 0 or i == len(init_dates) - 1:
            logger.info("  simulated forecast case %d/%d", i + 1, len(init_dates))

    data_vars = {
        v: (("init_time", "lead_day", "lat", "lon"), np.stack(collected[v], axis=0)) for v in var_names
    }
    data_vars["regime"] = (("init_time",), np.array(regimes, dtype=object))

    coords = {
        "init_time": np.array(init_dates, dtype="datetime64[ns]"),
        "lead_day": lead_days,
        "lat": lat,
        "lon": lon,
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["description"] = (
        "Synthetic NWP + ensemble-spread dataset for AI-based forecast bust "
        "detection (NCMRWF prototype). All fields are simulated, not observed."
    )
    return ds


# ===========================================================================
# 4. DYNAMICAL FEATURE ENGINEERING (vectorised over the full 4-D dataset)
# ===========================================================================


def add_dynamical_features(ds: xr.Dataset) -> xr.Dataset:
    """Add pressure-gradient, vorticity and wind-shear-proxy features,
    computed via centred finite differences (``numpy.gradient``) over the
    entire (init_time, lead_day, lat, lon) array in one vectorised call per
    axis -- no explicit Python loop over cases or lead times is needed."""
    lat = ds["lat"].values
    lat_res = float(np.mean(np.diff(lat)))
    lon_res = float(np.mean(np.diff(ds["lon"].values)))
    m_per_deg_lon = EARTH_M_PER_DEG_LAT * np.cos(np.radians(lat))  # shape (nlat,)

    def d_dlat(field: np.ndarray) -> np.ndarray:
        # dims: (init_time, lead_day, lat, lon) -> lat is axis 2
        return np.gradient(field, lat_res, axis=2, edge_order=2) / EARTH_M_PER_DEG_LAT

    def d_dlon(field: np.ndarray) -> np.ndarray:
        d_deg = np.gradient(field, lon_res, axis=3, edge_order=2)
        scale = m_per_deg_lon.reshape(1, 1, -1, 1)
        return d_deg / scale

    mslp = ds["MSLP"].values
    grad_mag_hpa_per_m = np.sqrt(d_dlat(mslp) ** 2 + d_dlon(mslp) ** 2)
    pressure_gradient_hpa_per_100km = grad_mag_hpa_per_m * 100_000.0  # hPa / m -> hPa / 100km

    u = ds["U10"].values
    v = ds["V10"].values
    relative_vorticity = d_dlon(v) - d_dlat(u)  # zeta = dv/dx - du/dy  [s^-1]
    relative_vorticity_1e5_s = relative_vorticity * 1.0e5

    wind_speed10 = np.sqrt(u ** 2 + v ** 2)
    wind_shear_proxy = np.sqrt(d_dlon(wind_speed10) ** 2 + d_dlat(wind_speed10) ** 2) * 100_000.0

    dims = ("init_time", "lead_day", "lat", "lon")
    ds["pressure_gradient_hpa_per_100km"] = (dims, pressure_gradient_hpa_per_100km.astype(np.float32))
    ds["relative_vorticity_1e5_s"] = (dims, relative_vorticity_1e5_s.astype(np.float32))
    ds["wind_shear_proxy_ms_per_100km"] = (dims, wind_shear_proxy.astype(np.float32))
    ds["wind_speed10"] = (dims, wind_speed10.astype(np.float32))
    return ds


# ===========================================================================
# 5. GROUND TRUTH + FORECAST BUST SIMULATION
# ===========================================================================


def simulate_ground_truth_and_bust(ds: xr.Dataset, sim_cfg: SimConfig, rng: np.random.Generator) -> xr.Dataset:
    """Simulate observed truth and derive the bust labels.

    Two error sources are combined, exactly as required:
      (a) ambient random perturbation, scaled by ensemble spread, applied
          everywhere;
      (b) deliberately injected SEVERE bust events (rain error > 50 mm or
          temperature error > 5 degC) representing rapid system
          intensification the deterministic/ensemble forecast fails to
          capture. These are NOT placed uniformly at random: a risk score
          (higher at longer lead times, near active systems, and where
          ensemble spread is already elevated) determines where they occur,
          then the top ``bust_fraction`` of risk scores are forced into a
          severe miss.

    ``is_bust`` / ``error_magnitude`` are derived strictly from the resulting
    errors against the stated physical thresholds -- they are outcomes of
    the simulation, not independently chosen.
    """
    t2m = ds["T2m"].values
    rain_fcst = ds["Rain_fcst"].values
    spread_t2m = ds["spread_T2m"].values
    spread_rain = ds["spread_Rain"].values
    pressure_deficit = ds["pressure_deficit"].values
    lead_day = ds["lead_day"].values.reshape(1, -1, 1, 1).astype(np.float64)

    shape = t2m.shape
    spread_composite = (spread_t2m + spread_rain / 10.0) / 2.0

    deficit_norm = np.clip(pressure_deficit / 40.0, 0, 1.5)
    spread_norm = np.clip(spread_composite / 1.5, 0, 1.5)
    risk_noise = rng.random(shape)
    # "Where" a deliberate bust lands is driven by local synoptic activity
    # (pressure deficit / already-elevated ensemble spread) + randomness.
    # "When" (how often) is handled separately below via a lead-time-scaled
    # target fraction, so the two effects don't compound into an unrealistic
    # all-or-nothing cliff at the longest lead times.
    risk_score = 0.55 * deficit_norm + 0.30 * spread_norm + 0.15 * risk_noise

    n_lead = shape[1]
    lead_day_values = ds["lead_day"].values
    # Bust likelihood grows gradually with lead time (skill decay) rather than
    # jumping from ~0 to ~50%: roughly 0.4x the average rate at Day 1 up to
    # ~1.6x at Day 10, scaled so the overall dataset still averages
    # ``sim_cfg.bust_fraction``.
    lead_scale = 0.4 + 0.12 * (lead_day_values - 1)  # Day1:0.4 .. Day10:1.48
    lead_scale = lead_scale / lead_scale.mean()  # renormalise to preserve overall target rate
    target_frac_per_lead = np.clip(sim_cfg.bust_fraction * lead_scale, 0.002, 0.5)

    deliberate_bust_mask = np.zeros(shape, dtype=bool)
    for li in range(n_lead):
        slice_scores = risk_score[:, li, :, :]
        thresh = np.quantile(slice_scores, 1.0 - target_frac_per_lead[li])
        deliberate_bust_mask[:, li, :, :] = slice_scores >= thresh

    # (a) ambient noise, applied everywhere
    ambient_temp_noise = rng.normal(0, 1.0, shape) * (0.5 + spread_t2m * 0.4)
    ambient_rain_noise = rng.normal(0, 1.0, shape) * (2.0 + spread_rain * 0.5)
    obs_t2m = t2m + ambient_temp_noise
    obs_rain = np.clip(rain_fcst + ambient_rain_noise, 0, None)

    # (b) deliberate severe bust injection
    rain_bust_choice = rng.random(shape) < 0.55
    rain_bust_mask = deliberate_bust_mask & rain_bust_choice
    temp_bust_mask = deliberate_bust_mask & (~rain_bust_choice)

    rain_extra = rng.uniform(55.0, 190.0, shape)
    temp_sign = np.where(rng.random(shape) < 0.5, 1.0, -1.0)
    temp_extra = rng.uniform(5.5, 13.0, shape) * temp_sign

    obs_rain = np.where(rain_bust_mask, obs_rain + rain_extra, obs_rain)
    obs_t2m = np.where(temp_bust_mask, obs_t2m + temp_extra, obs_t2m)

    rain_error = obs_rain - rain_fcst
    temp_error = obs_t2m - t2m

    is_bust = ((np.abs(rain_error) > 50.0) | (np.abs(temp_error) > 5.0)).astype(np.int8)
    error_magnitude = np.maximum(np.abs(rain_error) / 50.0, np.abs(temp_error) / 5.0).astype(np.float32)

    dims = ("init_time", "lead_day", "lat", "lon")
    ds["spread_composite"] = (dims, spread_composite.astype(np.float32))
    ds["Obs_T2m"] = (dims, obs_t2m.astype(np.float32))
    ds["Obs_Rain"] = (dims, obs_rain.astype(np.float32))
    ds["temp_error"] = (dims, temp_error.astype(np.float32))
    ds["rain_error"] = (dims, rain_error.astype(np.float32))
    ds["error_magnitude"] = (dims, error_magnitude)
    ds["is_bust"] = (dims, is_bust)
    return ds


# ===========================================================================
# 6. FLATTEN + LEAD-TIME / CONTEXTUAL / CLIMATOLOGY FEATURES
# ===========================================================================


def dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """Flatten the 4-D xarray Dataset into a tidy pandas DataFrame."""
    return ds.to_dataframe().reset_index()


def add_lead_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["lead_day_norm"] = df["lead_day"] / 10.0
    bins = [0, 3, 7, 10]
    labels = ["short_range_D1_D3", "medium_range_D4_D7", "extended_range_D8_D10"]
    df["lead_day_bucket"] = pd.cut(df["lead_day"], bins=bins, labels=labels, include_lowest=True)
    df["predictability_decay_factor"] = 1.0 - np.exp(-df["lead_day"] / 4.0)
    return df


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    df["region"] = assign_region(df["lat"].values, df["lon"].values)
    df["season"] = assign_season(df["init_time"].dt.month.values)
    df["valid_time"] = df["init_time"] + pd.to_timedelta(df["lead_day"], unit="D")
    return df


def _regional_difficulty_index(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Smooth, purely spatial index of historically-difficult-to-forecast
    zones (orography, coastal cyclone genesis regions, monsoon-trough belt).
    Used ONLY to build the independent historical climatology table below --
    never a function of any individual simulated sample's truth/error."""
    himalaya = np.exp(-((lat - 30.0) ** 2) / (2 * 4.0 ** 2))
    western_ghats = np.exp(-(((lat - 14.0) ** 2) + ((lon - 75.0) ** 2)) / (2 * 3.0 ** 2))
    bob_coast = np.exp(-(((lat - 16.0) ** 2) + ((lon - 90.0) ** 2)) / (2 * 5.0 ** 2))
    igp_trough = np.exp(-((lat - 25.0) ** 2) / (2 * 3.0 ** 2))
    return 0.9 * himalaya + 0.8 * western_ghats + 1.0 * bob_coast + 0.7 * igp_trough


def build_climatology_lookup(
    grid_cfg: GridConfig,
    lead_days: List[int],
    seed: int = 999,
    lat_bin_size: float = 2.0,
    lon_bin_size: float = 2.0,
) -> pd.DataFrame:
    """Build a synthetic 'historical model verification archive' lookup
    table: for each (lat_bin, lon_bin, season, lead_day), the model's
    long-run historical bust rate / mean absolute error / typical ensemble
    spread. Seeded independently of the live simulation -- this represents a
    FIXED reference dataset (as a real centre's archived verification stats
    would be), not something derived from the current run."""
    rng_hist = np.random.default_rng(seed)
    lat_edges = np.arange(grid_cfg.lat_min, grid_cfg.lat_max + lat_bin_size, lat_bin_size)
    lon_edges = np.arange(grid_cfg.lon_min, grid_cfg.lon_max + lon_bin_size, lon_bin_size)
    seasons = ["winter", "pre_monsoon", "monsoon", "post_monsoon"]
    season_mult = {"monsoon": 1.2, "post_monsoon": 1.3, "winter": 1.0, "pre_monsoon": 0.9}

    records = []
    for lb in lat_edges:
        lat_c = round(float(lb) + lat_bin_size / 2.0, 6)
        for lo in lon_edges:
            lon_c = round(float(lo) + lon_bin_size / 2.0, 6)
            idx = float(_regional_difficulty_index(np.array([lat_c]), np.array([lon_c]))[0])
            for season in seasons:
                sm = season_mult[season]
                for ld in lead_days:
                    lead_infl = 1.0 + 0.08 * (ld - 1)
                    hist_bust_rate = float(
                        np.clip(0.02 + 0.05 * idx * sm * lead_infl + rng_hist.normal(0, 0.004), 0.005, 0.45)
                    )
                    hist_mean_error = float(
                        np.clip(3.0 + 8.0 * idx * sm * lead_infl + rng_hist.normal(0, 0.4), 0.5, 40.0)
                    )
                    # Calibrated to the same scale as the live `spread_composite`
                    # field (~0.3 at Day 1 growing to ~1.2-1.5 at Day 10), with a
                    # modest regional/seasonal modulation on top -- so the
                    # resulting anomaly feature is centred near zero rather than
                    # permanently skewed by a scale mismatch.
                    hist_mean_spread = float(
                        np.clip(0.25 + 0.10 * (ld - 1) + 0.15 * idx * sm + rng_hist.normal(0, 0.03), 0.1, 3.0)
                    )
                    hist_std_spread = float(
                        np.clip(0.08 + 0.06 * idx * sm + abs(rng_hist.normal(0, 0.02)), 0.03, 1.0)
                    )
                    records.append(
                        dict(
                            lat_bin=lat_c,
                            lon_bin=lon_c,
                            season=season,
                            lead_day=int(ld),
                            clim_hist_bust_rate=hist_bust_rate,
                            clim_hist_mean_error=hist_mean_error,
                            clim_hist_mean_spread=hist_mean_spread,
                            clim_hist_std_spread=hist_std_spread,
                        )
                    )
    return pd.DataFrame.from_records(records)


def add_climatology_features(
    df: pd.DataFrame, grid_cfg: GridConfig, lat_bin_size: float = 2.0, lon_bin_size: float = 2.0
) -> pd.DataFrame:
    """Join the historical climatology lookup and derive a genuine anomaly
    feature (current ensemble-spread vs. its historical norm for that
    place/lead-time) -- computed only from forecast-side quantities, never
    from the current sample's own simulated truth/error."""
    lead_days_present = sorted(int(x) for x in df["lead_day"].unique())
    lookup = build_climatology_lookup(grid_cfg, lead_days_present, lat_bin_size=lat_bin_size, lon_bin_size=lon_bin_size)

    df["lat_bin"] = np.round(
        np.floor((df["lat"] - grid_cfg.lat_min) / lat_bin_size) * lat_bin_size + grid_cfg.lat_min + lat_bin_size / 2.0,
        6,
    )
    df["lon_bin"] = np.round(
        np.floor((df["lon"] - grid_cfg.lon_min) / lon_bin_size) * lon_bin_size + grid_cfg.lon_min + lon_bin_size / 2.0,
        6,
    )

    merged = df.merge(lookup, on=["lat_bin", "lon_bin", "season", "lead_day"], how="left")
    n_missing = merged["clim_hist_mean_spread"].isna().sum()
    if n_missing:
        logger.warning("Climatology lookup join left %d rows unmatched -- filling with grid medians.", n_missing)
        for c in ["clim_hist_bust_rate", "clim_hist_mean_error", "clim_hist_mean_spread", "clim_hist_std_spread"]:
            merged[c] = merged[c].fillna(merged[c].median())

    merged["spread_anomaly_vs_climatology"] = (
        merged["spread_composite"] - merged["clim_hist_mean_spread"]
    ) / merged["clim_hist_std_spread"]

    return merged.drop(columns=["lat_bin", "lon_bin"])


# ===========================================================================
# 7. DTYPE OPTIMISATION / EXPORT / SUMMARY
# ===========================================================================


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["regime", "region", "season", "lead_day_bucket"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)
    if "lead_day" in df.columns:
        df["lead_day"] = df["lead_day"].astype(np.int8)
    if "is_bust" in df.columns:
        df["is_bust"] = df["is_bust"].astype(np.int8)
    return df


def export_dataset(df: pd.DataFrame, output_path: str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Writing parquet requires 'pyarrow' (or 'fastparquet'). "
            "Install with: pip install pyarrow --break-system-packages"
        ) from exc
    logger.info("Dataset exported to %s (%.2f MB)", out, out.stat().st_size / 1e6)
    return out


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 74)
    print("FORECAST BUST DETECTION -- SYNTHETIC DATASET SUMMARY")
    print("=" * 74)
    print(f"Total records        : {len(df):,}")
    print(f"Grid points          : {df[['lat', 'lon']].drop_duplicates().shape[0]:,}")
    print(f"Lead times           : Day {int(df['lead_day'].min())} to Day {int(df['lead_day'].max())}")
    print(f"Forecast init dates  : {df['init_time'].nunique()}")
    print(f"Columns              : {df.shape[1]}")
    print("-" * 74)

    counts = df["is_bust"].value_counts().sort_index()
    pct = df["is_bust"].value_counts(normalize=True).sort_index() * 100
    print("Class balance (is_bust):")
    for k in counts.index:
        label = "BUST" if k == 1 else "RELIABLE"
        print(f"  {k} ({label:9s}): {counts[k]:>10,}  ({pct[k]:5.2f}%)")
    print("-" * 74)

    print("Synoptic regime distribution (per forecast case):")
    print(df["regime"].value_counts().to_string())
    print("-" * 74)

    numeric_df = df.select_dtypes(include=[np.number])
    if "is_bust" in numeric_df.columns:
        corr = numeric_df.corr(numeric_only=True)["is_bust"].drop(labels=["is_bust"], errors="ignore")
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
        print("Top feature correlations with is_bust:")
        print(corr.head(15).to_string(float_format=lambda x: f"{x:+.4f}"))
    print("-" * 74)

    predictor_cols = [c for c in df.columns if c not in TARGET_AND_DIAGNOSTIC_COLUMNS]
    print(f"Modelling target      : is_bust (binary)  |  error_magnitude (continuous, >=1 implies bust)")
    print(f"EXCLUDE from features : {TARGET_AND_DIAGNOSTIC_COLUMNS}")
    print(f"  (these require the simulated ground truth and are only known post-hoc verification)")
    print(f"Safe predictor columns ({len(predictor_cols)}): {predictor_cols}")
    print("=" * 74)


# ===========================================================================
# 8. PIPELINE ORCHESTRATION
# ===========================================================================


def run_pipeline(grid_cfg: GridConfig, sim_cfg: SimConfig) -> pd.DataFrame:
    t0 = time.perf_counter()

    logger.info(
        "Building synthetic NWP ensemble dataset (%d init dates x %d lead days over a %.1f deg grid)...",
        sim_cfg.n_init_dates, sim_cfg.n_lead_days, grid_cfg.resolution,
    )
    ds = build_dataset(grid_cfg, sim_cfg)

    logger.info("Computing dynamical diagnostic features (pressure gradient, vorticity, wind shear)...")
    ds = add_dynamical_features(ds)

    logger.info("Simulating ground-truth observations and injecting deliberate forecast-bust events...")
    rng = np.random.default_rng(sim_cfg.random_seed + 7)
    ds = simulate_ground_truth_and_bust(ds, sim_cfg, rng)

    logger.info("Flattening gridded dataset into a tidy ML-ready dataframe...")
    df = dataset_to_dataframe(ds)
    df = add_lead_time_features(df)
    df = add_contextual_features(df)

    logger.info("Joining historical climatological error/bust-rate lookup table...")
    df = add_climatology_features(df, grid_cfg)

    df = _optimize_dtypes(df)

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline complete in %.1f s. Final dataset shape: %s", elapsed, df.shape)
    return df


# ===========================================================================
# 9. CLI ENTRY POINT
# ===========================================================================


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic NWP ensemble dataset + engineered features "
        "for AI-based medium-range forecast bust detection (NCMRWF prototype)."
    )
    p.add_argument("--n-init-dates", type=int, default=36, help="Number of forecast initializations spread across a year (default: 36)")
    p.add_argument("--n-lead-days", type=int, default=10, help="Number of lead days, Day 1..N (default: 10)")
    p.add_argument("--resolution", type=float, default=0.5, help="Grid resolution in degrees (default: 0.5)")
    p.add_argument("--bust-fraction", type=float, default=0.06, help="Target fraction of deliberately-injected severe busts (default: 0.06)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument("--output", type=str, default="data/processed_features.parquet", help="Output parquet path")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    grid_cfg = GridConfig(resolution=args.resolution)
    sim_cfg = SimConfig(
        n_lead_days=args.n_lead_days,
        n_init_dates=args.n_init_dates,
        bust_fraction=args.bust_fraction,
        random_seed=args.seed,
        output_path=args.output,
    )
    df = run_pipeline(grid_cfg, sim_cfg)
    export_dataset(df, sim_cfg.output_path)
    print_summary(df)


if __name__ == "__main__":
    main()
