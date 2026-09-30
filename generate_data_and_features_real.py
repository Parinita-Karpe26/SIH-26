"""
generate_data_and_features_real.py
==============================================================================
REAL-DATA pipeline for BustGuard AI (SIH 2026, PS 26079) -- built on the
NCMRWF IMDAA reanalysis NetCDF files supplied with the problem statement.
Replaces BOTH the old synthetic generator and the Open-Meteo prototype: no
network access, no synthetic numbers, no dependency on
generate_data_and_features.py.

WHAT THE SUPPLIED FILES ACTUALLY ARE (checked from each file's CDO history)
------------------------------------------------------------------------------
  Every file is IMDAA *reanalysis* (NCMRWF / UK Met Office Indian Monsoon
  Data Assimilation and Analysis), cut to 68-98E, 6-38N, ~0.12 deg grid,
  2019-07-01 .. 2019-07-10:

    T2m_max_YYYYMMDD.nc                         daily max 2 m temperature (K)
    APCP-sfc_YYYYMMDD.nc                        daily accumulated rain (mm)
    UGRD-850mb_YYYYMMDDHH_..._850_hpa.nc        850 hPa u-wind, 00Z + 12Z
    VGRD-850mb_YYYYMMDDHH_..._850_hpa.nc        850 hPa v-wind, 00Z + 12Z

  Reanalysis is the best available estimate of what ACTUALLY HAPPENED. The
  supplied files contain NO forecast fields at any lead time.

HOW A "FORECAST" AND A "BUST" ARE DEFINED WITHOUT FORECAST FILES
------------------------------------------------------------------------------
  Forecast  = PERSISTENCE: the forecast issued on day I for valid day
              V = I + lead_day is "whatever IMDAA shows on day I". Persistence
              is the standard reference forecast in verification (WMO/WWRP);
              it is a real forecast built from real data, not synthetic noise.
  Truth     = IMDAA on valid day V.
  Bust      = |T2m_max error| > 5 C  OR  |24 h rain error| > 50 mm
              (same thresholds as before).
  So the model learns to flag, at issue time, where/when a persistence
  forecast is about to fail badly. When real NWP forecast files (e.g. NCUM /
  NEPS Day 1-10 output) become available, only build_pairs() needs to swap
  the persistence forecast for the NWP one -- every feature, label and
  downstream script stays the same.

LEAKAGE RULES (every predictor is known at issue time I)
------------------------------------------------------------------------------
  * All weather features come from day I or earlier -- never from day V.
  * clim_hist_* is an EXPANDING, PAST-ONLY verification history: for a row
    issued on day I it only uses (init, lead) pairs whose valid day <= I,
    i.e. outcomes a forecaster would already have verified by then.
  * init_time AND valid_time are both exported so train_model.py can split
    on valid_time (see that file) -- that stops a training label and a test
    label from being the same day's observation.

WHAT IS NOT IN THESE FILES (so NOT in the dataset -- nothing is invented)
------------------------------------------------------------------------------
  * No MSLP / surface pressure -> the old MSLP, pressure_deficit and
    pressure_gradient features are gone. 850 hPa vorticity/divergence now
    carry the synoptic-dynamics signal.
  * No ensemble -> no spread_* columns. Replaced by honestly-named real
    uncertainty proxies: sub-grid variability inside each 0.5 deg cell and
    recent day-to-day variability.
  * 10 days of one monsoon month -> 'season' is constant (monsoon) and
    'regime' is a simple, disclosed rule-based label (see infer_regime).

OUTPUT
------------------------------------------------------------------------------
  data/processed_features_real.parquet, one row per (init day, lead day,
  grid cell). With the supplied 10 days: 9 init days x leads 1-9 (45 valid
  pairs) x ~4,100 cells at ~0.48 deg ~= 185k rows.

USAGE
------------------------------------------------------------------------------
    pip install numpy pandas xarray netCDF4 pyarrow

    # point at the two zips exactly as downloaded (auto-extracted) ...
    python generate_data_and_features_real.py \
        --zips 54de903e-925e-4f67-a590-7312245ad181.zip \
               b7e92b5c-8377-414d-b562-a67a0243af71.zip

    # ... or at a folder that already contains the .nc files (searched
    # recursively, so more days can simply be dropped in)
    python generate_data_and_features_real.py --raw-dir data/raw

    python train_model.py --data data/processed_features_real.parquet
    python explainability.py --data data/processed_features_real.parquet
==============================================================================
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bust_detection.imdaa_datagen")

EARTH_M_PER_DEG_LAT = 111_320.0
KELVIN = 273.15

# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================


@dataclass
class Config:
    raw_dir: str = "data/raw"
    output_path: str = "data/processed_features_real.parquet"
    coarsen_factor: int = 4          # 0.12 deg native -> ~0.48 deg (the "0.5 deg grid")
    max_lead_day: int = 10           # Day 1-10; limited by how many days are supplied
    bust_rain_threshold_mm: float = 50.0
    bust_temp_threshold_c: float = 5.0
    recent_window_days: int = 3      # window for recent day-to-day variability
    neighbourhood_cells: int = 2     # neighbourhood radius in coarse cells (~1 deg at the default grid)
    # Rule-based regime thresholds (disclosed heuristic, see infer_regime)
    active_monsoon_rain_mm: float = 8.0      # monsoon-core area-mean rain
    bob_low_vorticity_1e5: float = 1.5       # head-BoB area-mean 850 hPa vorticity


# File-name patterns of the NCMRWF IMDAA downloads. Each maps to
# (variable key, regex with a YYYYMMDD group and optional HH group).
FILE_PATTERNS: Dict[str, re.Pattern] = {
    "t2m_max": re.compile(r"T2m_max_(\d{8})\.nc$", re.IGNORECASE),
    "rain": re.compile(r"APCP-sfc_(\d{8})\.nc$", re.IGNORECASE),
    "u850": re.compile(r"UGRD-850mb_(\d{8})(\d{2})_.*\.nc$", re.IGNORECASE),
    "v850": re.compile(r"VGRD-850mb_(\d{8})(\d{2})_.*\.nc$", re.IGNORECASE),
}
REQUIRED_VARS = ["t2m_max", "rain", "u850", "v850"]


# ===========================================================================
# 2. FILE DISCOVERY + LOADING
# ===========================================================================


def extract_zips(zip_paths: List[str], raw_dir: str) -> None:
    out = Path(raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    for zp in zip_paths:
        target = out / Path(zp).stem
        logger.info("Extracting %s -> %s", zp, target)
        with zipfile.ZipFile(zp) as zf:
            for member in zf.namelist():  # guard against path traversal
                if member.startswith("/") or ".." in Path(member).parts:
                    raise ValueError(f"Unsafe path inside {zp}: {member}")
            zf.extractall(target)


def discover_files(raw_dir: str) -> Dict[str, Dict[pd.Timestamp, List[Path]]]:
    """Find every recognised .nc file under raw_dir (recursively) and group
    them as {variable: {date: [paths]}} (winds have one file per 00Z/12Z)."""
    found: Dict[str, Dict[pd.Timestamp, List[Path]]] = {k: {} for k in FILE_PATTERNS}
    for path in sorted(Path(raw_dir).rglob("*.nc")):
        for key, pat in FILE_PATTERNS.items():
            m = pat.search(path.name)
            if m:
                date = pd.Timestamp(m.group(1))
                found[key].setdefault(date, []).append(path)
                break
    for key in REQUIRED_VARS:
        logger.info("  %-8s: %d day(s) found", key, len(found[key]))
        if not found[key]:
            raise FileNotFoundError(
                f"No '{key}' files found under {raw_dir}. Expected names like "
                f"T2m_max_20190701.nc, APCP-sfc_20190701.nc, UGRD-850mb_2019070100_..., VGRD-850mb_2019070100_..."
            )
    return found


def _read_2d(path: Path) -> xr.DataArray:
    """Open a single-field NetCDF and squeeze it to a (lat, lon) array."""
    with xr.open_dataset(path) as ds:
        var = list(ds.data_vars)[0]
        da = ds[var].load()
    da = da.squeeze(drop=True)  # drops time/height/plev singleton dims
    if set(da.dims) != {"lat", "lon"}:
        raise ValueError(f"{path.name}: expected (lat, lon) after squeeze, got {da.dims}")
    return da.transpose("lat", "lon").sortby("lat").sortby("lon")


def load_cubes(found: Dict[str, Dict[pd.Timestamp, List[Path]]]) -> Tuple[List[pd.Timestamp], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Load every variable for every common day onto ONE reference grid
    (the T2m grid). Winds are the daily mean of the 00Z and 12Z analyses.
    Returns (dates, lat, lon, {var: array[n_days, n_lat, n_lon]})."""
    common = sorted(set.intersection(*(set(found[k]) for k in REQUIRED_VARS)))
    if len(common) < 2:
        raise ValueError(f"Need at least 2 days present for all of {REQUIRED_VARS}; found {len(common)}.")
    gaps = np.diff(pd.DatetimeIndex(common)).astype("timedelta64[D]").astype(int)
    if (gaps != 1).any():
        logger.warning("Days are not contiguous (%s) -- lead days are computed from real date differences, so this is handled, but coverage has holes.", [str(d.date()) for d in common])
    logger.info("Using %d common day(s): %s -> %s", len(common), common[0].date(), common[-1].date())

    ref = _read_2d(found["t2m_max"][common[0]][0])
    lat, lon = ref["lat"].values, ref["lon"].values

    def on_ref_grid(da: xr.DataArray) -> np.ndarray:
        # IMDAA wind grid has one extra row (6.00N); select the T2m grid points exactly.
        return da.reindex(lat=lat, lon=lon, method="nearest", tolerance=1e-3).values

    cubes: Dict[str, List[np.ndarray]] = {k: [] for k in REQUIRED_VARS}
    for d in common:
        cubes["t2m_max"].append(on_ref_grid(_read_2d(found["t2m_max"][d][0])) - KELVIN)
        cubes["rain"].append(np.clip(on_ref_grid(_read_2d(found["rain"][d][0])), 0, None))
        for w in ("u850", "v850"):
            cubes[w].append(np.mean([on_ref_grid(_read_2d(p)) for p in found[w][d]], axis=0))

    out = {k: np.stack(v).astype(np.float64) for k, v in cubes.items()}
    for k, arr in out.items():
        nan_frac = float(np.isnan(arr).mean())
        if nan_frac > 0:
            logger.warning("  %s: %.2f%% NaN after grid alignment", k, 100 * nan_frac)
    logger.info(
        "Native grid %d x %d. Ranges -- T2m_max %.1f..%.1f C, rain %.1f..%.1f mm, u850 %.1f..%.1f, v850 %.1f..%.1f m/s",
        len(lat), len(lon),
        np.nanmin(out["t2m_max"]), np.nanmax(out["t2m_max"]), np.nanmin(out["rain"]), np.nanmax(out["rain"]),
        np.nanmin(out["u850"]), np.nanmax(out["u850"]), np.nanmin(out["v850"]), np.nanmax(out["v850"]),
    )
    return common, lat, lon, out


# ===========================================================================
# 3. COARSENING (native 0.12 deg -> ~0.5 deg cells)
# ===========================================================================


def coarsen(arr: np.ndarray, lat: np.ndarray, lon: np.ndarray, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Block-average by f x f (trimming the ragged edge). Also returns the
    within-block standard deviation -- real sub-grid heterogeneity, i.e. how
    much the 16 native pixels inside one 0.5 deg cell disagree."""
    n_lat, n_lon = (len(lat) // f) * f, (len(lon) // f) * f
    a = arr[:, :n_lat, :n_lon].reshape(arr.shape[0], n_lat // f, f, n_lon // f, f)
    mean = np.nanmean(a, axis=(2, 4))
    std = np.nanstd(a, axis=(2, 4))
    clat = lat[:n_lat].reshape(-1, f).mean(axis=1)
    clon = lon[:n_lon].reshape(-1, f).mean(axis=1)
    return mean, std, np.round(clat, 3), np.round(clon, 3)


# ===========================================================================
# 4. DYNAMICAL FEATURES FROM REAL 850 hPa WINDS
# ===========================================================================


def dynamics_850(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Dict[str, np.ndarray]:
    """Relative vorticity, divergence and wind-speed gradient on the coarse
    grid (centred differences, metric-corrected for latitude).
    Inputs are (n_days, n_lat, n_lon)."""
    dlat = float(np.mean(np.diff(lat)))
    dlon = float(np.mean(np.diff(lon)))
    m_per_deg_lon = (EARTH_M_PER_DEG_LAT * np.cos(np.radians(lat))).reshape(1, -1, 1)

    def d_dy(fld):
        return np.gradient(fld, dlat, axis=1, edge_order=2) / EARTH_M_PER_DEG_LAT

    def d_dx(fld):
        return np.gradient(fld, dlon, axis=2, edge_order=2) / m_per_deg_lon

    speed = np.sqrt(u ** 2 + v ** 2)
    return {
        "wind_speed850": speed,
        "relative_vorticity_850_1e5_s": (d_dx(v) - d_dy(u)) * 1e5,
        "divergence_850_1e5_s": (d_dx(u) + d_dy(v)) * 1e5,  # negative = convergence (feeds convection)
        "wind_speed_gradient_850_per_100km": np.sqrt(d_dx(speed) ** 2 + d_dy(speed) ** 2) * 1e5,
    }


# ===========================================================================
# 5. REGION / SEASON / REGIME / LEAD-TIME HELPERS (self-contained now)
# ===========================================================================

REGIONS = [
    "Himalaya", "Thar_Desert", "Indo_Gangetic_Plain", "Northeast_India",
    "Western_Ghats", "BoB_Coast", "Arabian_Sea_Coast", "Peninsular_Plateau",
]


def assign_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Approximate physiographic regions from simple lat/lon rules (no land
    mask is supplied, so adjacent sea points fall into the nearest coastal
    region). An approximation for grouping/reporting -- not official
    IMD subdivisions."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    out = np.full(lat.shape, "Peninsular_Plateau", dtype=object)

    west_coast = 72.8 + (22.0 - lat) * 0.27            # ~Gujarat (22N) -> Kerala (8N)
    east_coast = np.where(lat < 13.0, 79.5, 80.2 + (lat - 13.0) * 0.8)  # Chennai -> Odisha

    south = lat < 23.5
    out[south & (lon >= east_coast - 1.0)] = "BoB_Coast"
    out[south & (lon <= west_coast)] = "Arabian_Sea_Coast"
    out[south & (lon > west_coast) & (lon <= west_coast + 1.5) & (lat > 8.0) & (lat < 21.0)] = "Western_Ghats"
    out[(lat < 21.5) & (lon > 89.0)] = "BoB_Coast"

    north = lat >= 23.5
    out[north & (lon < 75.0)] = "Thar_Desert"
    out[north & (lon >= 75.0) & (lon <= 89.0)] = "Indo_Gangetic_Plain"
    out[(lat >= 21.5) & (lon > 89.0)] = "Northeast_India"
    out[(lat >= 30.0) & (lon <= 81.0)] = "Himalaya"
    out[(lat >= 27.5) & (lon > 81.0) & (lon <= 89.0)] = "Himalaya"
    out[(lat >= 30.0) & (lon > 81.0)] = "Himalaya"
    return out


def assign_season(month: np.ndarray) -> np.ndarray:
    month = np.asarray(month)
    return np.select(
        [np.isin(month, [6, 7, 8, 9]), np.isin(month, [10, 11, 12]), np.isin(month, [3, 4, 5])],
        ["monsoon", "post_monsoon", "pre_monsoon"],
        default="winter",
    )


def infer_regime(rain_day: np.ndarray, vort_day: np.ndarray, lat: np.ndarray, lon: np.ndarray, cfg: Config) -> Tuple[str, float, float]:
    """Disclosed rule-based synoptic label for ONE issue day, computed only
    from that day's IMDAA fields (so it is known at issue time):
      * bay_of_bengal_low : head-Bay (15-23N, 85-92E) mean 850 hPa vorticity
                            >= cfg.bob_low_vorticity_1e5 (a monsoon low/depression signature)
      * monsoon_trough    : monsoon-core (18-28N, 73-87E) mean rain
                            >= cfg.active_monsoon_rain_mm (active monsoon)
      * monsoon_break     : otherwise (weak / break phase)
    Not a validated synoptic classifier."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    bob = (la >= 15) & (la <= 23) & (lo >= 85) & (lo <= 92)
    core = (la >= 18) & (la <= 28) & (lo >= 73) & (lo <= 87)
    bob_vort = float(np.nanmean(vort_day[bob]))
    core_rain = float(np.nanmean(rain_day[core]))
    if bob_vort >= cfg.bob_low_vorticity_1e5:
        return "bay_of_bengal_low", bob_vort, core_rain
    if core_rain >= cfg.active_monsoon_rain_mm:
        return "monsoon_trough", bob_vort, core_rain
    return "monsoon_break", bob_vort, core_rain


def lead_day_bucket(lead: np.ndarray) -> np.ndarray:
    lead = np.asarray(lead)
    return np.select([lead <= 3, lead <= 7], ["day_1_3", "day_4_7"], default="day_8_10")


# ===========================================================================
# 6. PAIRS, LABELS, PAST-ONLY CLIMATOLOGY
# ===========================================================================


def build_pairs(dates: List[pd.Timestamp], max_lead: int) -> List[Tuple[int, int, int]]:
    """All (init_index, valid_index, lead_day) with 1 <= lead <= max_lead,
    lead measured in real calendar days."""
    pairs = []
    for i, di in enumerate(dates):
        for j, dj in enumerate(dates):
            lead = (dj - di).days
            if 1 <= lead <= max_lead:
                pairs.append((i, j, lead))
    return pairs


def past_only_climatology(
    pairs: List[Tuple[int, int, int]], t: np.ndarray, r: np.ndarray, cfg: Config, n_days: int, dates: List[pd.Timestamp]
) -> Dict[str, np.ndarray]:
    """For each issue day I, per grid cell: bust rate and mean absolute
    errors over every (init, lead) pair ALREADY VERIFIED by day I
    (valid date <= I). NaN when no history exists yet (first day).
    Returns {name: array[n_days, n_lat, n_lon]} indexed by issue day."""
    shape = (n_days,) + t.shape[1:]
    out = {k: np.full(shape, np.nan) for k in ("clim_hist_bust_rate", "clim_hist_mean_abs_temp_error", "clim_hist_mean_abs_rain_error")}
    for i in range(n_days):
        past = [(a, b) for a, b, _ in pairs if dates[b] <= dates[i]]
        if not past:
            continue
        te = np.stack([np.abs(t[b] - t[a]) for a, b in past])
        re_ = np.stack([np.abs(r[b] - r[a]) for a, b in past])
        bust = (te > cfg.bust_temp_threshold_c) | (re_ > cfg.bust_rain_threshold_mm)
        out["clim_hist_bust_rate"][i] = bust.mean(axis=0)
        out["clim_hist_mean_abs_temp_error"][i] = te.mean(axis=0)
        out["clim_hist_mean_abs_rain_error"][i] = re_.mean(axis=0)
    return out


def recent_features(t: np.ndarray, r: np.ndarray, dates: List[pd.Timestamp], window: int) -> Dict[str, np.ndarray]:
    """Day-to-day tendency and recent variability, using day I and the days
    BEFORE it only. NaN where not enough history exists."""
    n = t.shape[0]
    out = {k: np.full(t.shape, np.nan) for k in ("T2m_tendency_1d", "Rain_tendency_1d", "T2m_recent_std", "Rain_recent_std")}
    for i in range(n):
        if i >= 1 and (dates[i] - dates[i - 1]).days == 1:
            out["T2m_tendency_1d"][i] = t[i] - t[i - 1]
            out["Rain_tendency_1d"][i] = r[i] - r[i - 1]
        idx = [k for k in range(n) if 0 <= (dates[i] - dates[k]).days < window]
        if len(idx) >= 2:
            out["T2m_recent_std"][i] = np.std(t[idx], axis=0)
            out["Rain_recent_std"][i] = np.std(r[idx], axis=0)
    return out


def anomaly_and_context_features(
    t: np.ndarray, r: np.ndarray, dates: List[pd.Timestamp], lat: np.ndarray, lon: np.ndarray, cfg: "Config"
) -> Dict[str, np.ndarray]:
    """Regime-portable predictors, all known on the issue day (they use day I
    and EARLIER days only):
      * ANOMALIES -- how unusual today is at this spot, relative to the
        previous few days (T2m_anom_prev3, Rain_anom_prev3) and to everything
        seen at this spot so far (T2m_anom_to_date). Raw values say "it is
        38 C"; anomalies say "it is 4 C hotter than usual here", which carries
        over to new weather regimes far better.
      * RECENT WETNESS -- share of the last 5 days with > 10 mm rain.
      * NEIGHBOURHOOD -- heavy rain / sharp temperature contrasts within
        ~1 deg (about 2 grid cells). Weather moves, so what is nearby today
        is often here by the valid day.
    """
    from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

    n = t.shape[0]
    names = ("T2m_anom_prev3", "Rain_anom_prev3", "T2m_anom_to_date", "Rain_wet_days_prev5",
             "Rain_nbr_max", "Rain_nbr_mean", "T2m_nbr_range", "T2m_gradient_per_100km")
    out = {k: np.full(t.shape, np.nan) for k in names}
    size = 2 * cfg.neighbourhood_cells + 1

    def filled(a):
        return np.where(np.isfinite(a), a, np.nanmean(a))

    dlat, dlon = float(np.mean(np.diff(lat))), float(np.mean(np.diff(lon)))
    m_lon = (EARTH_M_PER_DEG_LAT * np.cos(np.radians(lat)))[:, None]
    for i in range(n):
        prev = [k for k in range(n) if 1 <= (dates[i] - dates[k]).days <= 3]
        if prev:
            out["T2m_anom_prev3"][i] = t[i] - np.nanmean(t[prev], axis=0)
            out["Rain_anom_prev3"][i] = r[i] - np.nanmean(r[prev], axis=0)
        to_date = [k for k in range(n) if 0 <= (dates[i] - dates[k]).days]
        if len(to_date) >= 3:
            out["T2m_anom_to_date"][i] = t[i] - np.nanmean(t[to_date], axis=0)
        last5 = [k for k in range(n) if 0 <= (dates[i] - dates[k]).days < 5]
        if len(last5) >= 3:
            out["Rain_wet_days_prev5"][i] = np.mean(r[last5] > 10.0, axis=0)
        mask = ~np.isfinite(r[i])
        rf, tf = filled(r[i]), filled(t[i])
        out["Rain_nbr_max"][i] = np.where(mask, np.nan, maximum_filter(rf, size=size, mode="nearest"))
        out["Rain_nbr_mean"][i] = np.where(mask, np.nan, uniform_filter(rf, size=size, mode="nearest"))
        out["T2m_nbr_range"][i] = np.where(mask, np.nan, maximum_filter(tf, size=size, mode="nearest") - minimum_filter(tf, size=size, mode="nearest"))
        gy = np.gradient(tf, dlat, axis=0) / EARTH_M_PER_DEG_LAT
        gx = np.gradient(tf, dlon, axis=1) / m_lon
        out["T2m_gradient_per_100km"][i] = np.where(mask, np.nan, np.sqrt(gx ** 2 + gy ** 2) * 1e5)
    return out


# ===========================================================================
# 7. MAIN PIPELINE
# ===========================================================================


def run_pipeline(cfg: Config) -> pd.DataFrame:
    t0 = time.time()
    logger.info("Scanning %s for IMDAA NetCDF files ...", cfg.raw_dir)
    found = discover_files(cfg.raw_dir)
    dates, lat_n, lon_n, fine = load_cubes(found)

    f = cfg.coarsen_factor
    t, t_sub, lat, lon = coarsen(fine["t2m_max"], lat_n, lon_n, f)
    r, r_sub, _, _ = coarsen(fine["rain"], lat_n, lon_n, f)
    u, _, _, _ = coarsen(fine["u850"], lat_n, lon_n, f)
    v, _, _, _ = coarsen(fine["v850"], lat_n, lon_n, f)
    n_days, n_lat, n_lon = t.shape
    logger.info("Coarsened by %dx%d -> %d x %d cells (~%.2f deg)", f, f, n_lat, n_lon, float(np.mean(np.diff(lat))))

    dyn = dynamics_850(u, v, lat, lon)
    recent = recent_features(t, r, dates, cfg.recent_window_days)
    recent.update(anomaly_and_context_features(t, r, dates, lat, lon, cfg))
    pairs = build_pairs(dates, cfg.max_lead_day)
    if not pairs:
        raise ValueError("No (init, valid) pairs could be formed -- need at least two dates one day apart.")
    max_lead_found = max(p[2] for p in pairs)
    logger.info("Built %d (issue day, valid day) pairs; lead days available: 1..%d", len(pairs), max_lead_found)
    clim = past_only_climatology(pairs, t, r, cfg, n_days, dates)

    regimes = []
    regime_idx = []  # continuous versions of the regime label -- generalise better than a category
    for i, d in enumerate(dates):
        reg, bob_v, core_r = infer_regime(r[i], dyn["relative_vorticity_850_1e5_s"][i], lat, lon, cfg)
        regimes.append(reg)
        regime_idx.append((bob_v, core_r))
        logger.info("  %s regime=%-18s (head-BoB vort %.2f e-5/s, monsoon-core rain %.1f mm)", d.date(), reg, bob_v, core_r)

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    region2d = assign_region(lat2d, lon2d)
    flat = lambda a: a.ravel()

    frames = []
    for i, j, lead in pairs:
        frames.append(pd.DataFrame({
            "init_time": dates[i],
            "valid_time": dates[j],
            "lead_day": lead,
            "lat": flat(lat2d), "lon": flat(lon2d),
            "region": flat(region2d),
            "regime": regimes[i],
            "idx_bob_vorticity": regime_idx[i][0],
            "idx_monsoon_core_rain": regime_idx[i][1],
            # persistence "forecast" = what IMDAA shows on the issue day
            "T2m": flat(t[i]), "Rain_fcst": flat(r[i]),
            "T2m_subgrid_std": flat(t_sub[i]), "Rain_subgrid_std": flat(r_sub[i]),
            "U850": flat(u[i]), "V850": flat(v[i]),
            **{k: flat(a[i]) for k, a in dyn.items()},
            **{k: flat(a[i]) for k, a in recent.items()},
            **{k: flat(a[i]) for k, a in clim.items()},
            # truth on the valid day -- LABEL ONLY, excluded from features by train_model.py
            "Obs_T2m": flat(t[j]), "Obs_Rain": flat(r[j]),
        }))
    df = pd.concat(frames, ignore_index=True)

    # Drop cells with no data at all (ragged coastline / edge NaN)
    before = len(df)
    df = df.dropna(subset=["T2m", "Rain_fcst", "Obs_T2m", "Obs_Rain", "U850", "V850"]).reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing core fields.", before - len(df))

    df["temp_error"] = df["Obs_T2m"] - df["T2m"]
    df["rain_error"] = df["Obs_Rain"] - df["Rain_fcst"]
    df["is_bust"] = ((df["temp_error"].abs() > cfg.bust_temp_threshold_c) | (df["rain_error"].abs() > cfg.bust_rain_threshold_mm)).astype(np.int8)
    df["error_magnitude"] = np.maximum(df["rain_error"].abs() / cfg.bust_rain_threshold_mm, df["temp_error"].abs() / cfg.bust_temp_threshold_c)

    df["lead_day_norm"] = df["lead_day"] / float(cfg.max_lead_day)
    df["lead_day_bucket"] = lead_day_bucket(df["lead_day"].to_numpy())
    df["season"] = assign_season(df["init_time"].dt.month.to_numpy())

    df = optimize_dtypes(df)
    print_sanity_report(df)
    logger.info("Pipeline complete in %.1f s. Final shape: %s", time.time() - t0, df.shape)
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)
    df["lead_day"] = df["lead_day"].astype(np.int16)
    for c in ("region", "regime", "season", "lead_day_bucket"):
        df[c] = df[c].astype("category")
    return df


def print_sanity_report(df: pd.DataFrame) -> None:
    n_bust = int(df["is_bust"].sum())
    print("\n" + "=" * 74)
    print("CLASS BALANCE CHECK (real IMDAA truth vs persistence forecast)")
    print("=" * 74)
    print(f"  Rows            : {len(df):,}")
    print(f"  Busts           : {n_bust:,}  ({100 * n_bust / len(df):.2f}%)")
    print(f"    rain-driven   : {int((df['rain_error'].abs() > 50).sum()):,}")
    print(f"    temp-driven   : {int((df['temp_error'].abs() > 5).sum()):,}")
    print("  Bust rate by lead day (%):")
    print("   ", (df.groupby("lead_day")["is_bust"].mean() * 100).round(2).to_dict())
    print("  Bust rate by region (%):")
    print("   ", (df.groupby("region", observed=True)["is_bust"].mean() * 100).round(2).to_dict())
    print("  Rows per valid day (train_model.py splits on this):")
    print("   ", df.groupby("valid_time").size().rename(lambda d: str(d.date())).to_dict())
    if n_bust < 50:
        print("\n  WARNING: very few real busts. Add more IMDAA days (drop more .nc files into --raw-dir).")
    print("=" * 74 + "\n")


def export_dataset(df: pd.DataFrame, path: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Wrote %s (%.1f MB, %d columns)", out, out.stat().st_size / 1e6, df.shape[1])
    return out


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the BustGuard training dataset from NCMRWF IMDAA NetCDF files.")
    p.add_argument("--raw-dir", default=Config.raw_dir, help="Folder searched recursively for .nc files.")
    p.add_argument("--zips", nargs="*", default=[], help="IMDAA download zips to extract into --raw-dir first.")
    p.add_argument("--output", default=Config.output_path)
    p.add_argument("--coarsen-factor", type=int, default=Config.coarsen_factor, help="1 = native 0.12 deg (large!), 4 = ~0.5 deg.")
    p.add_argument("--max-lead-day", type=int, default=Config.max_lead_day)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config(raw_dir=args.raw_dir, output_path=args.output, coarsen_factor=args.coarsen_factor, max_lead_day=args.max_lead_day)
    if args.zips:
        extract_zips(args.zips, cfg.raw_dir)
    df = run_pipeline(cfg)
    out = export_dataset(df, cfg.output_path)
    print(f"Done. Real IMDAA dataset written to {out}")
    print(f"Next: python train_model.py --data {cfg.output_path}")


if __name__ == "__main__":
    main()