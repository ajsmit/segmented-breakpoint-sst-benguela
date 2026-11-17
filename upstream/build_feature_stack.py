#!/usr/bin/env python3
"""07_build_feature_stack.py — Stage 4: SST feature stack on the FMM mesh.

Produces the per-(product, year, season, sigma-node) feature vector that
feeds the GMM latent-class fit (Stage 5).  Covers 14 features: three
breakpoint-regression features, the 11 SST-derived canonical features, and
the two static bathymetric features (d_200, d_500) from the mesh.
Wind-derived features (tau_eq, zeta, w_theory) are appended by a separate
script once ERA5 data are complete.

Three-pass algorithm
--------------------
  Pass 1 — Breakpoint regression (per transect-day):
    Fit the continuous two-segment piecewise-linear model with continuity at
    the breakpoint x_b, namely
        T_p(x) = alpha_in + beta_in * min(x, x_b) + beta_off * max(0, x - x_b),
    using a vectorised prefix-sum grid search over candidate x_b.  The
    inshore/offshore partition is data-determined; no fixed sigma threshold
    is applied.  Three acceptance criteria must all be satisfied:
    DELTA_AIC_MIN, BP_MIN_OFFSHORE_KM, and BP_MIN_SLOPE_RATIO (see module
    constants).  Accepted fits contribute beta_inshore, beta_offshore,
    x_b_km and the offshore-segment mean as the per-transect offshore
    reference T_hat_bg.

  Pass 2 — SST features (per node):
    Eleven SST-derived features as in the uc2d baseline, but the anomaly
    A_h = T_hat_bg - T_p now uses the breakpoint-derived offshore reference.
    Nodes on unaccepted transects have NaN anomaly values.

  Pass 3 — Bathymetry (static, joined from mesh):
    d_200_km and d_500_km from mesh_crossline_isobath_clips.

Breakpoint features (3)
-----------------------
  beta_inshore   Inshore OLS slope (K/km)
  beta_offshore  Offshore OLS slope (K/km)
  x_b_km         Cross-shore distance of breakpoint from coast (km)
  bp_accepted    0/1 acceptance flag (per transect-day)
  bp_segment     Node label: "inshore", "offshore", or "unaccepted"

SST features (11)
-----------------
  anomaly_g1/g2/g4   SST anomaly relative to breakpoint offshore reference
  gn_g1/g2/g4        Cross-shore (normal) SST gradient at three scales
  ratio_g1/g2        |gn| / |gt| cross-to-along-shore gradient ratio
  var_g2             Local SST variance at intermediate scale
  lap_g1             Laplacian at finest scale
  dog_g2             Difference-of-Gaussians (s_g1 - s_g4)

Static bathymetric features (2)
--------------------------------
  d_200_km, d_500_km  Distance from coast to isobath (km); NaN if absent

Outputs
-------
  data/outputs/uc2d_v2/features_bp/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>.parquet
  data/outputs/uc2d_v2/features_bp/<PRODUCT>/<PRODUCT>_<YEAR>_<SEASON>_meta.json

Usage
-----
  python3 upstream/build_feature_stack.py \\
      --product OSTIA --year 2015
  python3 upstream/build_feature_stack.py \\
      --product OSTIA --year 2015 --season JJA
  python3 upstream/build_feature_stack.py \\
      --product OSTIA --date 2020-07-15     # smoke-test: single day
  python3 upstream/build_feature_stack.py \\
      --product OSTIA --year 2015 --dry-run

Dependencies: numpy, scipy, pandas, netCDF4, pyarrow
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from calendar import monthrange
from datetime import date, datetime

import netCDF4 as nc4
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# ── Configuration ──────────────────────────────────────────────────────────────
BBOX = dict(lon_min=4.5, lon_max=28.0, lat_min=-41.0, lat_max=-11.0)

SUPPORT_THRESHOLD = 0.50   # minimum Gaussian support fraction for valid output
KM_PER_DEG_LAT   = 111.32

# Native pixel grain per product.  The SST profile is aggregated to one value
# per native pixel before the breakpoint regression runs, so that n in the AIC
# formula reflects independent observations rather than the mesh sigma-node count
# (~480 per transect regardless of product resolution).
NATIVE_GRAIN_KM: dict[str, float] = {
    "OSTIA": 5.56,
    "OSPO":  5.56,
    "AVHRR": 27.8,
    "MW_OI": 25.0,   # 0.25-deg product; ~25 km cross-shore at 26 S
}

# Breakpoint acceptance criteria.  DELTA_AIC_MIN and BP_MIN_OFFSHORE_KM are
# uniform across all products: the AIC test now operates on native-pixel n so
# the threshold is statistically equivalent for all products.  BP_MIN_SLOPE_RATIO
# is per-product: at coarse grain (27–28 km) the SST smoothing compresses both
# gradients, but the inshore gradient more than the offshore one, so the apparent
# slope ratio is structurally lower than at fine grain.  1.5 is the relaxed value
# for AVHRR and MW_OI; the AIC at n~18 still has real bite (needs ~56% RSS
# reduction), so the relaxed slope-ratio is not the only noise filter.
DELTA_AIC_MIN      = 4.0    # minimum ΔAIC = AIC_linear - AIC_segmented (uniform)
BP_MIN_OFFSHORE_KM = 100.0  # minimum offshore-segment length (km, uniform)
BP_MIN_SLOPE_RATIO = 2.0    # minimum |beta_inshore / beta_offshore| (uniform)
BP_EDGE_BUFFER_KM  = 10.0   # minimum distance of breakpoint from coast (km, uniform)

# Parameter counts in the AIC test (Burnham & Anderson 2002).  K_ONE_SEGMENT is
# the one-segment linear comparator (alpha, beta).  K_TWO_SEGMENT is the
# continuous two-segment model (alpha_in, beta_in, beta_off, x_b); alpha_off is
# derived as alpha_off = alpha_in + beta_in * x_b at the breakpoint and is NOT a
# free parameter.  DELTA_AIC_MIN = 2 * (K_TWO_SEGMENT - K_ONE_SEGMENT) = 4 is
# Burnham & Anderson's "considerable evidence" bar.
K_ONE_SEGMENT = 2
K_TWO_SEGMENT = 4

# SIGMA_BG retained for the Stage 10 baseline comparison only; not used here.
SIGMA_BG = 0.70   # DEPRECATED: fixed offshore threshold (Stage 10 baseline)

BP_FEATURES  = ["alpha_inshore", "beta_inshore", "beta_offshore", "x_b_km"]
BP_DIAGNOSTIC_FEATURES = [
    "rss_one_segment_K2",
    "rss_two_segment_K2",
    "aic_one_segment",
    "aic_two_segment",
    "delta_aic",
]
SST_FEATURES = [
    "anomaly_g1", "anomaly_g2", "anomaly_g4",
    "gn_g1", "gn_g2", "gn_g4",
    "ratio_g1", "ratio_g2",
    "var_g2", "lap_g1", "dog_g2",
]
CANONICAL_FEATURES = BP_FEATURES + SST_FEATURES   # 15 total (alpha_inshore added under the continuity refactor)

SEASONS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

PRODUCT_CFG: dict[str, dict] = {
    "OSTIA": {
        "dir": "data/raw/OSTIA",
        "glob": "*.nc4",
        "sst_var": "analysed_sst",
        "g1_km": 21.14, "g2_km": 42.27, "g4_km": 84.55,
        "nominal_km": 5.56,
    },
    "OSPO": {
        "dir": "data/raw/OSPO",
        "glob": "*.nc",
        "sst_var": "analysed_sst",
        "g1_km": 21.16, "g2_km": 42.32, "g4_km": 84.63,
        "nominal_km": 5.56,
    },
    "AVHRR": {
        "dir": "data/raw/AVHRR",
        "glob": "*.nc4",
        "sst_var": "analysed_sst",
        "g1_km": 105.69, "g2_km": 211.37, "g4_km": 422.75,
        "nominal_km": 27.8,
    },
    "MW_OI": {
        "dir": "data/raw/MW_OI",
        "glob": "*.nc",
        "sst_var": "analysed_sst",
        "g1_km": 105.79, "g2_km": 211.59, "g4_km": 423.17,
        "nominal_km": 27.8,
    },
}

MESH_ROOT    = "data/derived/analysis_runs"
FEAT_BP_ROOT = "data/outputs/uc2d_v2/features_bp"   # breakpoint rebuild output
TARGET_YEARS = list(range(2015, 2026))

DATE_RE = re.compile(r"(\d{8})\d{6}")


# ── Grid helpers ───────────────────────────────────────────────────────────────

def km_per_deg_lon(lat_deg: float) -> float:
    return KM_PER_DEG_LAT * np.cos(np.radians(lat_deg))


def read_sst(path: str, cfg: dict, bbox: dict) -> dict | None:
    """Read SST (K) and valid mask from a GHRSST L4 NetCDF, subsetted to bbox."""
    try:
        with nc4.Dataset(path, "r") as ds:
            lon_var = next((ds.variables[k] for k in ("lon", "longitude")
                            if k in ds.variables), None)
            lat_var = next((ds.variables[k] for k in ("lat", "latitude")
                            if k in ds.variables), None)
            if lon_var is None or lat_var is None:
                return None
            lon = lon_var[:].data.astype(float)
            lat = lat_var[:].data.astype(float)

            lon_idx = np.where((lon >= bbox["lon_min"]) & (lon <= bbox["lon_max"]))[0]
            lat_idx = np.where((lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]))[0]
            if not len(lon_idx) or not len(lat_idx):
                return None

            sst_var = ds.variables.get(cfg["sst_var"])
            if sst_var is None:
                return None

            slices = []
            for dim in sst_var.dimensions:
                if dim in ("lon", "longitude", "x"):
                    slices.append(slice(int(lon_idx[0]), int(lon_idx[-1]) + 1))
                elif dim in ("lat", "latitude", "y"):
                    slices.append(slice(int(lat_idx[0]), int(lat_idx[-1]) + 1))
                else:
                    slices.append(0)

            raw_nc = sst_var[tuple(slices)]
            if hasattr(raw_nc, "filled"):
                raw = raw_nc.filled(np.nan).astype(float)
            else:
                raw = np.array(raw_nc, dtype=float)
            if raw.ndim > 2:
                raw = raw.squeeze()
            # Ensure (lat, lon) orientation
            lon_ax = next((i for i, d in enumerate(sst_var.dimensions)
                           if d in ("lon", "longitude", "x")), None)
            lat_ax = next((i for i, d in enumerate(sst_var.dimensions)
                           if d in ("lat", "latitude", "y")), None)
            if lon_ax is not None and lat_ax is not None and lat_ax > lon_ax:
                raw = raw.T

            raw[~np.isfinite(raw)] = np.nan
            raw[(raw < 250) | (raw > 330)] = np.nan
            valid = np.isfinite(raw)

            return {
                "lon": lon[lon_idx],
                "lat": lat[lat_idx],
                "sst": raw,
                "valid": valid,
            }
    except Exception:
        return None


# ── Gaussian smoothing ─────────────────────────────────────────────────────────

def smooth_valid(sst: np.ndarray, valid: np.ndarray,
                 sigma_cells: float) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Gaussian smooth that respects NaN boundaries."""
    sigma_cells = max(0.75, sigma_cells)
    num = gaussian_filter(np.where(valid, sst, 0.0), sigma=sigma_cells,
                          mode="constant", cval=0, truncate=3.0)
    den = gaussian_filter(valid.astype(float), sigma=sigma_cells,
                          mode="constant", cval=0, truncate=3.0)
    out = np.full_like(num, np.nan)
    np.divide(num, den, out=out, where=den > 1e-9)
    out[den < SUPPORT_THRESHOLD] = np.nan
    return out, den


# ── Gradient computation ───────────────────────────────────────────────────────

def gradient_fields(mat: np.ndarray, lon: np.ndarray,
                    lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central-difference gx (K/km east), gy (K/km north), and Laplacian."""
    nr, nc = mat.shape
    dx_deg = float(np.median(np.diff(lon)))
    dy_km  = float(np.median(np.diff(lat))) * KM_PER_DEG_LAT

    gx  = np.full((nr, nc), np.nan)
    gy  = np.full((nr, nc), np.nan)
    lap = np.full((nr, nc), np.nan)

    dx_km_row = np.abs(km_per_deg_lon(lat) * dx_deg)
    L, R = mat[:, :-2], mat[:, 2:]
    ok = np.isfinite(L) & np.isfinite(R)
    gx[:, 1:-1] = np.where(ok, (R - L) / (2 * dx_km_row[:, None]), np.nan)

    U, D = mat[:-2, :], mat[2:, :]
    ok = np.isfinite(U) & np.isfinite(D)
    gy[1:-1, :] = np.where(ok, (D - U) / (2 * dy_km), np.nan)

    C = mat[1:-1, 1:-1]
    ok = (np.isfinite(C) & np.isfinite(mat[1:-1, :-2]) & np.isfinite(mat[1:-1, 2:])
          & np.isfinite(mat[:-2, 1:-1]) & np.isfinite(mat[2:, 1:-1]))
    dx2 = (dx_km_row[1:-1, None]) ** 2
    lap[1:-1, 1:-1] = np.where(
        ok,
        (mat[1:-1, 2:] - 2*C + mat[1:-1, :-2]) / dx2
        + (mat[2:, 1:-1] - 2*C + mat[:-2, 1:-1]) / dy_km**2,
        np.nan,
    )
    return gx, gy, lap


# ── Interpolation ──────────────────────────────────────────────────────────────

def make_interp(lon: np.ndarray, lat: np.ndarray,
                field: np.ndarray) -> RegularGridInterpolator:
    return RegularGridInterpolator(
        (lat, lon), field,
        method="linear", bounds_error=False, fill_value=np.nan,
    )


def interp_to_nodes(lon: np.ndarray, lat: np.ndarray,
                    field: np.ndarray,
                    node_lat: np.ndarray, node_lon: np.ndarray) -> np.ndarray:
    interp = make_interp(lon, lat, field)
    return interp(np.column_stack([node_lat, node_lon]))


# ── Mesh loading ───────────────────────────────────────────────────────────────

def load_mesh(mesh_tag: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = os.path.join(MESH_ROOT, mesh_tag, "mesh_curvilinear")
    nodes = pd.read_csv(os.path.join(base, "mesh_nodes.csv"))
    bnd   = pd.read_csv(os.path.join(base, "mesh_boundary_match.csv"))
    iso   = pd.read_csv(os.path.join(base, "mesh_crossline_isobath_clips.csv"))
    return nodes, bnd, iso


def build_transect_geometry(bnd: pd.DataFrame) -> pd.DataFrame:
    """Compute per-transect unit normal and tangent in (east, north) km basis."""
    mid_lat = (bnd["coast_lat"] + bnd["outer_lat"]) / 2
    dlat_km = (bnd["outer_lat"] - bnd["coast_lat"]) * KM_PER_DEG_LAT
    dlon_km = (bnd["outer_lon"] - bnd["coast_lon"]) * km_per_deg_lon(mid_lat)
    dist    = np.sqrt(dlat_km**2 + dlon_km**2)
    bnd = bnd.copy()
    bnd["nx"] = dlon_km / dist
    bnd["ny"] = dlat_km / dist
    bnd["tx"] = -bnd["ny"]
    bnd["ty"] =  bnd["nx"]
    return bnd


def build_static_features(nodes: pd.DataFrame, bnd: pd.DataFrame,
                           iso: pd.DataFrame) -> pd.DataFrame:
    """Merge geometry and isobath features onto each sigma node."""
    iso_w = iso.pivot(index="col_id", columns="isobath_m", values="fraction").reset_index()
    iso_w.columns.name = None
    iso_w = iso_w.rename(columns={200: "frac_200", 500: "frac_500"})
    iso_w = iso_w[["col_id", "frac_200", "frac_500"]]

    geom = bnd[["col_id", "coast_s_km", "coast_lon", "coast_lat",
                 "line_length_km", "nx", "ny", "tx", "ty"]].copy()
    geom = geom.merge(iso_w, on="col_id", how="left")
    geom["d_200_km"] = geom["frac_200"] * geom["line_length_km"]
    geom["d_500_km"] = geom["frac_500"] * geom["line_length_km"]

    out = nodes.merge(geom, on="col_id", how="left")
    out["distance_km"] = out["sigma"] * out["line_length_km"]
    return out


# ── Breakpoint regression (Pass 1) ────────────────────────────────────────────

def _ols_rss(x: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares for OLS fit of y on x.  Returns inf if n < 2."""
    n = len(x)
    if n < 2:
        return np.inf
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx < 1e-12:
        return float(((y - ym) ** 2).sum())
    beta = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym + beta * (x - xm))
    return float((resid ** 2).sum())


def _ols_params(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope, intercept) for OLS fit.  Returns (0, mean(y)) if degenerate."""
    n = len(x)
    if n < 2:
        return 0.0, float(y.mean()) if len(y) else 0.0
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx < 1e-12:
        return 0.0, ym
    slope = ((x - xm) * (y - ym)).sum() / sxx
    intercept = ym - slope * xm
    return float(slope), float(intercept)


def _aic(n: int, rss: float, k: int) -> float:
    """AIC = n * log(rss/n) + 2*k.  Returns inf if rss <= 0 or n <= k."""
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def _grid_search_breakpoint(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """O(n) prefix-sum grid search for the constrained continuous breakpoint.

    Fits the continuous two-segment piecewise-linear model
        y = alpha_in + beta_in * min(x, x_b) + beta_off * max(0, x - x_b)
    at every candidate breakpoint x_b in the data, and returns the
    (x_b, alpha_in, beta_in, beta_off, RSS) minimising total RSS.  Continuity at
    the breakpoint is by construction: the offshore intercept is
    alpha_off = alpha_in + beta_in * x_b, derived rather than fitted.  The
    parameter count for the AIC test is therefore K_TWO_SEGMENT = 4
    (alpha_in, beta_in, beta_off, x_b), matching the manuscript Eq.~1.

    For each candidate index k (number of points on the inshore segment, with
    x_b = xs[k]), the 3x3 normal-equation matrix for (alpha_in, beta_in,
    beta_off) is built from prefix sums of x, x^2, y, y^2, and xy in O(1) and
    solved in O(1).  Total cost is O(n) per transect.

    Returns (x_b_km, alpha_inshore, beta_inshore, beta_offshore, rss_total).
    Raises ValueError if fewer than 4 valid points or if no candidate split
    yields a finite, well-posed solution.
    """
    n = len(x)
    if n < 4:
        raise ValueError("Fewer than 4 valid points; cannot fit breakpoint.")

    # Sort by distance (should already be sorted but be safe)
    order = np.argsort(x)
    xs, ys = x[order].astype(float), y[order].astype(float)

    # Prefix sums (P[k] = sum over i in 0..k-1; so P[0] = 0 and P[n] = total).
    Sx_pref  = np.concatenate(([0.0], np.cumsum(xs)))
    Sxx_pref = np.concatenate(([0.0], np.cumsum(xs * xs)))
    Sy_pref  = np.concatenate(([0.0], np.cumsum(ys)))
    Sxy_pref = np.concatenate(([0.0], np.cumsum(xs * ys)))
    Syy_tot  = float((ys * ys).sum())
    Sx_tot   = Sx_pref[-1]
    Sxx_tot  = Sxx_pref[-1]
    Sy_tot   = Sy_pref[-1]
    Sxy_tot  = Sxy_pref[-1]

    best_rss      = np.inf
    best_xb       = np.nan
    best_alpha_in = np.nan
    best_beta_in  = np.nan
    best_beta_off = np.nan

    # Candidate splits: left has k points, right has n-k points (both >= 2)
    for k in range(2, n - 1):
        x_b   = float(xs[k])
        n_R   = float(n - k)
        # Left-side prefix sums up to index k
        Sx_L  = Sx_pref[k]
        Sxx_L = Sxx_pref[k]
        Sy_L  = Sy_pref[k]
        Sxy_L = Sxy_pref[k]
        # Right-side raw sums starting at index k
        Sx_R  = Sx_tot  - Sx_L
        Sxx_R = Sxx_tot - Sxx_L
        Sy_R  = Sy_tot  - Sy_L
        Sxy_R = Sxy_tot - Sxy_L
        # Right-side centred sums (about x_b)
        T1   = Sx_R - n_R * x_b                          # sum_{i>=k} (x_i - x_b)
        T2   = Sxx_R - 2.0 * x_b * Sx_R + n_R * x_b * x_b  # sum_{i>=k} (x_i - x_b)^2
        Txy  = Sxy_R - x_b * Sy_R                        # sum_{i>=k} (x_i - x_b) y_i

        # Symmetric 3x3 normal-equation matrix X^T X for
        # design columns (1, u, v) where u_i = min(x_i, x_b), v_i = max(0, x_i - x_b).
        a00 = float(n)
        a01 = Sx_L + n_R * x_b
        a02 = T1
        a11 = Sxx_L + n_R * x_b * x_b
        a12 = x_b * T1
        a22 = T2

        # Right-hand side X^T y
        b0 = Sy_tot
        b1 = Sxy_L + x_b * Sy_R
        b2 = Txy

        A = np.array([
            [a00, a01, a02],
            [a01, a11, a12],
            [a02, a12, a22],
        ])
        bvec = np.array([b0, b1, b2])

        try:
            theta = np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError:
            continue
        alpha_in, beta_in, beta_off = float(theta[0]), float(theta[1]), float(theta[2])

        # Total RSS = sum(y_i^2) - theta^T (X^T y), guarded against tiny -ve.
        rss = float(Syy_tot - theta @ bvec)
        if rss < 0.0:
            rss = 0.0

        if rss < best_rss:
            best_rss      = rss
            best_xb       = x_b
            best_alpha_in = alpha_in
            best_beta_in  = beta_in
            best_beta_off = beta_off

    if not np.isfinite(best_rss):
        raise ValueError("No valid breakpoint found.")

    return best_xb, best_alpha_in, best_beta_in, best_beta_off, best_rss


def _sample_at_native_pixels(
    dist_km: np.ndarray, sst: np.ndarray, native_grain_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (dist_km, sst) at native-pixel centres along the transect.

    Places pixel centres at x_k = (k - 0.5) * native_grain_km for
    k = 1, 2, ..., ceil(x_max / native_grain_km).  For each centre the mesh
    node with the closest dist_km is selected (one independent observation per
    native pixel).  Entries where the sampled sst is NaN are dropped.

    The resulting arrays have length approximately x_max / native_grain_km,
    which is the correct effective sample size n for the AIC test.
    """
    x_max   = dist_km.max()
    centres = (np.arange(1, int(np.ceil(x_max / native_grain_km)) + 1) - 0.5) * native_grain_km
    idx     = np.argmin(np.abs(dist_km[:, None] - centres[None, :]), axis=0)
    dist_sub, sst_sub = dist_km[idx], sst[idx]
    valid   = np.isfinite(sst_sub)
    return dist_sub[valid], sst_sub[valid]


def _fit_one_transect(
    dist_km: np.ndarray, sst_g1: np.ndarray,
    sst_g2: np.ndarray, sst_g4: np.ndarray,
    native_grain_km: float,
) -> dict:
    """Fit the breakpoint on one (col_id, date) transect using g1 SST.

    The g1 profile is sampled at native-pixel centres before the regression so
    that n in the AIC formula equals the number of independent observations.
    The offshore reference (bg_g1, bg_g2, bg_g4) is then computed on the full
    480-node mesh using the breakpoint position from the sampled fit.

    Returns a dict with keys: x_b_km, beta_inshore, beta_offshore, bp_accepted,
    bg_g1, bg_g2, bg_g4, n_eff, rss_one_segment_K2, rss_two_segment_K2,
    aic_one_segment, aic_two_segment, and delta_aic.  Early validity and
    position-constraint failures return NaN fit diagnostics and bp_accepted=0;
    transects that pass geometry but fail the operational ΔAIC or slope-ratio
    checks retain the fitted diagnostics with bp_accepted=0 so downstream
    sensitivity scans can re-screen at alternative thresholds.  n_eff carries
    the actual pixel count even for rejected transects so that
    acceptance-fraction diagnostics can report the correct n.
    """
    nan_result = dict(
        x_b_km=np.nan, alpha_inshore=np.nan,
        beta_inshore=np.nan, beta_offshore=np.nan,
        bp_accepted=0,
        bg_g1=np.nan, bg_g2=np.nan, bg_g4=np.nan,
        n_eff=0,
        rss_one_segment_K2=np.nan,
        rss_two_segment_K2=np.nan,
        aic_one_segment=np.nan,
        aic_two_segment=np.nan,
        delta_aic=np.nan,
    )

    valid = np.isfinite(sst_g1) & np.isfinite(sst_g2) & np.isfinite(sst_g4)
    if valid.sum() < 6:
        return nan_result

    # Sample at native-pixel centres; n_eff is the effective sample size for AIC
    dist_sub, sst_sub = _sample_at_native_pixels(
        dist_km[valid], sst_g1[valid], native_grain_km)
    n_eff = len(dist_sub)
    nan_result["n_eff"] = n_eff   # propagate to all subsequent rejection returns
    if n_eff < 6:
        return nan_result

    x_max = dist_sub.max()

    # Transect must be long enough to accommodate both buffers and min offshore
    if x_max < BP_EDGE_BUFFER_KM + BP_MIN_OFFSHORE_KM + BP_EDGE_BUFFER_KM:
        return nan_result

    try:
        x_b, alpha_in, beta_in, beta_off, rss_seg = _grid_search_breakpoint(
            dist_sub, sst_sub)
    except ValueError:
        return nan_result

    # AIC criterion: n_eff is the native-pixel count, not the 480 mesh nodes.
    # The two-segment count is K_TWO_SEGMENT = 4 because the offshore
    # intercept is determined by continuity (alpha_off = alpha_in + beta_in*x_b)
    # and is not a free parameter.
    rss_lin = _ols_rss(dist_sub, sst_sub)
    aic_lin = _aic(n_eff, rss_lin, K_ONE_SEGMENT)
    aic_seg = _aic(n_eff, rss_seg, K_TWO_SEGMENT)
    delta_aic = aic_lin - aic_seg

    # Position constraint
    offshore_len = x_max - x_b
    if x_b < BP_EDGE_BUFFER_KM or offshore_len < BP_MIN_OFFSHORE_KM:
        return nan_result

    # Slope-ratio criterion
    slope_ratio = abs(beta_in / beta_off) if abs(beta_off) > 1e-9 else np.inf

    # Offshore reference on the full 480-node mesh
    off_mask = valid & (dist_km >= x_b)
    if off_mask.sum() == 0:
        return nan_result

    result = dict(
        x_b_km=x_b,
        alpha_inshore=alpha_in,
        beta_inshore=beta_in,
        beta_offshore=beta_off,
        bp_accepted=int(delta_aic >= DELTA_AIC_MIN and slope_ratio >= BP_MIN_SLOPE_RATIO),
        bg_g1=float(sst_g1[off_mask].mean()),
        bg_g2=float(sst_g2[off_mask].mean()),
        bg_g4=float(sst_g4[off_mask].mean()),
        n_eff=n_eff,
        rss_one_segment_K2=float(rss_lin),
        rss_two_segment_K2=float(rss_seg),
        aic_one_segment=float(aic_lin),
        aic_two_segment=float(aic_seg),
        delta_aic=float(delta_aic),
    )
    return result


def fit_breakpoints(
    df: pd.DataFrame, native_grain_km: float,
) -> pd.DataFrame:
    """Run the breakpoint regression pass over all (col_id, date) groups.

    Adds columns: beta_inshore, beta_offshore, x_b_km, bp_accepted,
    bg_g1, bg_g2, bg_g4, n_eff, rss_one_segment_K2, rss_two_segment_K2,
    aic_one_segment, aic_two_segment, delta_aic, bp_segment.
    """
    records = []
    for (col_id, dt), grp in df.groupby(["col_id", "date"], sort=False):
        result = _fit_one_transect(
            dist_km=grp["distance_km"].values,
            sst_g1=grp["sst_g1"].values,
            sst_g2=grp["sst_g2"].values,
            sst_g4=grp["sst_g4"].values,
            native_grain_km=native_grain_km,
        )
        result["col_id"] = col_id
        result["date"]   = dt
        records.append(result)

    bp_df = pd.DataFrame(records)
    df = df.merge(bp_df, on=["col_id", "date"], how="left")

    conditions = [
        df["bp_accepted"].eq(1) & (df["distance_km"] <  df["x_b_km"]),
        df["bp_accepted"].eq(1) & (df["distance_km"] >= df["x_b_km"]),
    ]
    df["bp_segment"] = np.select(conditions, ["inshore", "offshore"],
                                 default="unaccepted")
    return df


# ── Per-date feature computation ───────────────────────────────────────────────

def features_one_day(
    nc_path: str,
    cfg: dict,
    static: pd.DataFrame,
    date_str: str,
    product: str,
) -> pd.DataFrame | None:
    """Compute all 14 features for one product/date; return per-node DataFrame."""
    subset = read_sst(nc_path, cfg, BBOX)
    if subset is None:
        return None
    lon, lat, sst, valid = subset["lon"], subset["lat"], subset["sst"], subset["valid"]

    nominal_km = cfg["nominal_km"]
    s1, _  = smooth_valid(sst, valid, cfg["g1_km"] / nominal_km)
    s2, _  = smooth_valid(sst, valid, cfg["g2_km"] / nominal_km)
    s4, _  = smooth_valid(sst, valid, cfg["g4_km"] / nominal_km)
    s2sq,_ = smooth_valid(sst**2, valid, cfg["g2_km"] / nominal_km)

    var_g2 = s2sq - s2**2
    dog_g2 = s1 - s4

    gx1, gy1, lap1 = gradient_fields(s1, lon, lat)
    gx2, gy2, _    = gradient_fields(s2, lon, lat)
    gx4, gy4, _    = gradient_fields(s4, lon, lat)

    node_lat = static["lat"].values
    node_lon = static["lon"].values

    def interp(field):
        return interp_to_nodes(lon, lat, field, node_lat, node_lon)

    df = static.copy()
    df["sst_k"]     = interp(sst)
    df["valid_raw"] = np.isfinite(df["sst_k"])
    df["sst_g1"]    = interp(s1)
    df["sst_g2"]    = interp(s2)
    df["sst_g4"]    = interp(s4)
    df["gx_g1"]     = interp(gx1)
    df["gy_g1"]     = interp(gy1)
    df["gx_g2"]     = interp(gx2)
    df["gy_g2"]     = interp(gy2)
    df["gx_g4"]     = interp(gx4)
    df["gy_g4"]     = interp(gy4)
    df["lap_g1"]    = interp(lap1)
    df["var_g2"]    = interp(var_g2)
    df["dog_g2"]    = interp(dog_g2)

    # Pass 2b: gradient projections onto transect normal/tangent
    for scale in ("g1", "g2", "g4"):
        gxc = df[f"gx_{scale}"]
        gyc = df[f"gy_{scale}"]
        gn = gxc * df["nx"] + gyc * df["ny"]
        gt = gxc * df["tx"] + gyc * df["ty"]
        df[f"gn_{scale}"]    = gn
        df[f"gt_{scale}"]    = gt
        df[f"ratio_{scale}"] = gn.abs() / (gt.abs() + 1e-6)

    # Pass 1: breakpoint regression
    df["date"]    = date_str
    df["product"] = product.lower()
    df = fit_breakpoints(df, NATIVE_GRAIN_KM[product.upper()])

    # Pass 2c: anomaly uses breakpoint-derived offshore reference
    df["anomaly_g1"] = df["bg_g1"] - df["sst_g1"]
    df["anomaly_g2"] = df["bg_g2"] - df["sst_g2"]
    df["anomaly_g4"] = df["bg_g4"] - df["sst_g4"]

    # valid_z: breakpoint accepted AND all 14 canonical features present
    df["valid_z"] = (
        df["bp_accepted"].eq(1)
        & df[CANONICAL_FEATURES].notna().all(axis=1)
    )

    df.drop(columns=["gx_g1", "gy_g1", "gx_g2", "gy_g2", "gx_g4", "gy_g4",
                      "sst_g1", "sst_g2", "sst_g4",
                      "gt_g1", "gt_g2", "gt_g4",
                      "nx", "ny", "tx", "ty",
                      "frac_200", "frac_500"],
            errors="ignore", inplace=True)
    return df


# ── Date index ────────────────────────────────────────────────────────────────

def index_nc_files(cfg: dict) -> dict[date, str]:
    """Return {date: filepath} for all files in the product directory."""
    result: dict[date, str] = {}
    import glob as _glob
    pattern = os.path.join(cfg["dir"], cfg["glob"])
    for path in _glob.glob(pattern):
        m = DATE_RE.search(os.path.basename(path))
        if m:
            d = datetime.strptime(m.group(1), "%Y%m%d").date()
            result[d] = path
    return result


def season_dates(year: int, season: str) -> list[date]:
    """All dates in a (year, season).  DJF crosses year boundary."""
    months = SEASONS[season]
    days: list[date] = []
    for m in months:
        y = year - 1 if (season == "DJF" and m == 12) else year
        for d in range(1, monthrange(y, m)[1] + 1):
            days.append(date(y, m, d))
    return sorted(days)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Smoke-test mode ───────────────────────────────────────────────────────────

def smoke_test(
    product: str,
    target_date: date,
    mesh_tag: str,
    out_fig_dir: str,
) -> None:
    """Run the full pipeline on a single day and print the smoke-test report."""
    import time
    import tracemalloc

    cfg = PRODUCT_CFG[product]
    nc_index = index_nc_files(cfg)
    nc_path = nc_index.get(target_date)
    if nc_path is None:
        sys.exit(f"No file found for {product} {target_date}.")

    nodes, bnd, iso = load_mesh(mesh_tag)
    bnd    = build_transect_geometry(bnd)
    static = build_static_features(nodes, bnd, iso)

    t0 = time.perf_counter()
    tracemalloc.start()

    df = features_one_day(nc_path, cfg, static,
                          date_str=target_date.isoformat(),
                          product=product)
    if df is None:
        sys.exit("features_one_day returned None.")

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall = time.perf_counter() - t0

    # ── Item 1: n_eff after native-pixel binning ──
    neff_td = (df[["col_id", "date", "n_eff"]]
               .drop_duplicates(subset=["col_id", "date"]))
    neff_vals = neff_td["n_eff"].values.astype(float)

    print(f"\n{'='*60}")
    print(f"SMOKE TEST  {product}  {target_date}  "
          f"(native grain {NATIVE_GRAIN_KM[product.upper()]} km)")
    print("="*60)
    print(f"\n[1] n_eff after binning (per transect):  "
          f"median={int(np.median(neff_vals))}  "
          f"P5={int(np.percentile(neff_vals, 5))}  "
          f"P95={int(np.percentile(neff_vals, 95))}")

    # ── Item 2: acceptance statistics ──
    bp_td = (df[["col_id", "date", "bp_accepted"]]
             .drop_duplicates(subset=["col_id", "date"]))
    n_total      = len(bp_td)
    n_accepted   = int(bp_td["bp_accepted"].sum())
    n_unaccepted = n_total - n_accepted
    frac         = n_accepted / n_total if n_total else 0.0

    print(f"\n[2] Acceptance: {n_accepted}/{n_total} transects accepted "
          f"({frac:.1%}), {n_unaccepted} unaccepted")

    if frac < 0.50:
        print("WARNING: acceptance fraction below 50%. "
              "Check parameters or implementation before proceeding.")

    # ── Item 3: five Luderitz-region transects ──
    # Luderitz is at ~26.6 S, which maps to coast_s_km ~ 1850-1960 km
    luderitz_mask = df["coast_s_km"].between(1850, 1960)
    lu_bp = (df[luderitz_mask]
             .drop_duplicates(subset=["col_id"])
             [["col_id", "coast_s_km", "beta_inshore", "beta_offshore",
               "x_b_km", "bp_accepted", "bg_g1"]]
             .sort_values("coast_s_km"))
    lu_bp = lu_bp.copy()
    lu_bp["slope_ratio"] = (lu_bp["beta_inshore"].abs()
                            / lu_bp["beta_offshore"].abs().clip(lower=1e-9))
    lu_5 = lu_bp.head(5)

    slope_thresh = BP_MIN_SLOPE_RATIO
    print(f"\n[3] Luderitz-region transect breakpoint fits "
          f"(5 transects; slope-ratio threshold = {slope_thresh}):")
    print(f"  {'col_id':>8}  {'coast_s_km':>10}  {'beta_in (K/km)':>14}  "
          f"{'beta_off (K/km)':>15}  {'x_b_km':>8}  {'slope_ratio':>11}  {'accepted':>8}")
    for _, row in lu_5.iterrows():
        sr_str = f"{row['slope_ratio']:>11.2f}" if np.isfinite(row['slope_ratio']) else f"{'—':>11}"
        print(f"  {int(row['col_id']):>8}  {row['coast_s_km']:>10.1f}  "
              f"{row['beta_inshore']:>14.4f}  {row['beta_offshore']:>15.4f}  "
              f"{row['x_b_km']:>8.1f}  {sr_str}  {int(row['bp_accepted']):>8}")

    # ── Item 5: T_hat_bg vs SIGMA_BG baseline ──
    # bg_g1/bg_g2/bg_g4 are kept in the output; compute baseline by temporarily
    # re-reading sst_g1 equivalent from anomaly inversion: sst_g1 = bg_g1 - anomaly_g1
    # Baseline bg: median of nodes where sigma > SIGMA_BG.
    # We reconstruct approximate sst_g1 as bg_g1 - anomaly_g1 for accepted transects.
    df["sst_g1_approx"] = df["bg_g1"] - df["anomaly_g1"]
    bg_base_mask = df["sigma"] > SIGMA_BG
    bg_base = (
        df[bg_base_mask & df["sst_g1_approx"].notna()]
        .groupby("col_id")["sst_g1_approx"]
        .median()
        .rename("bg_baseline_g1")
        .reset_index()
    )
    lu_compare = lu_5[["col_id", "coast_s_km", "bg_g1", "x_b_km"]].merge(
        bg_base, on="col_id", how="left"
    )
    lu_compare["delta_bg"] = lu_compare["bg_g1"] - lu_compare["bg_baseline_g1"]

    print("\n[5] T_hat_bg breakpoint vs SIGMA_BG=0.70 baseline (5 Luderitz transects):")
    print(f"  {'col_id':>8}  {'T_bg_bp (K)':>11}  {'T_bg_base (K)':>13}  "
          f"{'delta (K)':>9}  {'x_b_km':>8}  {'x_b sigma':>9}")
    for _, row in lu_compare.iterrows():
        grp = df[df["col_id"] == row["col_id"]]
        ll   = grp["line_length_km"].iloc[0] if len(grp) else np.nan
        xb_sigma = row["x_b_km"] / ll if (np.isfinite(ll) and ll > 0
                                           and np.isfinite(row["x_b_km"])) else np.nan
        print(f"  {int(row['col_id']):>8}  {row['bg_g1']:>11.3f}  "
              f"{row['bg_baseline_g1']:>13.3f}  "
              f"{row['delta_bg']:>9.3f}  {row['x_b_km']:>8.1f}  {xb_sigma:>9.3f}")
    df.drop(columns=["sst_g1_approx"], inplace=True)

    # ── Item 6: diagnostic plot ──
    _plot_smoke_diagnostics(df, product, target_date.isoformat(),
                            out_fig_dir, lu_5)
    _plot_rejection_sample(df, product, target_date.isoformat(), out_fig_dir)

    # ── Item 7: resources ──
    print(f"\n[7] Wall-clock: {wall:.1f} s  |  Peak memory: {peak_mem/1e6:.0f} MB")
    print("="*60 + "\n")


def _plot_smoke_diagnostics(
    df: pd.DataFrame, product: str, date_str: str,
    out_fig_dir: str, lu_5: pd.DataFrame,
) -> None:
    """Save diagnostic plots for one accepted and one unaccepted Luderitz transect."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[4] matplotlib not available; skipping diagnostic plot.")
        return

    os.makedirs(out_fig_dir, exist_ok=True)

    accepted_ids   = lu_5.loc[lu_5["bp_accepted"] == 1, "col_id"].values
    unaccepted_ids = lu_5.loc[lu_5["bp_accepted"] == 0, "col_id"].values

    chosen_pairs: list[tuple[int, str]] = []
    if len(accepted_ids):
        chosen_pairs.append((int(accepted_ids[0]), "accepted"))
    if len(unaccepted_ids):
        chosen_pairs.append((int(unaccepted_ids[0]), "unaccepted"))
    if not chosen_pairs:
        chosen_pairs = [(int(lu_5["col_id"].iloc[0]), "first"),
                        (int(lu_5["col_id"].iloc[-1]), "last")]

    fig, axes = plt.subplots(1, len(chosen_pairs),
                             figsize=(6 * len(chosen_pairs), 5),
                             squeeze=False)

    # anomaly_g1 is available; reconstruct approximate sst profile from it if
    # needed.  However, we stored anomaly = bg - sst, so sst = bg - anomaly.
    for ax, (col_id, label) in zip(axes[0], chosen_pairs):
        grp = df[df["col_id"] == col_id].sort_values("distance_km")
        bg  = grp["bg_g1"].iloc[0]
        if np.isfinite(bg):
            sst_approx = bg - grp["anomaly_g1"]
        else:
            sst_approx = pd.Series(np.full(len(grp), np.nan))

        ax.scatter(grp["distance_km"].values, sst_approx.values,
                   s=8, alpha=0.6, label="sst_g1 (approx)", color="steelblue")

        x_b = grp["x_b_km"].iloc[0]
        if np.isfinite(x_b):
            ax.axvline(x_b, color="red", lw=1.2, linestyle="--",
                       label=f"x_b={x_b:.0f} km")
            beta_in  = grp["beta_inshore"].iloc[0]
            beta_off = grp["beta_offshore"].iloc[0]
            x_arr    = grp["distance_km"].values
            sst_arr  = sst_approx.values
            x_in  = x_arr[x_arr < x_b]
            y_in  = sst_arr[x_arr < x_b]
            x_off = x_arr[x_arr >= x_b]
            y_off = sst_arr[x_arr >= x_b]
            if len(x_in) > 1 and np.isfinite(beta_in):
                _, icpt = _ols_params(x_in[np.isfinite(y_in)],
                                      y_in[np.isfinite(y_in)])
                ax.plot(x_in, icpt + beta_in * x_in,
                        color="orange", lw=1.5, label="inshore OLS")
            if len(x_off) > 1 and np.isfinite(beta_off):
                _, icpt = _ols_params(x_off[np.isfinite(y_off)],
                                      y_off[np.isfinite(y_off)])
                ax.plot(x_off, icpt + beta_off * x_off,
                        color="green", lw=1.5, label="offshore OLS")

        ax.set_xlabel("Distance from coast (km)")
        ax.set_ylabel("SST g1 (K)")
        ax.set_title(f"col_id={col_id}  ({label})")
        ax.legend(fontsize=7)

    fig.suptitle(f"{product}  {date_str}  Luderitz diagnostic", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(out_fig_dir, f"bp_smoke_neff_{product}_{date_str}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[4] Diagnostic plot saved: {out_path}")


def _plot_rejection_sample(
    df: pd.DataFrame, product: str, date_str: str,
    out_fig_dir: str,
) -> None:
    """Save a sibling diagnostic plot showing unaccepted transect profiles.

    Searches for candidates from the tAUS warm-water sector (coast_s_km < 640,
    no upwelling gradient expected) and from the northernmost equatorial-margin
    transects.  Plots raw sst_k vs distance_km (sst_g1 is not retained after
    features_one_day returns).  The sigma=0.70 reference line is marked so the
    reader can judge whether a knee exists near the fixed-threshold boundary.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[4b] matplotlib not available; skipping rejection diagnostic plot.")
        return

    # Count valid (non-NaN) nodes per transect to exclude cloud-only rejections
    valid_counts = df.groupby("col_id")["valid_raw"].sum().rename("n_valid")

    td = (df.drop_duplicates(subset=["col_id"])
          .merge(valid_counts, on="col_id", how="left"))

    # Only unaccepted transects that had enough data for the fitter to attempt a fit;
    # purely cloud-covered transects (n_valid == 0) are uninteresting for diagnostics.
    unac = (td[(td["bp_accepted"] == 0) & (td["n_valid"] >= 6)]
            .sort_values("coast_s_km"))
    if unac.empty:
        print("[4b] All unaccepted transects are cloud-only (n_valid < 6); "
              "skipping rejection diagnostic plot.")
        return

    # Candidate 1: tAUS sector with data (flat warm water, gradient-free expected)
    taus = unac[unac["coast_s_km"] < 640]
    # Candidate 2: BUS transect with data that still failed criteria (weak gradient)
    bus  = unac[unac["coast_s_km"] >= 640]

    chosen: list[tuple[int, str]] = []
    if not taus.empty:
        mid = taus.iloc[len(taus) // 2]
        chosen.append((int(mid["col_id"]),
                       f"tAUS  coast_s={mid['coast_s_km']:.0f} km  "
                       f"n_valid={int(mid['n_valid'])}"))
    if not bus.empty:
        row = bus.iloc[len(bus) // 2]
        cid = int(row["col_id"])
        if cid not in [c for c, _ in chosen]:
            chosen.append((cid,
                           f"BUS (weak gradient)  coast_s={row['coast_s_km']:.0f} km  "
                           f"n_valid={int(row['n_valid'])}"))
    if not chosen:
        chosen.append((int(unac.iloc[0]["col_id"]),
                       f"unaccepted  coast_s={unac.iloc[0]['coast_s_km']:.0f} km"))
        if len(unac) > 1:
            chosen.append((int(unac.iloc[-1]["col_id"]),
                           f"unaccepted  coast_s={unac.iloc[-1]['coast_s_km']:.0f} km"))

    fig, axes = plt.subplots(1, len(chosen),
                             figsize=(6 * len(chosen), 5), squeeze=False)

    for ax, (col_id, label) in zip(axes[0], chosen):
        grp = df[df["col_id"] == col_id].sort_values("distance_km")
        valid = grp["valid_raw"].astype(bool)

        ax.scatter(grp.loc[valid, "distance_km"].values,
                   grp.loc[valid, "sst_k"].values,
                   s=10, alpha=0.7, color="steelblue", label="sst_k (raw)")
        if (~valid).any():
            yref = grp.loc[valid, "sst_k"].mean() if valid.any() else np.nan
            ax.scatter(grp.loc[~valid, "distance_km"].values,
                       np.full((~valid).sum(), yref),
                       s=20, marker="x", color="grey", alpha=0.5, label="NaN (domain edge)")

        # Mark sigma=0.70 position
        near_sig = grp.iloc[(grp["sigma"] - 0.70).abs().argsort().iloc[:1]]
        if not near_sig.empty:
            ax.axvline(near_sig["distance_km"].values[0], color="purple",
                       lw=0.9, linestyle=":", alpha=0.8, label="σ=0.70")

        ax.set_xlabel("Distance from coast (km)")
        ax.set_ylabel("SST (K, raw sst_k)")
        ax.set_title(f"col_id={col_id}  ({label})\nstatus: unaccepted")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"{product}  {date_str}  rejection sample\n"
        f"(sst_k plotted; g1-smoothed SST used in fitting is not retained)",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = os.path.join(out_fig_dir,
                            f"bp_smoke_neff_{product}_{date_str}_unaccepted.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[4b] Rejection diagnostic plot saved: {out_path}")


# ── Main processing ────────────────────────────────────────────────────────────

def process_product_year(
    product: str,
    year: int,
    seasons_to_run: list[str],
    mesh_tag: str,
    dry_run: bool,
    overwrite: bool,
) -> None:
    cfg = PRODUCT_CFG[product]
    out_dir = os.path.join(FEAT_BP_ROOT, product)
    os.makedirs(out_dir, exist_ok=True)

    pending = []
    for season in seasons_to_run:
        out_path = os.path.join(out_dir, f"{product}_{year}_{season}.parquet")
        if os.path.exists(out_path) and not overwrite:
            print(f"  skip {product} {year} {season} (exists)")
        else:
            pending.append(season)
    if not pending:
        return

    if dry_run:
        for season in pending:
            print(f"  would build {product} {year} {season}")
        return

    print(f"\n{'='*60}")
    print(f"  {product}  {year}  mesh={mesh_tag}")
    nodes, bnd, iso = load_mesh(mesh_tag)
    bnd    = build_transect_geometry(bnd)
    static = build_static_features(nodes, bnd, iso)
    nc_index = index_nc_files(cfg)

    for season in pending:
        dates = season_dates(year, season)
        out_path  = os.path.join(out_dir, f"{product}_{year}_{season}.parquet")
        meta_path = out_path.replace(".parquet", "_meta.json")

        daily_frames: list[pd.DataFrame] = []
        n_ok = n_miss = n_fail = 0
        input_shas: list[str] = []
        bp_accepted_total = 0
        bp_total = 0
        n_eff_vals: list[int] = []

        print(f"\n  {season} ({len(dates)} days)")
        for d in dates:
            nc_path = nc_index.get(d)
            if nc_path is None:
                n_miss += 1
                continue

            frame = features_one_day(nc_path, cfg, static,
                                     date_str=d.isoformat(),
                                     product=product)
            if frame is None:
                n_fail += 1
                continue

            # Accumulate acceptance and n_eff stats
            td = frame[["col_id", "date", "bp_accepted", "n_eff"]].drop_duplicates(
                subset=["col_id", "date"])
            bp_total          += len(td)
            bp_accepted_total += int(td["bp_accepted"].sum())
            n_eff_vals.extend(td["n_eff"].tolist())

            daily_frames.append(frame)
            input_shas.append(_sha256(nc_path))
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"    {n_ok}/{len(dates)} days done ...", flush=True)

        if not daily_frames:
            print(f"  WARNING: no valid days for {product} {year} {season}")
            continue

        out_df = pd.concat(daily_frames, ignore_index=True)

        id_cols   = ["col_id", "sigma", "lon", "lat", "coast_s_km",
                      "coast_lon", "coast_lat", "distance_km", "line_length_km"]
        bathy     = ["d_200_km", "d_500_km"]
        bp_cols   = ["bp_accepted", "bp_segment"] + BP_FEATURES + \
                    ["bg_g1", "bg_g2", "bg_g4", "n_eff"] + BP_DIAGNOSTIC_FEATURES
        feat_cols = ["sst_k", "valid_raw"] + SST_FEATURES + ["valid_z"]
        meta_c    = ["product", "date"]
        keep = [c for c in id_cols + bathy + bp_cols + feat_cols + meta_c
                if c in out_df.columns]
        out_df = out_df[keep]

        tmp = out_path + ".tmp"
        out_df.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)

        accept_frac  = bp_accepted_total / bp_total if bp_total else 0.0
        neff_arr     = np.array(n_eff_vals, dtype=float)
        neff_stats   = (
            {"median": int(np.median(neff_arr)),
             "p5":     int(np.percentile(neff_arr,  5)),
             "p95":    int(np.percentile(neff_arr, 95))}
            if neff_arr.size else {"median": None, "p5": None, "p95": None}
        )
        meta = {
            "product": product,
            "year": year,
            "season": season,
            "mesh_tag": mesh_tag,
            "rebuild_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_dates_target": len(dates),
            "n_dates_ok": n_ok,
            "n_dates_missing": n_miss,
            "n_dates_failed": n_fail,
            "n_rows": len(out_df),
            "canonical_features": CANONICAL_FEATURES,
            "breakpoint_diagnostic_features": BP_DIAGNOSTIC_FEATURES,
            "native_grain_km": NATIVE_GRAIN_KM[product.upper()],
            "n_eff_per_transect": neff_stats,
            "bp_criteria": {
                "delta_aic_min":      DELTA_AIC_MIN,
                "bp_min_offshore_km": BP_MIN_OFFSHORE_KM,
                "bp_min_slope_ratio": BP_MIN_SLOPE_RATIO,
                "bp_edge_buffer_km":  BP_EDGE_BUFFER_KM,
            },
            "bp_transect_days_total": bp_total,
            "bp_transect_days_accepted": bp_accepted_total,
            "bp_acceptance_fraction": round(accept_frac, 4),
            "input_sha256": input_shas,
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"  -> {os.path.basename(out_path)}  "
              f"({n_ok} days, {len(out_df):,} rows, "
              f"accept {accept_frac:.1%}, "
              f"{n_miss} missing, {n_fail} failed)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--product", required=True,
                        choices=list(PRODUCT_CFG.keys()),
                        help="SST product key (case-sensitive).")
    parser.add_argument("--year",    type=int, choices=TARGET_YEARS, default=None,
                        help="Year to process (required unless --date is used).")
    parser.add_argument("--season",  choices=list(SEASONS.keys()), default=None,
                        help="Single season (default: all four).")
    parser.add_argument("--date",    default=None,
                        help="Single date (YYYY-MM-DD) for smoke-test mode; "
                             "supersedes --year and --season.")
    parser.add_argument("--mesh-tag", default=None,
                        help="FMM mesh tag.  Defaults to the most recent in MESH_ROOT.")
    parser.add_argument("--fig-dir",  default="data/outputs/uc2d_v2/figures",
                        help="Directory for smoke-test diagnostic plots.")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be done; write nothing.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output parquets.")
    args = parser.parse_args()

    # Resolve mesh tag
    if args.mesh_tag:
        mesh_tag = args.mesh_tag
    else:
        candidates = [d for d in os.listdir(MESH_ROOT)
                      if d.startswith("fmm_uc2d_v2") and
                      os.path.isdir(os.path.join(MESH_ROOT, d, "mesh_curvilinear"))]
        if not candidates:
            sys.exit(f"No uc2d_v2 mesh found under {MESH_ROOT}. "
                     "Pass --mesh-tag explicitly.")
        mesh_tag = sorted(candidates)[-1]
        print(f"Auto-selected mesh: {mesh_tag}")

    # Smoke-test mode
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            sys.exit("--date must be YYYY-MM-DD.")
        smoke_test(args.product, target_date, mesh_tag, args.fig_dir)
        return

    if args.year is None:
        sys.exit("--year is required unless --date is given.")

    seasons = [args.season] if args.season else list(SEASONS.keys())
    process_product_year(
        product=args.product,
        year=args.year,
        seasons_to_run=seasons,
        mesh_tag=mesh_tag,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
