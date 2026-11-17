#!/usr/bin/env python3
"""Stage 3: construct Bakun-Lamont wind-index phenology from ERA5 winds.

The script samples ERA5 daily 10 m winds at the five canonical uidx cells,
projects wind stress onto the same equatorward alongshore direction used by
uc2d_v2 Stage 4, and computes Lamont-style positive/offshore and
negative/onshore Ekman-transport summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from _uidx_common import (
    DIAG_DIR,
    FIGURE_DIR,
    MESH_TAG,
    OUTPUT_ROOT,
    UIDX_DATA_ACCESS,
    add_season,
    ensure_output_dirs,
    era5_root,
    mesh_root,
    write_json,
)


RHO_AIR_KG_M3 = 1.225
RHO_WATER_KG_M3 = 1025.0
C_D = 1.3e-3
OMEGA_RAD_S = 7.2921e-5
KM_PER_DEG_LAT = 111.32

DAILY_OUT = OUTPUT_ROOT / "uiw_lamont" / "daily_cell_wind.parquet"
ANNUAL_OUT = OUTPUT_ROOT / "uiw_lamont" / "per_cell_per_year.parquet"
SEASONAL_OUT = OUTPUT_ROOT / "uiw_lamont" / "per_cell_per_season_year.parquet"
SUMMARY_CSV = DIAG_DIR / "stage3_uiw_lamont_summary.csv"
SEASONAL_CSV = DIAG_DIR / "stage3_uiw_lamont_seasonal_summary.csv"
META_JSON = OUTPUT_ROOT / "uiw_lamont" / "stage3_meta.json"
FIGURE_OUT_PNG = FIGURE_DIR / "stage3_uiw_lamont_diagnostics.png"
FIGURE_OUT_PDF = FIGURE_DIR / "stage3_uiw_lamont_diagnostics.pdf"


@dataclass(frozen=True)
class Era5Day:
    path: Path
    time_index: int


def build_era5_index(root: Path) -> dict[date, Era5Day]:
    """Return {date: Era5Day} for ERA5 monthly files."""
    index: dict[date, Era5Day] = {}
    pat = re.compile(r"ERA5_wind10m_(\d{4})(\d{2})\.nc$")
    for path in sorted(root.glob("ERA5_wind10m_*.nc")):
        match = pat.match(path.name)
        if match is None:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        with nc4.Dataset(path, "r") as ds:
            n_times = len(ds.variables["valid_time"])
        start = date(year, month, 1)
        for i in range(n_times):
            index[start + timedelta(days=i)] = Era5Day(path, i)
    return index


def read_era5_day(day: Era5Day) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ascending-latitude (lat, lon, u10, v10) for one daily ERA5 slice."""
    with nc4.Dataset(day.path, "r") as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=float)
        lon = np.asarray(ds.variables["longitude"][:], dtype=float)
        u10 = np.asarray(ds.variables["u10"][day.time_index], dtype=float)
        v10 = np.asarray(ds.variables["v10"][day.time_index], dtype=float)

    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u10 = u10[::-1, :]
        v10 = v10[::-1, :]
    return lat, lon, u10, v10


def km_per_deg_lon(lat_deg: np.ndarray | float) -> np.ndarray | float:
    return KM_PER_DEG_LAT * np.cos(np.radians(lat_deg))


def load_cell_geometry() -> pd.DataFrame:
    """Cell centre plus local equatorward unit vector from mesh geometry."""
    windows = pd.read_csv(UIDX_DATA_ACCESS / "canonical_cell_windows.csv")
    transects = pd.read_csv(UIDX_DATA_ACCESS / "canonical_cell_transects.csv")
    bnd = pd.read_csv(mesh_root() / "mesh_boundary_match.csv")

    mid_lat = (bnd["coast_lat"] + bnd["outer_lat"]) / 2.0
    dlat_km = (bnd["outer_lat"] - bnd["coast_lat"]) * KM_PER_DEG_LAT
    dlon_km = (bnd["outer_lon"] - bnd["coast_lon"]) * km_per_deg_lon(mid_lat)
    dist = np.sqrt(dlat_km**2 + dlon_km**2)

    geom = bnd[["col_id"]].copy()
    nx = dlon_km / dist
    ny = dlat_km / dist
    tx = -ny
    ty = nx
    flip = np.where(ty < 0, -1.0, 1.0)
    geom["eq_x"] = tx * flip
    geom["eq_y"] = ty * flip

    merged = transects.merge(geom, on="col_id", how="left")
    vectors = (
        merged.groupby("cell", as_index=False)[["eq_x", "eq_y"]]
        .median()
        .merge(windows, on="cell", how="left")
    )
    norm = np.sqrt(vectors["eq_x"] ** 2 + vectors["eq_y"] ** 2)
    vectors["eq_x"] = vectors["eq_x"] / norm
    vectors["eq_y"] = vectors["eq_y"] / norm
    vectors["lat"] = vectors["nearest_mesh_lat"]
    vectors["lon"] = vectors["nearest_mesh_lon"]
    vectors["f_s-1"] = 2.0 * OMEGA_RAD_S * np.sin(np.radians(vectors["lat"]))
    return vectors[["cell", "lat", "lon", "eq_x", "eq_y", "f_s-1", "n_transects"]]


def bulk_stress(u10: np.ndarray, v10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speed = np.sqrt(u10**2 + v10**2)
    return RHO_AIR_KG_M3 * C_D * speed * u10, RHO_AIR_KG_M3 * C_D * speed * v10


def interpolate_field(
    lat: np.ndarray,
    lon: np.ndarray,
    field: np.ndarray,
    points: pd.DataFrame,
) -> np.ndarray:
    interp = RegularGridInterpolator(
        (lat, lon), field, method="linear", bounds_error=False, fill_value=np.nan
    )
    return interp(points[["lat", "lon"]].to_numpy())


def compute_daily_wind(
    era5_index: dict[date, Era5Day],
    cell_geom: pd.DataFrame,
    start: date,
    end: date,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    days = pd.date_range(start, end, freq="D").date
    missing = 0
    for i, d in enumerate(days, 1):
        era_day = era5_index.get(d)
        if era_day is None:
            missing += 1
            continue
        lat, lon, u10, v10 = read_era5_day(era_day)
        tau_x, tau_y = bulk_stress(u10, v10)
        day_df = cell_geom.copy()
        day_df["date"] = pd.Timestamp(d)
        day_df["u10_m_s-1"] = interpolate_field(lat, lon, u10, cell_geom)
        day_df["v10_m_s-1"] = interpolate_field(lat, lon, v10, cell_geom)
        day_df["tau_x_Pa"] = interpolate_field(lat, lon, tau_x, cell_geom)
        day_df["tau_y_Pa"] = interpolate_field(lat, lon, tau_y, cell_geom)
        day_df["tau_eq_Pa"] = (
            day_df["tau_x_Pa"] * day_df["eq_x"] + day_df["tau_y_Pa"] * day_df["eq_y"]
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            day_df["UI_w_m2_s"] = -day_df["tau_eq_Pa"] / (
                RHO_WATER_KG_M3 * day_df["f_s-1"]
            )
        day_df["UI_w_m3_s_per_100m"] = day_df["UI_w_m2_s"] * 100.0
        rows.append(day_df)
        if i % 365 == 0:
            print(f"  {i:,}/{len(days):,} days processed", flush=True)
    if missing:
        print(f"WARNING: {missing} days missing from ERA5 index")
    daily = pd.concat(rows, ignore_index=True)
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["season"] = add_season(daily["date"])
    daily["lamont_year"] = np.where(
        daily["date"].dt.month >= 7, daily["date"].dt.year, daily["date"].dt.year - 1
    )
    return daily


def count_events(values: pd.Series, positive: bool) -> int:
    if positive:
        flags = values > 0
    else:
        flags = values < 0
    starts = flags & ~flags.shift(fill_value=False)
    return int(starts.sum())


def summarize_lamont_year(group: pd.DataFrame) -> pd.Series:
    vals = group.sort_values("date")["UI_w_m3_s_per_100m"]
    valid = vals.notna()
    pos = vals.where(vals > 0, 0.0)
    neg = vals.where(vals < 0, 0.0)
    return pd.Series(
        {
            "n_days": int(valid.sum()),
            "TCU_m3_s_per_100m_days": float(pos.sum(skipna=True)),
            "TCD_m3_s_per_100m_days": float(neg.sum(skipna=True)),
            "TCU_m2_s_days": float((pos / 100.0).sum(skipna=True)),
            "TCD_m2_s_days": float((neg / 100.0).sum(skipna=True)),
            "NUD": int((vals > 0).sum()),
            "NDD": int((vals < 0).sum()),
            "NUE": count_events(vals, positive=True),
            "NDE": count_events(vals, positive=False),
            "UI_w_mean_m2_s": float(group["UI_w_m2_s"].mean(skipna=True)),
            "UI_w_median_m2_s": float(group["UI_w_m2_s"].median(skipna=True)),
        }
    )


def summarize_season(group: pd.DataFrame) -> pd.Series:
    vals = group.sort_values("date")["UI_w_m3_s_per_100m"]
    pos = vals.where(vals > 0, 0.0)
    neg = vals.where(vals < 0, 0.0)
    return pd.Series(
        {
            "n_days": int(vals.notna().sum()),
            "TCU_m3_s_per_100m_days": float(pos.sum(skipna=True)),
            "TCD_m3_s_per_100m_days": float(neg.sum(skipna=True)),
            "NUD": int((vals > 0).sum()),
            "NDD": int((vals < 0).sum()),
            "NUE": count_events(vals, positive=True),
            "NDE": count_events(vals, positive=False),
            "UI_w_mean_m2_s": float(group["UI_w_m2_s"].mean(skipna=True)),
            "UI_w_median_m2_s": float(group["UI_w_m2_s"].median(skipna=True)),
        }
    )


def write_outputs(daily: pd.DataFrame, era5_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = OUTPUT_ROOT / "uiw_lamont"
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_OUT, index=False)

    annual = (
        daily.groupby(["cell", "lamont_year"], sort=True)
        .apply(summarize_lamont_year, include_groups=False)
        .reset_index()
        .rename(columns={"lamont_year": "year"})
    )
    # Lamont years require a complete July-June window.
    annual["year_definition"] = "lamont_july_to_june"
    annual["complete_lamont_year"] = annual["n_days"] >= 365
    annual = annual[annual["complete_lamont_year"]].reset_index(drop=True)
    annual.to_parquet(ANNUAL_OUT, index=False)
    annual.to_csv(SUMMARY_CSV, index=False)

    seasonal = (
        daily.groupby(["cell", "year", "season"], sort=True)
        .apply(summarize_season, include_groups=False)
        .reset_index()
    )
    seasonal.to_parquet(SEASONAL_OUT, index=False)
    seasonal.to_csv(SEASONAL_CSV, index=False)

    write_json(
        META_JSON,
        {
            "stage": 3,
            "script": "pipeline/03_construct_uiw_lamont.py",
            "era5_dir": str(era5_dir),
            "mesh_tag": MESH_TAG,
            "daily_output": str(DAILY_OUT),
            "annual_output": str(ANNUAL_OUT),
            "seasonal_output": str(SEASONAL_OUT),
            "rho_air_kg_m3": RHO_AIR_KG_M3,
            "rho_water_kg_m3": RHO_WATER_KG_M3,
            "drag_coefficient": C_D,
            "sign_convention": "UI_w = -tau_eq / (rho_water * f); positive is offshore/upwelling-favourable in the southern hemisphere",
            "lamont_year_definition": "year N covers 1 July N to 30 June N+1",
            "event_rule": "NUE/NDE are starts of continuous positive/negative UI_w runs",
        },
    )
    return annual, seasonal


def plot_diagnostics(annual: pd.DataFrame, seasonal: pd.DataFrame) -> None:
    cell_order = annual["cell"].drop_duplicates().tolist()
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    colors = {
        "Cape Frio": "#1f77b4",
        "Walvis Bay": "#2ca02c",
        "Luderitz": "#8e44ad",
        "Cape Columbine": "#d62728",
        "Cape Peninsula": "#ff7f0e",
    }
    for cell in cell_order:
        sub = annual[annual["cell"] == cell].sort_values("year")
        axes[0].plot(sub["year"], sub["TCU_m3_s_per_100m_days"], marker="o", label=cell, color=colors.get(cell))
        axes[1].plot(sub["year"], sub["NUD"], marker="o", label=cell, color=colors.get(cell))
        axes[2].plot(sub["year"], sub["NUE"], marker="o", label=cell, color=colors.get(cell))
    axes[0].set_ylabel("TCU\n(m3 s-1 per 100 m days)")
    axes[1].set_ylabel("NUD\n(days)")
    axes[2].set_ylabel("NUE\n(runs)")
    axes[2].set_xlabel("Lamont year (Jul-Jun)")
    axes[0].legend(ncol=3, fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Stage 3 Bakun-Lamont wind phenology from ERA5")
    fig.tight_layout()
    fig.savefig(FIGURE_OUT_PNG, dpi=220)
    fig.savefig(FIGURE_OUT_PDF)
    plt.close(fig)

    jja = seasonal[(seasonal["cell"] == "Luderitz") & (seasonal["season"] == "JJA")]
    jja.to_csv(DIAG_DIR / "stage3_luderitz_jja_wind_check.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    ensure_output_dirs()
    era5_dir = era5_root()
    print(f"Indexing ERA5 under {era5_dir}")
    era5_index = build_era5_index(era5_dir)
    print(f"  {len(era5_index):,} days indexed")
    cell_geom = load_cell_geometry()
    print(f"Loaded {len(cell_geom)} canonical cell wind sample points")
    daily = compute_daily_wind(
        era5_index=era5_index,
        cell_geom=cell_geom,
        start=pd.Timestamp(args.start).date(),
        end=pd.Timestamp(args.end).date(),
    )
    annual, seasonal = write_outputs(daily, era5_dir)
    plot_diagnostics(annual, seasonal)
    print(f"Wrote {len(daily):,} daily rows -> {DAILY_OUT}")
    print(f"Wrote {len(annual):,} annual rows -> {ANNUAL_OUT}")
    print(f"Wrote diagnostics -> {SUMMARY_CSV}, {FIGURE_OUT_PNG}")


if __name__ == "__main__":
    main()
