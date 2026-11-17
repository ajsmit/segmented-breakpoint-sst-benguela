#!/usr/bin/env python3
"""Shared paths and helpers for the uidx active scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds


MESH_TAG = "fmm_uc2d_v2_outer500km_v2026_05_21_r2"
FEATURE_ROOT_CANDIDATES = (
    Path("data/outputs/uc2d_v2/features_bp"),
)
MESH_ROOT_CANDIDATES = (
    # Repo-local canonical copy first (verified, volume-independent; see
    # data/outputs/versioned/<tag>/mesh_curvilinear/REBUILD_PROVENANCE.md).
    Path("data/outputs/versioned") / MESH_TAG / "mesh_curvilinear",
    Path("data/derived/analysis_runs") / MESH_TAG / "mesh_curvilinear",
)
ERA5_ROOT_CANDIDATES = (
    Path("data/raw/era5"),
)

UIDX_ROOT = Path("results")
UIDX_DATA_ACCESS = UIDX_ROOT / "data_access"
OUTPUT_ROOT = Path("data/outputs/uidx")
FIGURE_DIR = OUTPUT_ROOT / "figures"
DIAG_DIR = OUTPUT_ROOT / "diagnostics"

PRODUCTS = ("OSTIA", "OSPO", "AVHRR", "MW_OI")
YEARS = range(2015, 2026)
SEASONS = ("DJF", "MAM", "JJA", "SON")
REFERENCE_DISTANCES_KM = (50.0, 100.0, 200.0, 500.0)

CELL_DEFS = (
    {
        "cell": "Cape Frio",
        "lat_centre": -18.4,
        "sub_system": "nBUS",
        "source_citation": "LutjeharmsMeeuwis1987; Hutchings2009; KampfChapman2016",
    },
    {
        "cell": "Walvis Bay",
        "lat_centre": -23.0,
        "sub_system": "nBUS",
        "source_citation": "LutjeharmsMeeuwis1987; Hutchings2009; KampfChapman2016",
    },
    {
        "cell": "Luderitz",
        "lat_centre": -26.5,
        "sub_system": "nBUS / divide",
        "source_citation": "LutjeharmsMeeuwis1987; Hutchings2009; KampfChapman2016",
    },
    {
        "cell": "Cape Columbine",
        "lat_centre": -32.8,
        "sub_system": "sBUS",
        "source_citation": "LutjeharmsMeeuwis1987; Hutchings2009; KampfChapman2016",
    },
    {
        "cell": "Cape Peninsula",
        "lat_centre": -34.0,
        "sub_system": "sBUS",
        "source_citation": "LutjeharmsMeeuwis1987; Hutchings2009; KampfChapman2016",
    },
)


def first_existing(candidates: tuple[Path, ...], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {label} path exists: {', '.join(map(str, candidates))}")


def feature_root() -> Path:
    return first_existing(FEATURE_ROOT_CANDIDATES, "features_bp")


def mesh_root() -> Path:
    return first_existing(MESH_ROOT_CANDIDATES, "mesh")


def era5_root() -> Path:
    return first_existing(ERA5_ROOT_CANDIDATES, "ERA5 wind")


def ensure_output_dirs() -> None:
    for path in (UIDX_DATA_ACCESS, OUTPUT_ROOT, FIGURE_DIR, DIAG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def feature_path(product: str, year: int, season: str) -> Path:
    return feature_root() / product / f"{product}_{year}_{season}.parquet"


def years_for(product: str) -> list[int]:
    return list(range(2016, 2026)) if product == "AVHRR" else list(YEARS)


def load_canonical_transects() -> pd.DataFrame:
    path = UIDX_DATA_ACCESS / "canonical_cell_transects.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 00_define_windows_verify_inputs.py first."
        )
    return pd.read_csv(path)


def load_coastal_transects_from_features() -> pd.DataFrame:
    """Read one coastal row per transect from a representative feature parquet."""
    candidates = sorted((feature_root() / "OSTIA").glob("OSTIA_*_JJA.parquet"))
    if not candidates:
        candidates = sorted(feature_root().glob("*/*_JJA.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No feature parquets found under {feature_root()}")

    dataset = ds.dataset(str(candidates[0]), format="parquet")
    table = dataset.to_table(
        columns=["col_id", "sigma", "coast_s_km", "coast_lon", "coast_lat"],
        filter=pc.field("sigma") == 0,
    )
    df = table.to_pandas()
    df = (
        df.drop_duplicates("col_id")
        .rename(columns={"coast_lon": "lon", "coast_lat": "lat"})
        .sort_values("coast_s_km")
        .reset_index(drop=True)
    )
    return df[["col_id", "coast_s_km", "lat", "lon"]]


def read_filtered_parquet(path: Path, columns: list[str], col_ids: list[int]) -> pd.DataFrame:
    dataset = ds.dataset(str(path), format="parquet")
    table = dataset.to_table(columns=columns, filter=pc.field("col_id").isin(col_ids))
    return table.to_pandas()


def add_season(date_series: pd.Series) -> pd.Series:
    months = pd.to_datetime(date_series).dt.month
    return months.map(
        {
            12: "DJF",
            1: "DJF",
            2: "DJF",
            3: "MAM",
            4: "MAM",
            5: "MAM",
            6: "JJA",
            7: "JJA",
            8: "JJA",
            9: "SON",
            10: "SON",
            11: "SON",
        }
    )


def write_json(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
