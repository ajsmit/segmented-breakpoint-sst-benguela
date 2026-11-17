#!/usr/bin/env python3
"""paperA_common.py -- shared paths, constants, and helpers for paperA scripts.

Addendum task: Stage 1, Task 1.3 (expose parameter-count constants in
the paperA namespace; centralise the r2 mesh tag and cell-membership
file so downstream paperA scripts do not import from the upstream
_uidx_common.py with its hard-coded lost-mesh tag).

Use:
    from paperA_common import (
        K_ONE_SEGMENT, K_TWO_SEGMENT, DELTA_AIC_BAR,
        PRODUCTS, YEARS, SEASONS, CELLS,
        MESH_TAG, mesh_root, feature_root,
        paperA_cell_transects_path,
    )
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# ---- AIC parameter counts ----------------------------------------------------
# Burnham & Anderson (2002, "Model Selection and Multimodel Inference",
# Springer, 2nd ed.) derive the "considerable evidence" bar
# Delta AIC >= 2 * (Delta k) = 2 * (K_TWO_SEGMENT - K_ONE_SEGMENT) = 4.
# The two-segment count is K_TWO_SEGMENT = 4 because the offshore intercept
# is determined by continuity (alpha_off = alpha_in + beta_in * x_b) and is
# therefore derived, not fitted. The one-segment count is K_ONE_SEGMENT = 2
# (intercept and slope). See upstream/build_feature_stack.py
# function-level docstring at line 407 for the matching upstream definition.
K_ONE_SEGMENT = 2
K_TWO_SEGMENT = 4
DELTA_AIC_BAR = 4.0


# ---- Mesh and paths ----------------------------------------------------------
MESH_TAG = "fmm_uc2d_v2_outer500km_v2026_05_21_r2"

MESH_ROOT_CANDIDATES = (
    ROOT / "data/derived/analysis_runs" / MESH_TAG / "mesh_curvilinear",
)
FEATURE_ROOT_CANDIDATES = (
    ROOT / "data/outputs/uc2d_v2/features_bp",
)
ERA5_ROOT_CANDIDATES = (
    ROOT / "data/raw/era5",
)


PAPERA_ROOT = ROOT / "results"
PAPERA_SCRIPTS = ROOT / "analysis"
PAPERA_AUDIT = PAPERA_ROOT / "audit"
PAPERA_DIAGNOSTICS = PAPERA_ROOT / "diagnostics"
PAPERA_FIGURES = PAPERA_ROOT / "figures"
PAPERA_TABLES = PAPERA_ROOT / "tables"


# ---- Product / cell / season constants ---------------------------------------
PRODUCTS = ("OSTIA", "OSPO", "AVHRR", "MW_OI")
YEARS = tuple(range(2015, 2026))
SEASONS = ("DJF", "MAM", "JJA", "SON")
REFERENCE_DISTANCES_KM = (50.0, 100.0, 200.0, 500.0)


# Cell column-header strings use the ASCII "Luderitz" spelling (matching
# canonical_cell_windows.csv and the v1 archive's per_cell_summary.parquet).
# The diacritic spelling "Lüderitz" is used only in paper-facing strings
# (figure titles, table headers), not in identifier columns.
CELLS = ("Cape Frio", "Walvis Bay", "Luderitz", "Cape Columbine", "Cape Peninsula")


CELL_LATEX = {
    "Cape Frio":       "Cape Frio",
    "Walvis Bay":      "Walvis Bay",
    "Luderitz":        r"L\"uderitz",
    "Cape Columbine":  "Cape Columbine",
    "Cape Peninsula":  "Cape Peninsula",
}


# Cell membership keyed by the r2 mesh. Regenerated in Task 0.6 from
# canonical_cell_windows.csv x mesh_boundary_match.csv (membership rule:
# coast_s_km in [coast_s_km_min, coast_s_km_max]). 189 transects total.
PAPERA_CELL_TRANSECTS = PAPERA_AUDIT / "paperA_cell_transects.csv"


# ---- Helpers ----------------------------------------------------------------
def first_existing(candidates: tuple[Path, ...], label: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No {label} path exists: {', '.join(map(str, candidates))}"
    )


def mesh_root() -> Path:
    return first_existing(MESH_ROOT_CANDIDATES, "mesh")


def feature_root() -> Path:
    return first_existing(FEATURE_ROOT_CANDIDATES, "features_bp")


def era5_root() -> Path:
    return first_existing(ERA5_ROOT_CANDIDATES, "ERA5 wind")


def feature_path(product: str, year: int, season: str) -> Path:
    return feature_root() / product / f"{product}_{year}_{season}.parquet"


def years_for(product: str) -> list[int]:
    """Operational years per product (AVHRR starts 2016; the rest 2015)."""
    if product == "AVHRR":
        return list(range(2016, 2026))
    return list(YEARS)


def paperA_cell_transects_path() -> Path:
    if not PAPERA_CELL_TRANSECTS.exists():
        raise FileNotFoundError(
            f"Missing {PAPERA_CELL_TRANSECTS}. "
            "Run analysis/paperA_v1_table1_reproduction.py "
            "to regenerate (Task 0.6)."
        )
    return PAPERA_CELL_TRANSECTS


def ensure_paperA_dirs() -> None:
    for path in (PAPERA_SCRIPTS, PAPERA_AUDIT, PAPERA_DIAGNOSTICS,
                 PAPERA_FIGURES, PAPERA_TABLES):
        path.mkdir(parents=True, exist_ok=True)


# ---- Column-rename shim (Task 7.1) ------------------------------------------
# v1 used the symbol A_in (anomaly) for the inshore-segment SST drawdown;
# paperA standardises on D_in (drawdown). Upstream parquets retain v1 names.
# Paper-A scripts read via this shim so the in-script vocabulary stays D_in.
COLUMN_RENAME_MAP = {
    "A_in":         "D_in",
    "A_in_daily_K": "D_in_daily_K",
    "A_in_bar_K":   "D_in_bar_K",
    "A_in_median":  "D_in_median",
    "A_in_P5":      "D_in_P5",
    "A_in_P95":     "D_in_P95",
    "A_in_valid":   "D_in_valid",
    "A_in_status":  "D_in_status",
    "A_in_v1":      "D_in_v1",
    "A_in_thin":    "D_in_thin",
}


def read_features_bp(path, **read_kwargs):
    """Read a parquet (uibr_transect, features_bp, per_cell_summary, or
    any other v1 file with an A_in family of columns) and rename A_in*
    to D_in*. All paperA scripts that read v1 parquets should use this
    helper instead of pandas.read_parquet directly.

    Imported lazily to avoid forcing pandas at module-load."""
    import pandas as pd
    df = pd.read_parquet(path, **read_kwargs)
    rename = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


__all__ = [
    "K_ONE_SEGMENT", "K_TWO_SEGMENT", "DELTA_AIC_BAR",
    "PRODUCTS", "YEARS", "SEASONS", "REFERENCE_DISTANCES_KM",
    "CELLS", "CELL_LATEX",
    "MESH_TAG", "PAPERA_ROOT", "PAPERA_SCRIPTS", "PAPERA_AUDIT",
    "PAPERA_DIAGNOSTICS", "PAPERA_FIGURES", "PAPERA_TABLES",
    "PAPERA_CELL_TRANSECTS",
    "ROOT",
    "first_existing", "mesh_root", "feature_root", "era5_root",
    "feature_path", "years_for",
    "paperA_cell_transects_path", "ensure_paperA_dirs",
]
