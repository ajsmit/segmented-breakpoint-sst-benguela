#!/usr/bin/env python3
"""paperA_phenology_filter.py -- Stage 7 Task 7.2: partial July-to-June-year
exclusion helper for paperA phenology aggregations.

A July-to-June year is defined as 1 July N to 30 June N+1; with the
SST window 2015-01-01 to 2025-12-31, only years 2015 through 2024 are
complete (the year starting in 2014 is missing Jul–Dec 2014; the year
starting in 2025 is missing Jan–Jun 2026).

The upstream Stage 3 already filters the per-cell-per-year wind
phenology table to complete Lamont years (see
pipeline/03_construct_uiw_lamont.py:261-262). This helper:

  - Loads the upstream `per_cell_per_year.parquet` and surfaces the
    complete July-to-June-year set.
  - Provides a `complete_lamont_years()` accessor that downstream
    paperA scripts can use to filter daily / seasonal aggregates.
  - Prints a summary; emits a small CSV audit log.

Outputs:
  results/diagnostics/complete_lamont_years.csv

Run from the release repository root:
    python3 analysis/paperA_phenology_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


WIND_PER_YEAR = P.ROOT / "data/outputs/uidx/uiw_lamont/per_cell_per_year.parquet"
OUT_CSV = P.PAPERA_ROOT / "diagnostics/complete_lamont_years.csv"


def complete_lamont_years() -> list[int]:
    """Return the sorted list of complete July-to-June years across all sectors.

    A complete July-to-June year is one where every sector has 365+ daily wind
    samples (`complete_lamont_year` == True in
    uiw_lamont/per_cell_per_year.parquet). For the operational period
    2015-01-01 through 2025-12-31, complete July-to-June years are 2015-2024."""
    df = pd.read_parquet(WIND_PER_YEAR)
    if "complete_lamont_year" not in df.columns:
        raise RuntimeError(
            f"{WIND_PER_YEAR} missing complete_lamont_year column; "
            "re-run pipeline/03_construct_uiw_lamont.py first."
        )
    complete = df[df["complete_lamont_year"]]["year"].unique()
    return sorted(int(y) for y in complete)


def partial_lamont_years() -> list[int]:
    """The partial July-to-June years excluded from per-year aggregates."""
    return [2014, 2025]


def filter_to_complete(df: pd.DataFrame, year_col: str = "lamont_year") -> pd.DataFrame:
    """Filter a per-(cell, lamont_year, ...) DataFrame to only complete years."""
    complete = set(complete_lamont_years())
    return df[df[year_col].isin(complete)].reset_index(drop=True)


def main() -> None:
    complete = complete_lamont_years()
    partial = partial_lamont_years()
    print(f"[upstream] {WIND_PER_YEAR}")
    print(f"[complete July-to-June years] ({len(complete)}): {complete}")
    print(f"[partial July-to-June years]  ({len(partial)}): {partial}")
    print(f"  - 2014 = 2014-07-01 to 2015-06-30 (no SST coverage before 2015-01-01)")
    print(f"  - 2025 = 2025-07-01 to 2026-06-30 (no SST coverage after 2025-12-31)")

    # Audit CSV
    rows = []
    for y in complete:
        rows.append({"lamont_year": y, "status": "complete",
                     "covers": f"{y}-07-01 to {y+1}-06-30"})
    for y in partial:
        rows.append({"lamont_year": y, "status": "partial",
                     "covers": (f"{y}-07-01 to {y+1}-06-30  "
                                f"(missing {'Jul-Dec ' + str(y) if y == 2014 else 'Jan-Jun ' + str(y+1)})")})
    out = pd.DataFrame(rows).sort_values("lamont_year").reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[output] {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
