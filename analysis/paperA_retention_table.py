#!/usr/bin/env python3
"""paperA_retention_table.py -- Stage 2 Task 2.3: retention rate per
(cell, product) on the deduplicated set.

Addendum task: Stage 2, Task 2.3 (retention rate per (cell, product)).

Reads the Task 2.2 thinned tree at data/outputs/uidx/paperA_thinned/
and reports four retention metrics per (cell, product), aggregated over
the operational period and broken out by season:

  retention_pct_nominal      = (accepted AND retained) / nominal
  retention_pct_accepted     = (accepted AND retained) / accepted
  pixel_retention_pct        = mean of (n_unique / total) over accepted
  cell_day_pixel_retention   = sum(n_unique) / sum(total) over the period

The "nominal" denominator includes both bp_accepted = 0 and bp_accepted = 1
rows (every cell-window col_id-day in the operational period). The manuscript's
"ca. 74 %" prose at paperA_manuscript.tex line 221 refers to the
retention_pct_nominal interpretation; we report all four so the prose
update has the right number whatever the reader's preferred metric.

Output: results/tables/table_s_pixel_stride_retention.csv

Run from the release repository root:
    python3 analysis/paperA_retention_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


SRC_ROOT = P.ROOT / "data/outputs/uidx/paperA_thinned"
OUT_CSV = P.PAPERA_ROOT / "tables/table_s_pixel_stride_retention.csv"
OUT_LATEX = P.PAPERA_ROOT / "tables/table_s_pixel_stride_retention.tex"


def load_all() -> pd.DataFrame:
    parts = []
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                fp = SRC_ROOT / product / f"{product}_{year}_{season}.parquet"
                if not fp.exists():
                    continue
                df = pd.read_parquet(fp)
                df["year"] = year
                df["season"] = season
                parts.append(df)
    return pd.concat(parts, ignore_index=True)


def compute_metrics(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    by = df.groupby(group)
    nominal = by.size().rename("n_nominal")
    accepted_count = by["accepted"].sum().rename("n_accepted")
    retained_and_accepted = by.apply(
        lambda g: int(((g["accepted"] == 1) & (g["retained_bool"] == 1)).sum()),
        include_groups=False,
    ).rename("n_retained_accepted")
    total_pix = by["total_pixels_in_transect"].sum().rename("sum_total_pixels")
    unique_pix = by["n_unique_pixels"].sum().rename("sum_unique_pixels")

    pixel_retention_per_transect = by.apply(
        lambda g: float(
            (g.loc[g["accepted"] == 1, "n_unique_pixels"]
             / g.loc[g["accepted"] == 1, "total_pixels_in_transect"].replace(0, np.nan)
             ).mean()
        ),
        include_groups=False,
    ).rename("pixel_retention_per_transect")

    out = pd.concat(
        [nominal, accepted_count, retained_and_accepted, total_pix, unique_pix,
         pixel_retention_per_transect],
        axis=1,
    ).reset_index()
    out["retention_pct_nominal"] = 100.0 * out["n_retained_accepted"] / out["n_nominal"]
    out["retention_pct_accepted"] = 100.0 * out["n_retained_accepted"] / out["n_accepted"]
    out["cell_day_pixel_retention_pct"] = 100.0 * out["sum_unique_pixels"] / out["sum_total_pixels"]
    out["pixel_retention_per_transect_pct"] = 100.0 * out["pixel_retention_per_transect"]
    return out.drop(columns=["pixel_retention_per_transect"])


def main() -> None:
    print(f"[src] {SRC_ROOT}")
    df = load_all()
    print(f"[loaded] {len(df):,} (cell, product, year, season, date, col_id) rows\n")

    overall = compute_metrics(df, ["cell", "product"])
    by_season = compute_metrics(df, ["cell", "product", "season"])

    # Wide-format season breakdown for the overall row
    season_wide = by_season.pivot_table(
        index=["cell", "product"],
        columns="season",
        values="retention_pct_nominal",
    ).reset_index()
    season_wide.columns = [c if not isinstance(c, str) or c in ["cell", "product"]
                           else f"retention_{c}_pct" for c in season_wide.columns]
    overall = overall.merge(season_wide, on=["cell", "product"], how="left")
    overall = overall.sort_values(["cell", "product"]).reset_index(drop=True)

    # Round and write
    fmt_cols = [
        "retention_pct_nominal", "retention_pct_accepted",
        "cell_day_pixel_retention_pct", "pixel_retention_per_transect_pct",
        "retention_DJF_pct", "retention_MAM_pct", "retention_JJA_pct", "retention_SON_pct",
    ]
    for c in fmt_cols:
        if c in overall.columns:
            overall[c] = overall[c].round(1)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    overall.to_csv(OUT_CSV, index=False)
    print(f"[output] {OUT_CSV}")
    print(overall.to_string(index=False))

    # Per-product summary across cells (the "ca. 74 % / 21 %" comparison)
    print("\n=== Per-product retention (operational-period mean across 5 cells) ===")
    summary = overall.groupby("product")[fmt_cols[:4]].mean().round(1)
    print(summary.to_string())

    # Simple LaTeX fragment for paperA_supplement.tex
    cells_ordered = ["Cape Frio", "Walvis Bay", "Luderitz", "Cape Columbine", "Cape Peninsula"]
    products_ordered = ["OSTIA", "OSPO", "AVHRR", "MW_OI"]
    lines = [
        r"% S_PIXEL_STRIDE_RETENTION -- generated by paperA_retention_table.py",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Cell & Product & $n_\mathrm{nominal}$ & $n_\mathrm{retained,accepted}$ & "
        r"Retention (\%) & Per-transect pixel retention (\%) \\",
        r"\midrule",
    ]
    for cell in cells_ordered:
        for product in products_ordered:
            row = overall[(overall["cell"] == cell) & (overall["product"] == product)]
            if row.empty:
                continue
            r = row.iloc[0]
            cell_disp = P.CELL_LATEX.get(cell, cell)
            lines.append(
                f"{cell_disp} & {product.replace('_', r'\_')} & "
                f"{int(r['n_nominal']):,} & {int(r['n_retained_accepted']):,} & "
                f"{r['retention_pct_nominal']:.1f} & "
                f"{r['pixel_retention_per_transect_pct']:.1f} \\\\"
            )
        if cell != cells_ordered[-1]:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    OUT_LATEX.write_text("\n".join(lines) + "\n")
    print(f"\n[latex fragment] {OUT_LATEX}")


if __name__ == "__main__":
    main()
