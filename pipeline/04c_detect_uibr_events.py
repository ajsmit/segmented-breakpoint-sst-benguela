#!/usr/bin/env python3
"""Stage 4c: detect SST breakpoint active days and annual UI_BR phenology."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _uidx_common import DIAG_DIR, OUTPUT_ROOT, PRODUCTS, ensure_output_dirs, years_for, write_json


DAILY_ROOT = OUTPUT_ROOT / "uibr_daily"
OUT_ROOT = OUTPUT_ROOT / "uibr_phenology"
DAILY_CLASS_OUT = OUT_ROOT / "daily_classification.parquet"
PHENO_OUT = OUT_ROOT / "per_cell_product_year.parquet"
SUMMARY_CSV = DIAG_DIR / "stage4c_uibr_phenology_summary.csv"
META_JSON = OUT_ROOT / "stage4c_meta.json"


def load_daily() -> pd.DataFrame:
    parts = []
    for product in PRODUCTS:
        for year in years_for(product):
            path = DAILY_ROOT / product / f"{product}_{year}.parquet"
            if path.exists():
                parts.append(pd.read_parquet(path))
    if not parts:
        raise FileNotFoundError(f"No Stage 4b daily parquets found under {DAILY_ROOT}")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["lamont_year"] = np.where(df["date"].dt.month >= 7, df["date"].dt.year, df["date"].dt.year - 1)
    return df


def bridge_single_quiet_day(active: pd.Series, max_gap: int) -> pd.Series:
    if max_gap < 1:
        return active.copy()
    out = active.copy()
    vals = active.to_numpy(dtype=bool)
    for i in range(1, len(vals) - 1):
        if not vals[i] and vals[i - 1] and vals[i + 1]:
            out.iloc[i] = True
    return out


def event_stats(active: pd.Series, min_len: int) -> tuple[int, float]:
    starts = active & ~active.shift(fill_value=False)
    event_ids = starts.cumsum().where(active, np.nan)
    lengths = event_ids.dropna().value_counts()
    valid_lengths = lengths[lengths >= min_len]
    if valid_lengths.empty:
        return 0, float("nan")
    return int(len(valid_lengths)), float(valid_lengths.mean())


def classify_group(group: pd.DataFrame, p_threshold: float, min_event_days: int, max_bridge_gap: int) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    if "cell" not in g.columns or "product" not in g.columns:
        cell, product = group.name
        g["cell"] = cell
        g["product"] = product
    g["is_active_raw"] = g["p_active"] >= p_threshold
    g["is_active"] = bridge_single_quiet_day(g["is_active_raw"], max_bridge_gap)
    g["active_rule"] = f"p_active_ge_{p_threshold:g}_bridge_{max_bridge_gap}_min_{min_event_days}"
    return g


def summarize_year(group: pd.DataFrame, min_event_days: int) -> pd.Series:
    g = group.sort_values("date")
    active = g["is_active"].astype(bool)
    event_count, mean_duration = event_stats(active, min_event_days)
    active_g = g[active]
    n_active = int(active.sum())
    n_active_with_A = int(active_g["A_in_daily_K"].notna().sum())
    tcu_applicable = n_active_with_A > 0
    tcu = float(active_g["A_in_daily_K"].sum(skipna=True)) if tcu_applicable else float("nan")
    return pd.Series(
        {
            "n_days": int(g["date"].nunique()),
            "NUD_BR": n_active,
            "NUE_BR": event_count,
            "mean_event_duration_BR": mean_duration,
            "TCU_BR": tcu,
            "tcu_br_applicable": bool(tcu_applicable),
            "n_active_days_with_A_in": n_active_with_A,
            "n_active_days_missing_A_in": n_active - n_active_with_A,
            "active_A_in_coverage": (n_active_with_A / n_active) if n_active else float("nan"),
            "A_in_daily_mean_active_K": float(active_g["A_in_daily_K"].mean(skipna=True)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-active-threshold", type=float, default=0.5)
    parser.add_argument("--min-event-days", type=int, default=3)
    parser.add_argument("--max-bridge-gap", type=int, default=1)
    args = parser.parse_args()

    ensure_output_dirs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    daily = load_daily()
    classified = (
        daily.groupby(["cell", "product"], group_keys=False)
        .apply(
            classify_group,
            p_threshold=args.p_active_threshold,
            min_event_days=args.min_event_days,
            max_bridge_gap=args.max_bridge_gap,
        )
        .reset_index(drop=True)
    )

    classified.to_parquet(DAILY_CLASS_OUT, index=False)
    phenology = (
        classified.groupby(["cell", "product", "lamont_year"], dropna=False)
        .apply(summarize_year, min_event_days=args.min_event_days, include_groups=False)
        .reset_index()
        .rename(columns={"lamont_year": "year"})
    )
    phenology["year_definition"] = "lamont_july_to_june"
    phenology["complete_lamont_year"] = phenology["n_days"] >= 365
    phenology["p_active_threshold"] = args.p_active_threshold
    phenology["min_event_days"] = args.min_event_days
    phenology["max_bridge_gap"] = args.max_bridge_gap
    phenology.to_parquet(PHENO_OUT, index=False)
    phenology.to_csv(SUMMARY_CSV, index=False)
    write_json(
        META_JSON,
        {
            "stage": "4c",
            "script": "pipeline/04c_detect_uibr_events.py",
            "daily_classification_output": str(DAILY_CLASS_OUT),
            "phenology_output": str(PHENO_OUT),
            "tcu_br_applicable": "False where no active day has valid A_in_daily_K; TCU_BR is NaN in those cases.",
            "mw_oi_rule": "MW_OI remains included for NUD_BR/NUE_BR and daily classification, but TCU_BR is non-applicable where A_in is unavailable.",
        },
    )
    print(f"Wrote {len(classified):,} classified daily rows -> {DAILY_CLASS_OUT}")
    print(f"Wrote {len(phenology):,} phenology rows -> {PHENO_OUT}")
    print(f"Wrote diagnostics -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
