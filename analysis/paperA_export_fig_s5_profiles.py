#!/usr/bin/env python3
"""Select and extract OSTIA profiles for Supplementary Figure S5.

Run this script on ``ocean-worker``, where the canonical ``features_bp``
parquets are available on the external volume. For each canonical cell and
for DJF and JJA, it selects one accepted profile near the population medians
of breakpoint distance and inshore slope. It also selects one rejected
profile with a finite candidate fit, favouring a simple slope-contrast failure.
Only the 20 selected profiles are written to the compact output parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds


FEATURE_ROOT = Path("data/outputs/uc2d_v2/features_bp/OSTIA")
YEARS = range(2015, 2026)
SEASONS = ("DJF", "JJA")
CELL_ORDER = ("Cape Frio", "Walvis Bay", "Luderitz", "Cape Columbine", "Cape Peninsula")
SUMMARY_COLUMNS = [
    "col_id",
    "date",
    "line_length_km",
    "bp_accepted",
    "alpha_inshore",
    "beta_inshore",
    "beta_offshore",
    "x_b_km",
    "n_eff",
    "delta_aic",
]
PROFILE_COLUMNS = ["distance_km", "sst_k", *SUMMARY_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_transects", type=Path)
    parser.add_argument("profiles_output", type=Path)
    parser.add_argument("manifest_output", type=Path)
    return parser.parse_args()


def load_population(transects: pd.DataFrame) -> pd.DataFrame:
    col_ids = transects["col_id"].astype(int).unique().tolist()
    parts: list[pd.DataFrame] = []
    for year in YEARS:
        for season in SEASONS:
            path = FEATURE_ROOT / f"OSTIA_{year}_{season}.parquet"
            dataset = ds.dataset(path, format="parquet")
            table = dataset.to_table(
                columns=SUMMARY_COLUMNS,
                filter=(pc.field("col_id").isin(col_ids)) & (pc.field("sigma") == 0),
            )
            frame = table.to_pandas()
            frame["year"] = year
            frame["season"] = season
            parts.append(frame)
            print(f"Summarised {path.name}: {len(frame):,} profiles", flush=True)
    population = pd.concat(parts, ignore_index=True)
    population["date"] = pd.to_datetime(population["date"]).dt.strftime("%Y-%m-%d")
    population = population.merge(transects[["cell", "col_id"]], on="col_id", how="inner")
    population["slope_ratio"] = (
        population["beta_inshore"].abs() / population["beta_offshore"].abs()
    )
    return population


def choose_profiles(population: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for cell in CELL_ORDER:
        for season in SEASONS:
            group = population[(population["cell"] == cell) & (population["season"] == season)]

            accepted = group[
                (group["bp_accepted"] == 1)
                & group["x_b_km"].notna()
                & group["beta_inshore"].notna()
            ].copy()
            xb_med = accepted["x_b_km"].median()
            beta_med = accepted["beta_inshore"].median()
            accepted["selection_score"] = (
                ((accepted["x_b_km"] - xb_med) / accepted["x_b_km"].std()).abs()
                + ((accepted["beta_inshore"] - beta_med) / accepted["beta_inshore"].std()).abs()
            )
            accepted_choice = accepted.sort_values(
                ["selection_score", "date", "col_id"]
            ).iloc[0].copy()
            accepted_choice["profile_status"] = "accepted"
            accepted_choice["failed_criteria"] = ""
            selected.append(accepted_choice)

            rejected = group[
                (group["bp_accepted"] == 0)
                & group["x_b_km"].notna()
                & group["delta_aic"].notna()
                & group["slope_ratio"].notna()
            ].copy()
            rejected["delta_failed"] = rejected["delta_aic"] < 4.0
            rejected["slope_failed"] = rejected["slope_ratio"] < 2.0
            rejected = rejected[rejected["delta_failed"] | rejected["slope_failed"]].copy()
            rejected["n_failed"] = rejected[["delta_failed", "slope_failed"]].sum(axis=1)
            rejected["selection_score"] = (
                rejected["slope_ratio"] - rejected["slope_ratio"].median()
            ).abs()
            rejected_choice = rejected.sort_values(
                ["n_failed", "selection_score", "date", "col_id"]
            ).iloc[0].copy()
            failures = []
            if rejected_choice["delta_failed"]:
                failures.append("Delta AIC < 4")
            if rejected_choice["slope_failed"]:
                failures.append("slope ratio < 2")
            rejected_choice["profile_status"] = "rejected"
            rejected_choice["failed_criteria"] = "; ".join(failures)
            selected.append(rejected_choice)

    manifest = pd.DataFrame(selected)
    keep = [
        "cell", "year", "season", "date", "col_id", "profile_status",
        "failed_criteria", "bp_accepted", "x_b_km", "beta_inshore",
        "beta_offshore", "slope_ratio", "delta_aic", "line_length_km",
    ]
    return manifest[keep].sort_values(
        ["cell", "season", "profile_status"], ascending=[True, True, True]
    ).reset_index(drop=True)


def print_population_summary(population: pd.DataFrame) -> None:
    """Print pooled DJF--JJA baseline quantities used to check manuscript prose."""
    rows = []
    for cell in CELL_ORDER:
        group = population[population["cell"] == cell]
        accepted = group[group["bp_accepted"] == 1]
        rejected = group[group["bp_accepted"] == 0]
        rows.append(
            {
                "cell": cell,
                "n_profiles": len(group),
                "rejection_rate": len(rejected) / len(group),
                "accepted_median_x_b_km": accepted["x_b_km"].median(),
                "accepted_median_beta_inshore": accepted["beta_inshore"].median(),
            }
        )
    print("Pooled DJF--JJA baseline population summary:", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def extract_profiles(manifest: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for (year, season), sample in manifest.groupby(["year", "season"], sort=True):
        path = FEATURE_ROOT / f"OSTIA_{int(year)}_{season}.parquet"
        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(
            columns=PROFILE_COLUMNS,
            filter=(pc.field("date").isin(sample["date"].tolist()))
            & (pc.field("col_id").isin(sample["col_id"].astype(int).tolist())),
        )
        frame = table.to_pandas()
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
        frame = frame.merge(
            sample[["cell", "date", "col_id", "profile_status", "failed_criteria"]],
            on=["date", "col_id"],
            how="inner",
            validate="many_to_one",
        )
        frame["year"] = year
        frame["season"] = season
        parts.append(frame)
        print(f"Extracted {path.name}: {len(frame):,} rows", flush=True)
    profiles = pd.concat(parts, ignore_index=True)
    profiles["sst_c"] = profiles["sst_k"] - 273.15
    return profiles


def main() -> None:
    args = parse_args()
    transects = pd.read_csv(args.canonical_transects)
    population = load_population(transects)
    print_population_summary(population)
    manifest = choose_profiles(population)
    profiles = extract_profiles(manifest)
    profiles.to_parquet(args.profiles_output, index=False)
    manifest.to_csv(args.manifest_output, index=False)
    print(manifest.to_string(index=False))
    print(f"Wrote {len(profiles):,} rows to {args.profiles_output}")


if __name__ == "__main__":
    main()
