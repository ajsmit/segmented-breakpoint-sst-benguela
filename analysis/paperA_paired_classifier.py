#!/usr/bin/env python3
"""paperA_paired_classifier.py -- Stage 5 Tasks 5.1 + 5.2 + 5.3:
paired-classifier diagnostic statistics for the SST-structure detector
vs the Bakun-Lamont wind detector.

Daily binary flags:
  SST detector  = thinned p_active >= 0.5
                  (where thinned p_active = n_retained_accepted / n_retained
                   per Task 2.4)
  wind detector = UI_w_m3_s_per_100m > 0
                  (from the Stage 3 wind index in
                   data/outputs/uidx/uiw_lamont/daily_cell_wind.parquet)

Stats per (cell, product, season):
  n_days, balanced_accuracy, sensitivity, specificity, MCC, p_o,
  cohen_kappa, prevalence_sst, prevalence_wind.

Null baseline (Task 5.2): 1000 within-season shuffles of the SST flag,
recompute the stats, record the [2.5, 97.5] percentiles and the
observed-vs-null z-score (in standard deviations of the shuffled
distribution).

Four-regime fractions (Task 5.3): per (cell, product, season) the
proportions of concordant_active / concordant_quiet / sst_only / wind_only,
plus the coverage_artefact flag for the two AVHRR coverage-artefact pairs.

Outputs:
  results/tables/table_s_paired_classifier.csv
  results/tables/table_s_four_regime.csv

Run from the release repository root:
    python3 analysis/paperA_paired_classifier.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


THIN_DAILY = P.ROOT / "data/outputs/uidx/paperA_p_active_thinned/per_cell_product_day.parquet"
WIND_DAILY = P.ROOT / "data/outputs/uidx/uiw_lamont/daily_cell_wind.parquet"
OUT_PAIRED  = P.PAPERA_ROOT / "tables/table_s_paired_classifier.csv"
OUT_REGIME  = P.PAPERA_ROOT / "tables/table_s_four_regime.csv"

P_ACTIVE_THRESHOLD = 0.5
N_SHUFFLES = 1000


# ---- Confusion-matrix metrics ------------------------------------------------
def confusion_stats(sst: np.ndarray, wind: np.ndarray) -> dict[str, float]:
    """SST = predicted; wind = reference. Returns confusion-matrix metrics."""
    s = sst.astype(bool); w = wind.astype(bool)
    tp = int(np.sum( s &  w))
    tn = int(np.sum(~s & ~w))
    fp = int(np.sum( s & ~w))
    fn = int(np.sum(~s &  w))
    n = tp + tn + fp + fn
    if n == 0:
        return dict(n_days=0, balanced_accuracy=np.nan, sensitivity=np.nan,
                    specificity=np.nan, mcc=np.nan, p_o=np.nan, cohen_kappa=np.nan,
                    prevalence_sst=np.nan, prevalence_wind=np.nan)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    bal_acc = np.nanmean([sens, spec])
    p_o = (tp + tn) / n
    p_e = (((tp + fn) / n) * ((tp + fp) / n)
           + ((tn + fp) / n) * ((tn + fn) / n))
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 1e-12 else np.nan
    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den > 0 else np.nan
    return dict(n_days=n, balanced_accuracy=bal_acc, sensitivity=sens,
                specificity=spec, mcc=mcc, p_o=p_o, cohen_kappa=kappa,
                prevalence_sst=float(s.mean()),
                prevalence_wind=float(w.mean()))


# ---- Loaders -----------------------------------------------------------------
def load_paired_daily() -> pd.DataFrame:
    """Returns one row per (cell, product, year, season, date) with
    the SST flag (thinned p_active >= 0.5) and the wind flag."""
    sst = pd.read_parquet(THIN_DAILY)
    sst["date"] = pd.to_datetime(sst["date"])
    sst["sst_active"] = (sst["p_active_thin"] >= P_ACTIVE_THRESHOLD).astype(int)
    sst = sst[["cell", "product", "year", "season", "date", "sst_active",
               "p_active_thin", "p_active_v1"]]

    wind = pd.read_parquet(WIND_DAILY)
    wind["date"] = pd.to_datetime(wind["date"])
    wind["wind_active"] = (wind["UI_w_m3_s_per_100m"] > 0.0).astype(int)
    wind = wind[["cell", "date", "season", "wind_active", "UI_w_m3_s_per_100m"]]

    daily = sst.merge(wind, on=["cell", "date", "season"], how="inner")
    return daily


# ---- Task 5.1: paired-classifier stats per (cell, product, season) ----------
def paired_classifier(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, product, season), g in daily.groupby(["cell", "product", "season"]):
        st = confusion_stats(g["sst_active"].to_numpy(), g["wind_active"].to_numpy())
        st.update(cell=cell, product=product, season=season)
        rows.append(st)
    df = pd.DataFrame(rows)
    cols = (["cell", "product", "season", "n_days",
             "prevalence_sst", "prevalence_wind",
             "balanced_accuracy", "sensitivity", "specificity",
             "mcc", "p_o", "cohen_kappa"])
    return df[cols]


# ---- Task 5.2: date-shuffled null --------------------------------------------
def shuffled_null(daily: pd.DataFrame, n_shuffles: int = N_SHUFFLES) -> pd.DataFrame:
    """Shuffle the SST flag within each (cell, product, season) and recompute
    the stats per shuffle. Return per-(cell, product, season) the
    [2.5, 97.5] percentiles of each statistic and the observed z-score."""
    rng = np.random.default_rng(0)
    rows = []
    for (cell, product, season), g in daily.groupby(["cell", "product", "season"]):
        sst = g["sst_active"].to_numpy()
        wind = g["wind_active"].to_numpy()
        observed = confusion_stats(sst, wind)
        null_stats = {k: [] for k in ("balanced_accuracy", "sensitivity",
                                       "specificity", "mcc", "p_o", "cohen_kappa")}
        for _ in range(n_shuffles):
            shuffled = sst.copy()
            rng.shuffle(shuffled)
            st = confusion_stats(shuffled, wind)
            for k in null_stats:
                null_stats[k].append(st[k])
        row = dict(cell=cell, product=product, season=season,
                   n_days=observed["n_days"], n_shuffles=n_shuffles)
        for k, vals in null_stats.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size < 10:
                row[f"{k}_observed"] = observed[k]
                row[f"{k}_null_p025"] = np.nan
                row[f"{k}_null_p500"] = np.nan
                row[f"{k}_null_p975"] = np.nan
                row[f"{k}_null_z"]    = np.nan
                continue
            mu = float(arr.mean()); sd = float(arr.std(ddof=1))
            obs = observed[k]
            row[f"{k}_observed"] = obs
            row[f"{k}_null_p025"] = float(np.percentile(arr, 2.5))
            row[f"{k}_null_p500"] = float(np.percentile(arr, 50))
            row[f"{k}_null_p975"] = float(np.percentile(arr, 97.5))
            row[f"{k}_null_z"]    = ((obs - mu) / sd) if (sd > 0 and np.isfinite(obs)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Task 5.3: four-regime fractions -----------------------------------------
def four_regime(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, product, season), g in daily.groupby(["cell", "product", "season"]):
        s = g["sst_active"].astype(bool)
        w = g["wind_active"].astype(bool)
        n = len(g)
        ca = int(np.sum( s &  w))  # concordant_active
        cq = int(np.sum(~s & ~w))  # concordant_quiet
        so = int(np.sum( s & ~w))  # sst_only
        wo = int(np.sum(~s &  w))  # wind_only
        coverage_artefact = (
            cell in ("Cape Frio", "Cape Peninsula") and product == "AVHRR"
        )
        rows.append({
            "cell": cell, "product": product, "season": season,
            "n_days": n,
            "concordant_active":  ca / n if n else np.nan,
            "concordant_quiet":   cq / n if n else np.nan,
            "sst_only":           so / n if n else np.nan,
            "wind_only":          wo / n if n else np.nan,
            "coverage_artefact":  coverage_artefact,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"[src] {THIN_DAILY}")
    print(f"[src] {WIND_DAILY}")
    daily = load_paired_daily()
    print(f"[paired daily] {len(daily):,} (cell, product, season, date) rows\n")

    # Task 5.1: paired-classifier stats
    print("=== Task 5.1: paired-classifier stats ===")
    t0 = time.monotonic()
    paired = paired_classifier(daily)
    print(f"  {len(paired)} rows in {time.monotonic() - t0:.1f}s")

    # Task 5.2: shuffled null
    print("\n=== Task 5.2: date-shuffled null (within-season; "
          f"{N_SHUFFLES} shuffles per row) ===")
    t0 = time.monotonic()
    null_df = shuffled_null(daily, N_SHUFFLES)
    print(f"  {len(null_df)} rows in {time.monotonic() - t0:.1f}s")

    # Merge paired stats + null bounds into one wide table
    paired_full = paired.merge(null_df, on=["cell", "product", "season", "n_days"],
                                how="left")
    OUT_PAIRED.parent.mkdir(parents=True, exist_ok=True)
    paired_full.to_csv(OUT_PAIRED, index=False, float_format="%.4f")
    print(f"\n[output] {OUT_PAIRED}")

    # Task 5.3: four-regime fractions
    print("\n=== Task 5.3: four-regime fractions ===")
    regime = four_regime(daily)
    OUT_REGIME.parent.mkdir(parents=True, exist_ok=True)
    regime.to_csv(OUT_REGIME, index=False, float_format="%.4f")
    print(f"[output] {OUT_REGIME}")

    # ---- Verification block ----
    print("\n=== Verification block ===")
    # Hand-check three rows
    spot = paired[paired["product"].isin(["OSTIA", "OSPO"])
                  & paired["season"].isin(["DJF", "JJA"])].copy()
    spot = spot.sort_values("n_days", ascending=False).head(3)
    print("\n  Spot-check three (cell, product, season) rows:")
    print("    " + spot[["cell", "product", "season", "n_days",
                         "p_o", "mcc", "cohen_kappa",
                         "prevalence_sst", "prevalence_wind"]].to_string(index=False))

    # Headline F3: sBUS SST-only fraction at OSTIA/OSPO in 0.25-0.32
    print("\n  Headline F3 check (sBUS SST-only fraction 0.25-0.32):")
    sbus = regime[regime["cell"].isin(["Cape Columbine", "Cape Peninsula"])
                  & regime["product"].isin(["OSTIA", "OSPO"])]
    print("    " + sbus[["cell", "product", "season", "sst_only"]].to_string(index=False))
    print(f"\n  Pooled across (sBUS cells × OSTIA/OSPO) and seasons:")
    pooled = sbus["sst_only"].mean()
    print(f"    mean sst_only = {pooled:.3f}  (manuscript: 0.25-0.32 range)")
    # Per-cell range
    print("  Per-cell range across OSTIA+OSPO seasons:")
    for cell in ("Cape Columbine", "Cape Peninsula"):
        s = sbus[sbus["cell"] == cell]["sst_only"]
        print(f"    {cell}: min {s.min():.3f}, max {s.max():.3f}, mean {s.mean():.3f}")

    # Null-baseline sanity (one row with observed MCC > null p97.5; one with not)
    print("\n  Null-baseline sanity: rows with observed MCC > null p97.5:")
    mcc_above = paired_full[paired_full["mcc_observed"] > paired_full["mcc_null_p975"]]
    mcc_below = paired_full[paired_full["mcc_observed"] <= paired_full["mcc_null_p975"]]
    print(f"    {len(mcc_above)} of {len(paired_full)} rows: observed MCC exceeds null p97.5")
    print(f"    {len(mcc_below)} of {len(paired_full)} rows: observed MCC is within / below null p97.5")

    # Hand-check confusion-matrix recomputation on first row
    g = daily[(daily["cell"] == spot.iloc[0]["cell"])
              & (daily["product"] == spot.iloc[0]["product"])
              & (daily["season"] == spot.iloc[0]["season"])]
    s = g["sst_active"].astype(bool).to_numpy(); w = g["wind_active"].astype(bool).to_numpy()
    tp = int(np.sum(s & w)); tn = int(np.sum(~s & ~w)); fp = int(np.sum(s & ~w)); fn = int(np.sum(~s & w))
    print(f"\n  Hand-check on top spot-check row "
          f"({spot.iloc[0]['cell']}, {spot.iloc[0]['product']}, {spot.iloc[0]['season']}):")
    print(f"    confusion: TP={tp}  TN={tn}  FP={fp}  FN={fn}  (n={tp+tn+fp+fn})")
    p_o_hand = (tp + tn) / (tp + tn + fp + fn)
    print(f"    p_o (hand) = {p_o_hand:.4f}  vs  spot row p_o = {spot.iloc[0]['p_o']:.4f}")


if __name__ == "__main__":
    main()
