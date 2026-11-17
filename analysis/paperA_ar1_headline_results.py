#!/usr/bin/env python3
"""paperA_ar1_headline_results.py -- AR(1)-corrected headline results.

This script makes the AR(1)-corrected acceptance set the primary analysis
set for the paperA headline results. It is deliberately local to paperA:
it reads persisted paperA intermediates and model-evidence files, then
writes manuscript-facing diagnostics without editing upstream pipelines.

Outputs:
  data/outputs/uidx/paperA_ar1/per_profile_flags.parquet
  data/outputs/uidx/paperA_ar1/per_cell_product_day.parquet
  data/outputs/uidx/paperA_ar1/fixed_distance_profiles.parquet
  results/tables/table1_per_cell_summary.csv
  results/tables/table_s8_fixed_distance_ar1.csv
  results/tables/table_s5_paired_classifier_ar1.csv
  results/tables/fig4_four_regime_stacked_bar.csv
  results/tables/table_s9_block_bootstrap_ar1.csv
  results/diagnostics/ar1_headline_summary.txt

Run from the ucxd repository root:
  python3 analysis/paperA_ar1_headline_results.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P
import paperA_model_evidence as M


PHI_CSV = P.PAPERA_ROOT / "diagnostics/ar1_residuals.csv"
INTERMEDIATE = P.ROOT / "data/outputs/uidx/paperA_sensitivity/per_profile_intermediate.parquet"
MODEL_ROOT = P.ROOT / "data/outputs/uidx/paperA_model_evidence"
WIND_DAILY = P.ROOT / "data/outputs/uidx/uiw_lamont/daily_cell_wind.parquet"

AR1_ROOT = P.ROOT / "data/outputs/uidx/paperA_ar1"
FLAGS_PARQUET = AR1_ROOT / "per_profile_flags.parquet"
DAILY_PARQUET = AR1_ROOT / "per_cell_product_day.parquet"
FIXED_PROFILE_PARQUET = AR1_ROOT / "fixed_distance_profiles.parquet"

OUT_TABLE1 = P.PAPERA_TABLES / "table1_per_cell_summary.csv"
OUT_FIXED = P.PAPERA_TABLES / "table_s8_fixed_distance_ar1.csv"
OUT_PAIRED = P.PAPERA_TABLES / "table_s5_paired_classifier_ar1.csv"
OUT_REGIME = P.PAPERA_TABLES / "fig4_four_regime_stacked_bar.csv"
OUT_BOOT = P.PAPERA_TABLES / "table_s9_block_bootstrap_ar1.csv"
OUT_SUMMARY = P.PAPERA_DIAGNOSTICS / "ar1_headline_summary.txt"

D0_VALUES = (50, 100, 200, 500)
FINE_PRODUCTS = ("OSTIA", "OSPO")
P_ACTIVE_THRESHOLD = 0.5
N_BOOT = 1000
BLOCK_KM_VALUES = (25.0, 50.0)
COAST_SPACING_KM = 3.7


def load_phi() -> dict[tuple[str, str], float]:
    phi = pd.read_csv(PHI_CSV)
    return {(r.cell, r.season): float(r.phi) for r in phi.itertuples(index=False)}


def load_model_neff() -> pd.DataFrame:
    phi_map = load_phi()
    parts = []
    cols = [
        "cell", "product", "year", "season", "date", "col_id", "n_eff",
        "bp_accepted", "rss_one_segment_K2", "rss_two_segment_K2",
    ]
    for product in P.PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                fp = MODEL_ROOT / product / f"{product}_{year}_{season}.parquet"
                if not fp.exists():
                    continue
                df = pd.read_parquet(fp, columns=cols)
                df["phi"] = df.apply(
                    lambda r: phi_map.get((r["cell"], r["season"]), np.nan),
                    axis=1,
                )
                df["n_eff_star"] = df["n_eff"] * (1.0 - df["phi"]) / (1.0 + df["phi"])
                n_star = df["n_eff_star"].to_numpy(dtype=float)
                r1 = df["rss_one_segment_K2"].to_numpy(dtype=float)
                r2 = df["rss_two_segment_K2"].to_numpy(dtype=float)
                df["delta_aic_ar1"] = (
                    M.aic(n_star, r1, P.K_ONE_SEGMENT)
                    - M.aic(n_star, r2, P.K_TWO_SEGMENT)
                )
                parts.append(df[[
                    "cell", "product", "year", "season", "date", "col_id",
                    "phi", "n_eff_star", "delta_aic_ar1",
                ]])
    out = pd.concat(parts, ignore_index=True)
    out["date_key"] = out["date"].astype(str)
    return out


def build_flags(force: bool = False) -> pd.DataFrame:
    if FLAGS_PARQUET.exists() and not force:
        return pd.read_parquet(FLAGS_PARQUET)

    print(f"[src] {INTERMEDIATE}")
    inter = pd.read_parquet(INTERMEDIATE)
    inter["date_key"] = inter["date"].astype(str)
    print(f"[intermediate] {len(inter):,} rows")

    model = load_model_neff()
    print(f"[model neff] {len(model):,} rows")

    key = ["cell", "product", "year", "season", "date_key", "col_id"]
    flags = inter.merge(
        model.drop(columns="date"),
        on=key,
        how="left",
        validate="one_to_one",
    )
    flags["accepted_baseline"] = (
        (flags["bp_accepted"] == 1) & (flags["retained_bool"] == 1)
    ).astype(np.int8)
    flags["accepted_ar1"] = (
        (flags["accepted_baseline"] == 1)
        & (flags["delta_aic_ar1"] >= P.DELTA_AIC_BAR)
    ).astype(np.int8)
    flags["date"] = pd.to_datetime(flags["date"])
    AR1_ROOT.mkdir(parents=True, exist_ok=True)
    flags.to_parquet(FLAGS_PARQUET, index=False)
    print(f"[output] {FLAGS_PARQUET}")
    return flags


def build_daily(flags: pd.DataFrame) -> pd.DataFrame:
    den = flags[flags["retained_bool"] == 1].copy()
    daily = den.groupby(["cell", "product", "year", "season", "date"], as_index=False).agg(
        n_retained=("col_id", "nunique"),
        n_accepted_baseline=("accepted_baseline", "sum"),
        n_accepted_ar1=("accepted_ar1", "sum"),
    )
    daily["p_active_baseline"] = daily["n_accepted_baseline"] / daily["n_retained"]
    daily["p_active_ar1"] = daily["n_accepted_ar1"] / daily["n_retained"]
    daily["sst_active_ar1"] = (daily["p_active_ar1"] >= P_ACTIVE_THRESHOLD).astype(np.int8)
    AR1_ROOT.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PARQUET, index=False)
    print(f"[output] {DAILY_PARQUET}")
    return daily


def nearest_ui_for_file(product: str, year: int, season: str,
                        accepted: pd.DataFrame) -> pd.DataFrame:
    fp = P.feature_root() / product / f"{product}_{year}_{season}.parquet"
    if not fp.exists() or accepted.empty:
        return pd.DataFrame()

    col_ids = sorted(accepted["col_id"].unique().astype(int).tolist())
    keys = set(zip(accepted["date_key"], accepted["col_id"].astype(int)))
    tbl = pq.read_table(
        fp,
        columns=["col_id", "date", "distance_km", "sst_k", "beta_inshore"],
        filters=[("col_id", "in", col_ids)],
    ).to_pandas()
    if tbl.empty:
        return pd.DataFrame()
    tbl["date_key"] = tbl["date"].astype(str)
    tbl = tbl[[k in keys for k in zip(tbl["date_key"], tbl["col_id"].astype(int))]]
    if tbl.empty:
        return pd.DataFrame()

    valid = tbl[np.isfinite(tbl["sst_k"])].copy()
    valid["d_abs_0"] = valid["distance_km"].abs()
    t_coast = valid.loc[
        valid.groupby(["col_id", "date_key"])["d_abs_0"].idxmin(),
        ["col_id", "date_key", "sst_k"],
    ].set_index(["col_id", "date_key"])["sst_k"]

    ui_parts = []
    for d0 in D0_VALUES:
        valid["d_abs"] = (valid["distance_km"] - float(d0)).abs()
        t_d0 = valid.loc[
            valid.groupby(["col_id", "date_key"])["d_abs"].idxmin(),
            ["col_id", "date_key", "sst_k"],
        ].set_index(["col_id", "date_key"])["sst_k"]
        ui_parts.append((t_d0 - t_coast).rename(f"ui_sst_d{d0}"))
    ui = pd.concat(ui_parts, axis=1).reset_index()

    beta = tbl.drop_duplicates(["col_id", "date_key"], keep="first")[
        ["col_id", "date_key", "beta_inshore"]
    ]
    out = accepted.merge(ui, on=["col_id", "date_key"], how="inner")
    out = out.merge(beta, on=["col_id", "date_key"], how="left")
    return out


def build_fixed_distance_profiles(flags: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    if FIXED_PROFILE_PARQUET.exists() and not force:
        return pd.read_parquet(FIXED_PROFILE_PARQUET)

    rows = []
    for product in FINE_PRODUCTS:
        for year in P.years_for(product):
            for season in P.SEASONS:
                acc = flags[
                    (flags["product"] == product)
                    & (flags["year"] == year)
                    & (flags["season"] == season)
                    & (flags["accepted_ar1"] == 1)
                ][["cell", "product", "year", "season", "date", "date_key", "col_id",
                  "D_in", "x_b_km"]]
                t0 = time.monotonic()
                part = nearest_ui_for_file(product, year, season, acc)
                if not part.empty:
                    rows.append(part)
                print(f"  [fixed {product} {year} {season}] {len(part):>6,} rows "
                      f"{time.monotonic() - t0:.1f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out.to_parquet(FIXED_PROFILE_PARQUET, index=False)
    print(f"[output] {FIXED_PROFILE_PARQUET}")
    return out


def pearson_row(g: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, int]:
    x = g[x_col].to_numpy(dtype=float)
    y = g[y_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 3:
        return (np.nan, np.nan, n)
    r, p = stats.pearsonr(x[mask], y[mask])
    return (float(r), float(p), n)


def bh_adjust(p: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return out
    q = valid * m / np.arange(1, m + 1)
    q = q.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    out.loc[q.index] = q
    return out


def fixed_distance_summary(prof: pd.DataFrame) -> pd.DataFrame:
    cell_day = prof.groupby(["cell", "product", "season", "date"], as_index=False).agg(
        D_in_med=("D_in", "median"),
        **{f"ui_d{d0}": (f"ui_sst_d{d0}", "median") for d0 in D0_VALUES},
    )
    rows = []
    for period, group_cols in [
        ("season", ["cell", "product", "season"]),
        ("annual", ["cell", "product"]),
    ]:
        for keys, g in cell_day.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_cols, keys))
            base["period"] = period
            if period == "annual":
                base["season"] = "ANN"
            for d0 in D0_VALUES:
                r, p, n = pearson_row(g, "D_in_med", f"ui_d{d0}")
                rows.append({**base, "d0_km": d0, "r": r, "p_raw": p, "n_days": n})
    out = pd.DataFrame(rows)
    mask40 = (
        (out["period"] == "season")
        & (out["product"].isin(FINE_PRODUCTS))
        & (out["d0_km"] == 50)
    )
    out["q_bh_40test"] = np.nan
    out.loc[mask40, "q_bh_40test"] = bh_adjust(out.loc[mask40, "p_raw"])
    out = out.sort_values(["period", "cell", "product", "season", "d0_km"])
    out.to_csv(OUT_FIXED, index=False, float_format="%.6g")
    print(f"[output] {OUT_FIXED}")
    return out


def table1_ar1(prof: pd.DataFrame) -> pd.DataFrame:
    per_ys = prof.groupby(["cell", "product", "year", "season"], as_index=False).agg(
        D_in_C=("D_in", "median"),
        x_b_km=("x_b_km", "median"),
        beta_inshore_C_per_km=("beta_inshore", "median"),
        n_profiles=("col_id", "size"),
    )
    climo = per_ys.groupby(["cell", "product"], as_index=False).agg(
        D_in_C=("D_in_C", "median"),
        x_b_km=("x_b_km", "median"),
        beta_inshore_C_per_km=("beta_inshore_C_per_km", "median"),
        n_year_seasons=("n_profiles", "size"),
        n_profiles=("n_profiles", "sum"),
    )
    # Keep the four-product table shape, but the coarse products do not pass AR(1).
    all_rows = pd.MultiIndex.from_product([P.CELLS, P.PRODUCTS],
                                          names=["cell", "product"]).to_frame(index=False)
    out = all_rows.merge(climo, on=["cell", "product"], how="left")
    out.to_csv(OUT_TABLE1, index=False, float_format="%.6g")
    print(f"[output] {OUT_TABLE1}")
    return out


def confusion_stats(sst: np.ndarray, wind: np.ndarray) -> dict[str, float]:
    s = sst.astype(bool)
    w = wind.astype(bool)
    tp = int(np.sum(s & w))
    tn = int(np.sum(~s & ~w))
    fp = int(np.sum(s & ~w))
    fn = int(np.sum(~s & w))
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    p_o = (tp + tn) / n if n else np.nan
    p_e = (((tp + fn) / n) * ((tp + fp) / n)
           + ((tn + fp) / n) * ((tn + fn) / n)) if n else np.nan
    kappa = (p_o - p_e) / (1.0 - p_e) if np.isfinite(p_e) and (1.0 - p_e) > 1e-12 else np.nan
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den > 0 else np.nan
    return {
        "n_days": n,
        "prevalence_sst": float(s.mean()) if n else np.nan,
        "prevalence_wind": float(w.mean()) if n else np.nan,
        "balanced_accuracy": np.nanmean([sens, spec]),
        "sensitivity": sens,
        "specificity": spec,
        "mcc": mcc,
        "p_o": p_o,
        "cohen_kappa": kappa,
    }


def paired_classifier_ar1(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wind = pd.read_parquet(WIND_DAILY)
    wind["date"] = pd.to_datetime(wind["date"])
    wind["wind_active"] = (wind["UI_w_m3_s_per_100m"] > 0.0).astype(np.int8)
    wind = wind[["cell", "date", "season", "wind_active", "UI_w_m3_s_per_100m"]]
    paired_daily = daily.merge(wind, on=["cell", "date", "season"], how="inner")

    paired_rows = []
    regime_rows = []
    for (cell, product, season), g in paired_daily.groupby(["cell", "product", "season"]):
        st = confusion_stats(g["sst_active_ar1"].to_numpy(), g["wind_active"].to_numpy())
        st.update(cell=cell, product=product, season=season)
        paired_rows.append(st)

        s = g["sst_active_ar1"].astype(bool)
        w = g["wind_active"].astype(bool)
        n = len(g)
        regime_rows.append({
            "cell": cell,
            "product": product,
            "season": season,
            "n_days": n,
            "concordant_active": float((s & w).sum() / n),
            "concordant_quiet": float(((~s) & (~w)).sum() / n),
            "sst_only": float((s & (~w)).sum() / n),
            "wind_only": float(((~s) & w).sum() / n),
            "coverage_artefact": bool(cell in ("Cape Frio", "Cape Peninsula") and product == "AVHRR"),
        })

    paired = pd.DataFrame(paired_rows)
    regime = pd.DataFrame(regime_rows)
    paired.to_csv(OUT_PAIRED, index=False, float_format="%.6g")
    regime.to_csv(OUT_REGIME, index=False, float_format="%.6g")
    print(f"[output] {OUT_PAIRED}")
    print(f"[output] {OUT_REGIME}")
    return paired_daily, paired, regime


def build_blocks(cell_transects: pd.DataFrame, block_km: float) -> dict[str, list[list[int]]]:
    n_per_block = max(1, int(round(block_km / COAST_SPACING_KM)))
    out = {}
    for cell, g in cell_transects.groupby("cell"):
        col_ids = sorted(g["col_id"].tolist())
        out[cell] = [col_ids[i:i + n_per_block] for i in range(0, len(col_ids), n_per_block)]
    return out


def resample_col_ids(blocks: list[list[int]], target_n: int,
                     rng: np.random.Generator) -> list[int]:
    chosen = []
    while len(chosen) < target_n:
        chosen.extend(blocks[int(rng.integers(0, len(blocks)))])
    return chosen[:target_n]


def luderitz_r50_ar1(prof: pd.DataFrame, col_ids: list[int]) -> float:
    sub = prof[
        (prof["cell"] == "Luderitz")
        & (prof["product"] == "OSTIA")
        & (prof["col_id"].isin(col_ids))
    ]
    if sub.empty:
        return np.nan
    cell_day = sub.groupby("date").agg(
        D_in_med=("D_in", "median"),
        ui_d50=("ui_sst_d50", "median"),
    ).dropna()
    if len(cell_day) < 30:
        return np.nan
    return float(np.corrcoef(cell_day["D_in_med"], cell_day["ui_d50"])[0, 1])


def sbus_sst_only_ar1(flags: pd.DataFrame, wind: pd.DataFrame,
                      resampled: dict[str, list[int]]) -> float:
    parts = []
    for cell in ("Cape Columbine", "Cape Peninsula"):
        for product in FINE_PRODUCTS:
            sub = flags[
                (flags["cell"] == cell)
                & (flags["product"] == product)
                & (flags["retained_bool"] == 1)
                & (flags["col_id"].isin(resampled[cell]))
            ]
            if sub.empty:
                continue
            d = sub.groupby(["cell", "product", "season", "date"], as_index=False).agg(
                n_retained=("col_id", "nunique"),
                n_accepted=("accepted_ar1", "sum"),
            )
            d["sst_active"] = ((d["n_accepted"] / d["n_retained"]) >= P_ACTIVE_THRESHOLD).astype(np.int8)
            parts.append(d[["cell", "product", "season", "date", "sst_active"]])
    if not parts:
        return np.nan
    d = pd.concat(parts, ignore_index=True)
    m = d.merge(wind[["cell", "date", "season", "wind_active"]],
                on=["cell", "date", "season"], how="inner")
    if m.empty:
        return np.nan
    return float(((m["sst_active"] == 1) & (m["wind_active"] == 0)).sum() / len(m))


def block_bootstrap_ar1(flags: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    membership = pd.read_csv(P.paperA_cell_transects_path())
    wind = pd.read_parquet(WIND_DAILY)
    wind["date"] = pd.to_datetime(wind["date"])
    wind["wind_active"] = (wind["UI_w_m3_s_per_100m"] > 0.0).astype(np.int8)
    wind = wind[["cell", "date", "season", "wind_active"]]

    rows = []
    for block_km in BLOCK_KM_VALUES:
        blocks = build_blocks(membership, block_km)
        target_n = {cell: len(membership[membership["cell"] == cell]) for cell in blocks}
        rng = np.random.default_rng(20260525)
        vals_l = []
        vals_s = []
        for i in range(N_BOOT):
            sampled = {cell: resample_col_ids(blocks[cell], target_n[cell], rng)
                       for cell in blocks}
            vals_l.append(luderitz_r50_ar1(prof, sampled["Luderitz"]))
            vals_s.append(sbus_sst_only_ar1(flags, wind, sampled))
        for stat_name, vals in [("L_r50_ar1", vals_l), ("sBUS_sst_only_ar1", vals_s)]:
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            rows.append({
                "stat": stat_name,
                "block_km": block_km,
                "n_boot": len(arr),
                "boot_p025": float(np.percentile(arr, 2.5)),
                "boot_p500": float(np.percentile(arr, 50.0)),
                "boot_p975": float(np.percentile(arr, 97.5)),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_BOOT, index=False, float_format="%.6g")
    print(f"[output] {OUT_BOOT}")
    return out


def write_summary(table1: pd.DataFrame, fixed: pd.DataFrame,
                  paired: pd.DataFrame, regime: pd.DataFrame,
                  boot: pd.DataFrame) -> None:
    lines = []
    lines.append("AR(1)-corrected paperA headline summary")
    lines.append("")
    lines.append("Table 1 AR(1) fine-product medians:")
    for _, r in table1[table1["product"].isin(FINE_PRODUCTS)].iterrows():
        lines.append(
            f"  {r['cell']} {r['product']}: "
            f"D_in={r['D_in_C']:.3g}, x_b={r['x_b_km']:.1f}, "
            f"beta={r['beta_inshore_C_per_km']:.4g}, n={int(r['n_profiles'])}"
        )
    lines.append("")
    for product in FINE_PRODUCTS:
        pooled = fixed[(fixed["period"] == "annual") & (fixed["product"] == product)]
        for d0 in D0_VALUES:
            sub = pooled[pooled["d0_km"] == d0]
            # Pooled across cells is not in fixed; compute from the figure profiles if needed.
            lines.append(f"Annual per-cell r values available for {product} d0={d0} in {OUT_FIXED.name}.")
    l_ann = fixed[
        (fixed["period"] == "annual")
        & (fixed["cell"] == "Luderitz")
        & (fixed["product"] == "OSTIA")
        & (fixed["d0_km"] == 50)
    ].iloc[0]
    cf_son = fixed[
        (fixed["period"] == "season")
        & (fixed["cell"] == "Cape Frio")
        & (fixed["product"] == "OSTIA")
        & (fixed["season"] == "SON")
        & (fixed["d0_km"] == 50)
    ].iloc[0]
    lines.append("")
    lines.append(
        f"Luderitz OSTIA annual r50 AR1: r={l_ann.r:.3f}, n={int(l_ann.n_days)}"
    )
    lines.append(
        f"Cape Frio OSTIA SON r50 AR1: r={cf_son.r:.3f}, n={int(cf_son.n_days)}, "
        f"p={cf_son.p_raw:.3g}, q_BH={cf_son.q_bh_40test:.3g}"
    )
    sbus = regime[
        regime["cell"].isin(["Cape Columbine", "Cape Peninsula"])
        & regime["product"].isin(FINE_PRODUCTS)
    ]
    lines.append("")
    lines.append(
        "sBUS AR1 SST-structure-only fraction across cell/product/season rows: "
        f"min={sbus['sst_only'].min():.3f}, max={sbus['sst_only'].max():.3f}, "
        f"mean={sbus['sst_only'].mean():.3f}"
    )
    lines.append("")
    lines.append(boot.to_string(index=False))
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text("\n".join(lines) + "\n")
    print(f"[output] {OUT_SUMMARY}")


def main() -> None:
    t0 = time.monotonic()
    flags = build_flags(force=True)
    daily = build_daily(flags)
    prof = build_fixed_distance_profiles(flags, force=True)
    fixed = fixed_distance_summary(prof)
    table1 = table1_ar1(prof)
    paired_daily, paired, regime = paired_classifier_ar1(daily)
    boot = block_bootstrap_ar1(flags, prof)
    write_summary(table1, fixed, paired, regime, boot)
    print(f"\n[done] {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
