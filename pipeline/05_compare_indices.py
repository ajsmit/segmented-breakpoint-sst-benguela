#!/usr/bin/env python3
"""Stage 5: cross-index comparison and reconciliation."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _uidx_common import (
    DIAG_DIR,
    FIGURE_DIR,
    OUTPUT_ROOT,
    PRODUCTS,
    REFERENCE_DISTANCES_KM,
    ensure_output_dirs,
    years_for,
    write_json,
)


UIBR_DAILY_ROOT = OUTPUT_ROOT / "uibr_daily"
UIBR_TRANSECT_ROOT = OUTPUT_ROOT / "uibr_transect"
UISST_ROOT = OUTPUT_ROOT / "uisst_baseline"
UIW_ANNUAL = OUTPUT_ROOT / "uiw_lamont" / "per_cell_per_year.parquet"

OUT_ROOT = OUTPUT_ROOT / "comparison"
REGRESSION_OUT = OUT_ROOT / "uisst_vs_uibr_regressions.parquet"
REGRESSION_CSV = OUT_ROOT / "uisst_vs_uibr_regressions.csv"
XB_CDF_OUT = OUT_ROOT / "x_b_cdf_percentiles.parquet"
XB_CDF_CSV = OUT_ROOT / "x_b_cdf_percentiles.csv"
UIW_CORR_OUT = OUT_ROOT / "annual_uibr_vs_uiw_correlations.parquet"
UIW_CORR_CSV = OUT_ROOT / "annual_uibr_vs_uiw_correlations.csv"
SPREAD_OUT = OUT_ROOT / "cross_product_uibr_spread.parquet"
SPREAD_CSV = OUT_ROOT / "cross_product_uibr_spread.csv"
DAILY_UISST_OUT = OUT_ROOT / "uisst_daily_cell_medians.parquet"
META_JSON = OUT_ROOT / "stage5_meta.json"

CELL_ORDER = ["Cape Frio", "Walvis Bay", "Luderitz", "Cape Columbine", "Cape Peninsula"]
PRODUCT_ORDER = ["OSTIA", "OSPO", "AVHRR", "MW_OI"]
MIN_ANNUAL_DAYS_FOR_CORRELATION = 330


def safe_linregress(x: pd.Series, y: pd.Series) -> dict[str, float]:
    valid = pd.concat([x, y], axis=1).dropna()
    valid = valid[np.isfinite(valid.iloc[:, 0]) & np.isfinite(valid.iloc[:, 1])]
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return {
            "n": int(len(valid)),
            "slope": math.nan,
            "intercept": math.nan,
            "r": math.nan,
            "r2": math.nan,
            "p_value": math.nan,
            "stderr": math.nan,
        }
    res = stats.linregress(valid.iloc[:, 0], valid.iloc[:, 1])
    return {
        "n": int(len(valid)),
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r": float(res.rvalue),
        "r2": float(res.rvalue**2),
        "p_value": float(res.pvalue),
        "stderr": float(res.stderr),
    }


def safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    valid = df[[x, y]].dropna()
    if len(valid) < 3 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return math.nan
    return float(valid[x].corr(valid[y], method=method))


def load_uibr_daily() -> pd.DataFrame:
    parts = []
    for product in PRODUCTS:
        for path in sorted((UIBR_DAILY_ROOT / product).glob("*.parquet")):
            parts.append(pd.read_parquet(path))
    if not parts:
        raise FileNotFoundError(f"No Stage 4b parquets under {UIBR_DAILY_ROOT}")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["lamont_year"] = np.where(df["date"].dt.month >= 7, df["date"].dt.year, df["date"].dt.year - 1)
    return df


def build_uisst_daily() -> pd.DataFrame:
    if DAILY_UISST_OUT.exists():
        return pd.read_parquet(DAILY_UISST_OUT)
    parts = []
    for product in PRODUCTS:
        for year in years_for(product):
            path = UISST_ROOT / product / f"{product}_{year}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            grouped = df.groupby(["product", "date", "season", "cell", "d_0_km"], dropna=False)
            summary = grouped.agg(
                UI_SST_daily_K=("UI_SST_K", "median"),
                UI_SST_cool_daily_K=("UI_SST_cool_K", "median"),
                n_transects_uisst=("col_id", "nunique"),
            ).reset_index()
            parts.append(summary)
    if not parts:
        raise FileNotFoundError(f"No Stage 2 parquets under {UISST_ROOT}")
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(DAILY_UISST_OUT, index=False)
    return out


def compute_uisst_regressions(uibr_daily: pd.DataFrame, uisst_daily: pd.DataFrame) -> pd.DataFrame:
    merged = uisst_daily.merge(
        uibr_daily[
            [
                "product",
                "date",
                "season",
                "cell",
                "A_in_daily_K",
                "x_b_daily_km",
                "beta_inshore_daily",
                "A_in_daily_valid",
            ]
        ],
        on=["product", "date", "season", "cell"],
        how="inner",
    )
    rows = []
    for (cell, product, season, d0), group in merged.groupby(["cell", "product", "season", "d_0_km"], dropna=False):
        stats_dict = safe_linregress(group["A_in_daily_K"], group["UI_SST_cool_daily_K"])
        stats_dict.update(
            {
                "cell": cell,
                "product": product,
                "season": season,
                "d_0_km": float(d0),
                "median_x_b_km": float(group["x_b_daily_km"].median(skipna=True)),
                "distance_from_median_x_b_km": float(abs(d0 - group["x_b_daily_km"].median(skipna=True))),
                "A_in_valid_day_fraction": float(group["A_in_daily_valid"].mean(skipna=True)),
                "UI_SST_sign": "UI_SST_cool_daily_K = T_offshore - T_coast",
            }
        )
        rows.append(stats_dict)
    out = pd.DataFrame(rows)
    return out.sort_values(["cell", "product", "season", "d_0_km"]).reset_index(drop=True)


def compute_xb_cdf() -> pd.DataFrame:
    rows = []
    for product in PRODUCTS:
        for year in years_for(product):
            path = UIBR_TRANSECT_ROOT / product / f"{product}_{year}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["cell", "season", "x_b_km", "bp_accepted"])
            df = df[(df["bp_accepted"] == 1) & df["x_b_km"].notna()]
            for (cell, season), group in df.groupby(["cell", "season"], dropna=False):
                x = group["x_b_km"].to_numpy(dtype=float)
                if len(x) == 0:
                    continue
                for d0 in REFERENCE_DISTANCES_KM:
                    rows.append(
                        {
                            "cell": cell,
                            "product": product,
                            "season": season,
                            "d_0_km": float(d0),
                            "n": int(len(x)),
                            "x_b_median_km": float(np.nanmedian(x)),
                            "x_b_P5_km": float(np.nanquantile(x, 0.05)),
                            "x_b_P95_km": float(np.nanquantile(x, 0.95)),
                            "percentile_x_b_at_d0": float(np.mean(x <= d0)),
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = (
        out.groupby(["cell", "product", "season", "d_0_km"], as_index=False)
        .agg(
            n=("n", "sum"),
            x_b_median_km=("x_b_median_km", "median"),
            x_b_P5_km=("x_b_P5_km", "median"),
            x_b_P95_km=("x_b_P95_km", "median"),
            percentile_x_b_at_d0=("percentile_x_b_at_d0", "mean"),
        )
        .sort_values(["cell", "product", "season", "d_0_km"])
    )
    return out


def compute_annual_uibr_vs_uiw(uibr_daily: pd.DataFrame) -> pd.DataFrame:
    annual = (
        uibr_daily.groupby(["cell", "product", "lamont_year"], dropna=False)
        .agg(
            n_days=("date", "nunique"),
            A_in_annual_mean_K=("A_in_daily_K", "mean"),
            beta_inshore_annual_mean=("beta_inshore_daily", "mean"),
            x_b_annual_mean_km=("x_b_daily_km", "mean"),
            A_in_valid_day_fraction=("A_in_daily_valid", "mean"),
        )
        .reset_index()
        .rename(columns={"lamont_year": "year"})
    )
    annual["complete_lamont_year"] = annual["n_days"] >= 365
    annual["meets_min_annual_days"] = annual["n_days"] >= MIN_ANNUAL_DAYS_FOR_CORRELATION
    wind = pd.read_parquet(UIW_ANNUAL).rename(columns={"TCU_m2_s_days": "TCU_w_m2_s_days"})
    merged = annual.merge(
        wind[["cell", "year", "TCU_w_m2_s_days", "NUD", "NUE"]],
        on=["cell", "year"],
        how="inner",
    )
    merged = merged[merged["meets_min_annual_days"]].copy()
    rows = []
    for (cell, product), group in merged.groupby(["cell", "product"], dropna=False):
        for component in ["A_in_annual_mean_K", "beta_inshore_annual_mean", "x_b_annual_mean_km"]:
            rows.append(
                {
                    "cell": cell,
                    "product": product,
                    "component": component,
                    "target": "TCU_w_m2_s_days",
                    "n_years": int(group[[component, "TCU_w_m2_s_days"]].dropna().shape[0]),
                    "min_annual_days": MIN_ANNUAL_DAYS_FOR_CORRELATION,
                    "median_n_days": float(group["n_days"].median(skipna=True)),
                    "pearson_r": safe_corr(group, component, "TCU_w_m2_s_days", "pearson"),
                    "spearman_r": safe_corr(group, component, "TCU_w_m2_s_days", "spearman"),
                    "median_A_in_valid_day_fraction": float(group["A_in_valid_day_fraction"].median(skipna=True)),
                }
            )
    return pd.DataFrame(rows).sort_values(["cell", "product", "component"]).reset_index(drop=True)


def compute_cross_product_spread() -> pd.DataFrame:
    df = pd.read_parquet(OUTPUT_ROOT / "per_cell_summary.parquet")
    components = {
        "beta_inshore": "beta_inshore_median",
        "x_b": "x_b_median",
        "A_in": "A_in_median",
    }
    rows = []
    for component, col in components.items():
        group_cols = ["cell", "season", "year"]
        for keys, group in df[group_cols + ["product", col]].dropna(subset=[col]).groupby(group_cols, dropna=False):
            values = group[col].to_numpy(dtype=float)
            rows.append(
                {
                    "cell": keys[0],
                    "season": keys[1],
                    "year": int(keys[2]),
                    "component": component,
                    "n_products": int(group["product"].nunique()),
                    "value_min": float(np.nanmin(values)),
                    "value_max": float(np.nanmax(values)),
                    "value_range": float(np.nanmax(values) - np.nanmin(values)),
                    "value_std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else math.nan,
                    "value_median": float(np.nanmedian(values)),
                }
            )
    return pd.DataFrame(rows).sort_values(["cell", "season", "year", "component"]).reset_index(drop=True)


def plot_regression_summary(reg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes = axes.ravel()
    for ax, product in zip(axes, PRODUCT_ORDER):
        sub = reg[reg["product"] == product]
        summary = sub.groupby("d_0_km")["r2"].median().reindex(REFERENCE_DISTANCES_KM)
        ax.plot(summary.index, summary.values, marker="o", lw=2)
        ax.set_title(product)
        ax.set_xlabel("fixed offshore distance d0 (km)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("median R2 across cell-seasons")
    axes[2].set_ylabel("median R2 across cell-seasons")
    fig.suptitle("Stage 5: UI_SST_cool vs A_in daily regression strength")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "stage5_uisst_uibr_r2_by_d0.png", dpi=220)
    fig.savefig(FIGURE_DIR / "stage5_uisst_uibr_r2_by_d0.pdf")
    plt.close(fig)


def plot_xb_cdf(cdf: pd.DataFrame) -> None:
    pivot = (
        cdf.groupby(["product", "d_0_km"])["percentile_x_b_at_d0"]
        .median()
        .unstack("d_0_km")
        .reindex(PRODUCT_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), [int(c) for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("d0 (km)")
    ax.set_title("Median empirical percentile P(x_b <= d0)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="percentile")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "stage5_xb_cdf_percentiles.png", dpi=220)
    fig.savefig(FIGURE_DIR / "stage5_xb_cdf_percentiles.pdf")
    plt.close(fig)


def plot_spread(spread: pd.DataFrame) -> None:
    sub = spread[spread["n_products"] >= 3].copy()
    summary = (
        sub.groupby(["cell", "component"])["value_range"]
        .median()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, component in zip(axes, ["beta_inshore", "x_b", "A_in"]):
        s = summary[summary["component"] == component].set_index("cell").reindex(CELL_ORDER)
        ax.bar(np.arange(len(CELL_ORDER)), s["value_range"])
        ax.set_title(component)
        ax.set_xticks(np.arange(len(CELL_ORDER)), CELL_ORDER, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("median cross-product range")
    fig.suptitle("Stage 5: cross-product spread in UI_BR components")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "stage5_cross_product_uibr_spread.png", dpi=220)
    fig.savefig(FIGURE_DIR / "stage5_cross_product_uibr_spread.pdf")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    uibr_daily = load_uibr_daily()
    uisst_daily = build_uisst_daily()

    regressions = compute_uisst_regressions(uibr_daily, uisst_daily)
    regressions.to_parquet(REGRESSION_OUT, index=False)
    regressions.to_csv(REGRESSION_CSV, index=False)

    xb_cdf = compute_xb_cdf()
    xb_cdf.to_parquet(XB_CDF_OUT, index=False)
    xb_cdf.to_csv(XB_CDF_CSV, index=False)

    annual_corr = compute_annual_uibr_vs_uiw(uibr_daily)
    annual_corr.to_parquet(UIW_CORR_OUT, index=False)
    annual_corr.to_csv(UIW_CORR_CSV, index=False)

    spread = compute_cross_product_spread()
    spread.to_parquet(SPREAD_OUT, index=False)
    spread.to_csv(SPREAD_CSV, index=False)

    plot_regression_summary(regressions)
    plot_xb_cdf(xb_cdf)
    plot_spread(spread)

    write_json(
        META_JSON,
        {
            "stage": 5,
            "script": "pipeline/05_compare_indices.py",
            "daily_regression_predictor": "A_in_daily_K",
            "daily_regression_response": "UI_SST_cool_daily_K",
            "outputs": {
                "regressions": str(REGRESSION_OUT),
                "x_b_cdf_percentiles": str(XB_CDF_OUT),
                "annual_uibr_vs_uiw_correlations": str(UIW_CORR_OUT),
                "cross_product_uibr_spread": str(SPREAD_OUT),
            },
        },
    )
    print(f"Wrote {len(regressions):,} regressions -> {REGRESSION_CSV}")
    print(f"Wrote {len(xb_cdf):,} x_b percentile rows -> {XB_CDF_CSV}")
    print(f"Wrote {len(annual_corr):,} annual wind-correlation rows -> {UIW_CORR_CSV}")
    print(f"Wrote {len(spread):,} cross-product spread rows -> {SPREAD_CSV}")


if __name__ == "__main__":
    main()
