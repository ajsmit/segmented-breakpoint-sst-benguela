#!/usr/bin/env python3
"""Stage 8: make manuscript figures and tables for uidx."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _uidx_common import OUTPUT_ROOT, ensure_output_dirs, load_coastal_transects_from_features, write_json


FIG_DIR = OUTPUT_ROOT / "manuscript_figures"
TABLE_DIR = OUTPUT_ROOT / "manuscript_tables"
META_JSON = OUTPUT_ROOT / "stage8_meta.json"

CELL_ORDER = ["Cape Frio", "Walvis Bay", "Luderitz", "Cape Columbine", "Cape Peninsula"]
PRODUCT_ORDER = ["OSTIA", "OSPO", "AVHRR", "MW_OI"]
SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]
D0_VALUES = [50.0, 100.0, 200.0, 500.0]
REGIME_ORDER = ["concordant_active", "sst_only", "wind_only", "concordant_quiet"]
REGIME_COLORS = {
    "concordant_active": "#2b8cbe",
    "sst_only": "#f03b20",
    "wind_only": "#feb24c",
    "concordant_quiet": "#bdbdbd",
}
CELL_COLORS = {
    "Cape Frio": "#1b9e77",
    "Walvis Bay": "#d95f02",
    "Luderitz": "#7570b3",
    "Cape Columbine": "#e7298a",
    "Cape Peninsula": "#66a61e",
}
PRODUCT_MARKERS = {"OSTIA": "o", "OSPO": "s", "AVHRR": "^", "MW_OI": "x"}
PRODUCT_LINESTYLES = {"OSTIA": "-", "OSPO": "--", "AVHRR": "-.", "MW_OI": ":"}


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=dpi)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    plt.close(fig)


def write_table(df: pd.DataFrame, stem: str, float_format: str = "%.3f") -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    df.to_latex(TABLE_DIR / f"{stem}.tex", index=False, float_format=float_format, escape=True)


def figure_1_domain_map() -> None:
    coastal = load_coastal_transects_from_features()
    windows = pd.read_csv("results/data_access/canonical_cell_windows.csv")
    transects = pd.read_csv("results/data_access/canonical_cell_transects.csv")
    fig, (ax_map, ax_s) = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax_map.plot(coastal["lon"], coastal["lat"], color="#9aa0a6", lw=0.8, zorder=1)
    for cell in CELL_ORDER:
        sub = transects[transects["cell"] == cell]
        ax_map.scatter(sub["lon"], sub["lat"], s=14, color=CELL_COLORS[cell], label=cell, zorder=3)
    ax_map.set_xlabel("Longitude (deg E)")
    ax_map.set_ylabel("Latitude (deg N)")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, color="#e6e6e6", lw=0.5)

    ax_s.plot(coastal["coast_s_km"], coastal["lat"], color="#9aa0a6", lw=0.8)
    for _, row in windows.iterrows():
        cell = row["cell"]
        ax_s.axvspan(row["coast_s_km_min"], row["coast_s_km_max"], color=CELL_COLORS[cell], alpha=0.22)
        ax_s.scatter(row["coast_s_km_centre"], row["nearest_mesh_lat"], color=CELL_COLORS[cell], s=20)
    ax_s.axvline(2842.5, color="#222222", lw=0.7, ls=":")
    ax_s.set_xlabel("Along-coast distance (km)")
    ax_s.set_ylabel("Latitude (deg N)")
    ax_s.grid(True, color="#e6e6e6", lw=0.5)
    handles = [plt.Line2D([], [], marker="o", color=CELL_COLORS[c], lw=0, ms=4, label=c) for c in CELL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.5)
    fig.suptitle("Fig. 1. Canonical Benguela cells on the inherited mesh", fontsize=10)
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
    save_figure(fig, "fig1_domain_map", dpi=300)


def load_daily_uibr() -> pd.DataFrame:
    return pd.read_parquet(OUTPUT_ROOT / "uibr_phenology" / "daily_classification.parquet")


def figure_2_uibr_vs_uisst() -> None:
    uibr = load_daily_uibr()[["cell", "product", "date", "A_in_daily_K", "A_in_daily_valid"]].copy()
    uibr["date"] = pd.to_datetime(uibr["date"])
    uisst = pd.read_parquet(OUTPUT_ROOT / "comparison" / "uisst_daily_cell_medians.parquet")
    uisst["date"] = pd.to_datetime(uisst["date"])
    df = uisst.merge(uibr, on=["cell", "product", "date"], how="inner")
    df = df[df["A_in_daily_valid"] & df["A_in_daily_K"].notna() & df["UI_SST_cool_daily_K"].notna()].copy()
    df = df[df["product"] == "OSTIA"].copy()
    rng = np.random.default_rng(20260503)
    if len(df) > 45000:
        df = df.iloc[rng.choice(len(df), size=45000, replace=False)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharex=True, sharey=True)
    for ax, d0 in zip(axes.ravel(), D0_VALUES):
        sub = df[df["d_0_km"] == d0]
        for cell in CELL_ORDER:
            cdf = sub[sub["cell"] == cell]
            ax.scatter(cdf["UI_SST_cool_daily_K"], cdf["A_in_daily_K"], s=4, color=CELL_COLORS[cell], alpha=0.12, rasterized=True)
        finite = sub[["UI_SST_cool_daily_K", "A_in_daily_K"]].dropna()
        if len(finite) > 3:
            r = finite["UI_SST_cool_daily_K"].corr(finite["A_in_daily_K"])
            ax.text(0.03, 0.95, f"d0={int(d0)} km\nr={r:.2f}", transform=ax.transAxes, va="top", fontsize=8)
        ax.axline((0, 0), slope=1, color="#333333", lw=0.6, ls=":")
        ax.grid(True, color="#e6e6e6", lw=0.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("UI_SST_cool (K)")
    for ax in axes[:, 0]:
        ax.set_ylabel("A_in daily median (K)")
    handles = [plt.Line2D([], [], marker="o", color=CELL_COLORS[c], lw=0, ms=5, label=c) for c in CELL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.suptitle("Fig. 3. OSTIA daily UI_BR intensity against fixed-distance SST contrasts", fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    save_figure(fig, "fig3_uibr_vs_uisst_scatter", dpi=300)


def figure_3_time_series() -> None:
    daily = load_daily_uibr()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[daily["product"].isin(["OSTIA", "OSPO"])].copy()
    monthly = (
        daily.groupby(["cell", "product", pd.Grouper(key="date", freq="MS")])
        .agg(p_active=("p_active", "mean"), active_fraction=("is_active", "mean"))
        .reset_index()
    )
    pheno = pd.read_parquet(OUTPUT_ROOT / "uibr_phenology" / "per_cell_product_year.parquet")
    wind = pd.read_parquet(OUTPUT_ROOT / "uiw_lamont" / "per_cell_per_year.parquet")[["cell", "year", "NUD"]].rename(columns={"NUD": "NUD_w"})
    pheno = pheno[pheno["product"].isin(["OSTIA", "OSPO"])].merge(wind, on=["cell", "year"], how="left")

    fig, axes = plt.subplots(len(CELL_ORDER), 3, figsize=(8.2, 8.8), sharex="col")
    month_nums = sorted(monthly["date"].unique())
    x_dates = pd.to_datetime(month_nums)
    for i, cell in enumerate(CELL_ORDER):
        c = CELL_COLORS[cell]
        for j, product in enumerate(["OSTIA", "OSPO"]):
            sub = monthly[(monthly["cell"] == cell) & (monthly["product"] == product)].set_index("date").reindex(x_dates)
            axes[i, 0].plot(x_dates, sub["p_active"], color=c, lw=1.0, alpha=0.75 if product == "OSTIA" else 0.45, ls="-" if product == "OSTIA" else "--")
            axes[i, 1].imshow(
                sub["active_fraction"].to_numpy()[None, :],
                aspect="auto",
                cmap="Greys",
                vmin=0,
                vmax=1,
                extent=[mdates.date2num(x_dates.min()), mdates.date2num(x_dates.max()), j, j + 0.8],
            )
        annual = pheno[pheno["cell"] == cell]
        for product in ["OSTIA", "OSPO"]:
            sub = annual[annual["product"] == product]
            axes[i, 2].plot(sub["year"], sub["NUD_BR"], marker=PRODUCT_MARKERS[product], color=c, lw=1.0, ls="-" if product == "OSTIA" else "--", label=product)
        w = wind[wind["cell"] == cell]
        axes[i, 2].plot(w["year"], w["NUD_w"], color="#222222", lw=1.0, ls=":", label="ERA5")
        axes[i, 0].set_ylim(0, 1.05)
        axes[i, 1].set_yticks([0.4, 1.4])
        axes[i, 1].set_yticklabels(["OSTIA", "OSPO"], fontsize=6)
        axes[i, 0].set_ylabel(cell, color=c, fontsize=8)
        for ax in axes[i, :]:
            ax.grid(True, color="#eeeeee", lw=0.4)
    axes[0, 0].set_title("Monthly accepted-profile fraction")
    axes[0, 1].set_title("Fraction of days meeting the SST criterion")
    axes[0, 2].set_title("Annual accepted-day count")
    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1, 1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1, 2].set_xlabel("July-to-June year")
    fig.suptitle("Daily accepted-profile frequency and annual accepted-day counts", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "fig5_daily_detection_time_series", dpi=300)


def figure_4_xb_cdf() -> None:
    daily = load_daily_uibr()
    daily = daily[daily["x_b_daily_km"].notna()].copy()
    fig, axes = plt.subplots(1, len(CELL_ORDER), figsize=(9.2, 2.8), sharey=True)
    absent = []
    for ax, cell in zip(axes, CELL_ORDER):
        for product in PRODUCT_ORDER:
            vals = daily[(daily["cell"] == cell) & (daily["product"] == product)]["x_b_daily_km"].dropna().sort_values().to_numpy()
            if vals.size == 0:
                absent.append((cell, product))
                continue
            y = np.arange(1, vals.size + 1) / vals.size
            ax.plot(vals, y, color=CELL_COLORS[cell], ls=PRODUCT_LINESTYLES[product], lw=1.2, label=product)
        absent_here = [product for absent_cell, product in absent if absent_cell == cell]
        if absent_here:
            ax.text(
                0.97,
                0.08,
                "absent: " + ", ".join(absent_here),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.8,
                color="#555555",
            )
        for d0 in D0_VALUES:
            ax.axvline(d0, color="#222222", lw=0.6, ls=":")
            if cell == CELL_ORDER[0]:
                ax.text(d0, 0.03, str(int(d0)), rotation=90, ha="right", va="bottom", fontsize=6)
        ax.set_title(cell, color=CELL_COLORS[cell], fontsize=8)
        ax.set_xlim(0, 520)
        ax.grid(True, color="#e6e6e6", lw=0.5)
    axes[0].set_ylabel("Empirical CDF")
    for ax in axes:
        ax.set_xlabel("x_b (km)")
    handles = [
        plt.Line2D([], [], color="#333333", ls=PRODUCT_LINESTYLES[product], lw=1.2, label=product)
        for product in PRODUCT_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7)
    fig.suptitle("Fig. 4. Breakpoint-distance distributions relative to literature d0 values", fontsize=10)
    fig.tight_layout(rect=[0, 0.11, 1, 0.92])
    save_figure(fig, "fig4_xb_cdf_literature_d0", dpi=300)


def figure_5_four_regime() -> None:
    reg = pd.read_parquet(OUTPUT_ROOT / "uibr_wind_cross_validation" / "four_regime_summary.parquet")
    kappa = pd.read_parquet(OUTPUT_ROOT / "uibr_wind_cross_validation" / "kappa_by_cell_product.parquet")
    reg = reg.merge(kappa[["cell", "product", "coverage_artefact"]], on=["cell", "product"], how="left")
    fig, axes = plt.subplots(len(SEASON_ORDER), 1, figsize=(8.8, 8.0), sharex=True)
    x_labels = [f"{c}\n{p}" for c in CELL_ORDER for p in ["OSTIA", "OSPO", "AVHRR", "MW_OI"]]
    x = np.arange(len(x_labels))
    for ax, season in zip(axes, SEASON_ORDER):
        bottoms = np.zeros(len(x_labels))
        for regime in REGIME_ORDER:
            vals = []
            hatches = []
            for cell in CELL_ORDER:
                for product in PRODUCT_ORDER:
                    row = reg[(reg["cell"] == cell) & (reg["product"] == product) & (reg["season"] == season) & (reg["regime"] == regime)]
                    vals.append(float(row["fraction"].iloc[0]) if len(row) else 0.0)
                    artefact = bool(row["coverage_artefact"].iloc[0]) if len(row) else False
                    hatches.append("///" if artefact else "")
            bars = ax.bar(x, vals, bottom=bottoms, color=REGIME_COLORS[regime], width=0.86, label=regime.replace("_", " "))
            for bar, hatch in zip(bars, hatches):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_edgecolor("#111111")
                    bar.set_linewidth(0.4)
            bottoms += np.asarray(vals)
        ax.set_ylim(0, 1)
        ax.set_ylabel(season)
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.5)
    axes[0].legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=90, fontsize=5.5)
    fig.text(0.01, 0.5, "Daily-classification fraction", va="center", rotation=90)
    fig.suptitle("Fig. 6. SST-wind four-regime daily classification", fontsize=10)
    fig.tight_layout(rect=[0.03, 0, 1, 0.96])
    save_figure(fig, "fig6_four_regime_stacked_bar", dpi=300)


def figure_6_tcu_scatter() -> None:
    br = pd.read_parquet(OUTPUT_ROOT / "uibr_phenology" / "per_cell_product_year.parquet")
    wind = pd.read_parquet(OUTPUT_ROOT / "uiw_lamont" / "per_cell_per_year.parquet")[["cell", "year", "TCU_m2_s_days"]]
    df = br.merge(wind, on=["cell", "year"], how="left")
    df = df[df["tcu_br_applicable"] & df["TCU_BR"].notna() & df["TCU_m2_s_days"].notna()].copy()
    fig, axes = plt.subplots(1, len(CELL_ORDER), figsize=(9.2, 2.9), sharex=False, sharey=False)
    for ax, cell in zip(axes, CELL_ORDER):
        sub = df[df["cell"] == cell]
        for product in [p for p in PRODUCT_ORDER if p != "MW_OI"]:
            sdf = sub[sub["product"] == product]
            ax.scatter(sdf["TCU_m2_s_days"], sdf["TCU_BR"], marker=PRODUCT_MARKERS[product], s=18, color=CELL_COLORS[cell], alpha=0.75, label=product)
        finite = sub[["TCU_m2_s_days", "TCU_BR"]].dropna()
        if len(finite) >= 4:
            r = finite["TCU_m2_s_days"].corr(finite["TCU_BR"])
            rho = stats.spearmanr(finite["TCU_m2_s_days"], finite["TCU_BR"]).correlation
            ax.text(0.04, 0.95, f"r={r:.2f}\nrho={rho:.2f}", transform=ax.transAxes, va="top", fontsize=7)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        lim0 = min(x0, y0)
        lim1 = max(x1, y1)
        ax.plot([lim0, lim1], [lim0, lim1], color="#333333", lw=0.6, ls=":")
        ax.set_title(cell, color=CELL_COLORS[cell], fontsize=8)
        ax.grid(True, color="#e6e6e6", lw=0.5)
    axes[0].set_ylabel("TCU_BR (K days)")
    for ax in axes:
        ax.set_xlabel("TCU_w (m2 s-1 days)")
    handles = [plt.Line2D([], [], marker=PRODUCT_MARKERS[p], color="#444444", lw=0, ms=5, label=p) for p in ["OSTIA", "OSPO", "AVHRR"]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle("Fig. 7. SST-based TCU_BR against wind-based TCU_w", fontsize=10)
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    save_figure(fig, "fig7_tcu_br_vs_tcu_w_scatter", dpi=300)


def table_1_summary() -> pd.DataFrame:
    df = pd.read_parquet(OUTPUT_ROOT / "per_cell_summary.parquet")
    table = (
        df.groupby(["cell", "product", "season"], dropna=False)
        .agg(
            beta_inshore_median=("beta_inshore_median", "median"),
            x_b_median_km=("x_b_median", "median"),
            A_in_median_K=("A_in_median", "median"),
            A_in_valid_fraction=("A_in_valid_fraction", "median"),
            UI_SST_cool_d200_K=("UI_SST_cool_median_d200", "median"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )
    table["cell"] = pd.Categorical(table["cell"], CELL_ORDER, ordered=True)
    table["product"] = pd.Categorical(table["product"], PRODUCT_ORDER, ordered=True)
    table["season"] = pd.Categorical(table["season"], SEASON_ORDER, ordered=True)
    table = table.sort_values(["cell", "product", "season"]).reset_index(drop=True)
    write_table(table, "table1_per_cell_summary")
    return table


def table_2_correlation_matrix() -> pd.DataFrame:
    rows = []
    uibr = load_daily_uibr()[
        ["cell", "product", "date", "season", "A_in_daily_K", "beta_inshore_daily", "x_b_daily_km"]
    ].copy()
    uibr["date"] = pd.to_datetime(uibr["date"])
    uisst = pd.read_parquet(OUTPUT_ROOT / "comparison" / "uisst_daily_cell_medians.parquet")
    uisst["date"] = pd.to_datetime(uisst["date"])
    daily = uisst.merge(uibr, on=["cell", "product", "date", "season"], how="inner")
    daily_components = ["A_in_daily_K", "beta_inshore_daily", "x_b_daily_km"]
    for (cell, product, season, d0), group in daily.groupby(["cell", "product", "season", "d_0_km"], dropna=False):
        for comp in daily_components:
            g = group[[comp, "UI_SST_cool_daily_K"]].dropna()
            if len(g) < 4 or g[comp].nunique() < 2 or g["UI_SST_cool_daily_K"].nunique() < 2:
                pr = sr = np.nan
            else:
                pr = g[comp].corr(g["UI_SST_cool_daily_K"])
                sr = stats.spearmanr(g[comp], g["UI_SST_cool_daily_K"]).correlation
            rows.append(
                {
                    "cell": cell,
                    "product": product,
                    "season": season,
                    "component": comp,
                    "target": f"UI_SST_cool_d{int(d0):03d}",
                    "n": len(g),
                    "pearson_r": pr,
                    "spearman_r": sr,
                    "level": "daily",
                }
            )
    annual = pd.read_parquet(OUTPUT_ROOT / "trends" / "annual_index_series.parquet")
    targets = ["TCU_w_m2_s_days", "NUD_w", "NUE_w"]
    comps = ["A_in_annual_mean_K", "beta_inshore_annual_mean", "x_b_annual_mean_km"]
    for (cell, product), group in annual[annual["meets_min_annual_days"]].groupby(["cell", "product"]):
        for comp in comps:
            for target in targets:
                g = group[[comp, target]].dropna()
                if len(g) < 4 or g[comp].nunique() < 2 or g[target].nunique() < 2:
                    pr = sr = np.nan
                else:
                    pr = g[comp].corr(g[target])
                    sr = stats.spearmanr(g[comp], g[target]).correlation
                rows.append({"cell": cell, "product": product, "season": "annual", "component": comp, "target": target, "n": len(g), "pearson_r": pr, "spearman_r": sr, "level": "annual"})
    table = pd.DataFrame(rows)
    table["cell"] = pd.Categorical(table["cell"], CELL_ORDER, ordered=True)
    table["product"] = pd.Categorical(table["product"], PRODUCT_ORDER, ordered=True)
    table = table.sort_values(["level", "cell", "product", "season", "component", "target"]).reset_index(drop=True)
    write_table(table, "table2_cross_correlation_matrix")
    return table


def table_3_kappa_marginals() -> pd.DataFrame:
    kappa = pd.read_parquet(OUTPUT_ROOT / "uibr_wind_cross_validation" / "kappa_by_cell_product.parquet")
    reg = pd.read_parquet(OUTPUT_ROOT / "uibr_wind_cross_validation" / "four_regime_summary.parquet")
    marg = (
        reg[reg["regime"].isin(["sst_only", "wind_only"])]
        .pivot_table(index=["cell", "product", "season"], columns="regime", values="fraction", aggfunc="first")
        .reset_index()
    )
    for col in ["sst_only", "wind_only"]:
        if col not in marg.columns:
            marg[col] = 0.0
        marg[col] = marg[col].fillna(0.0)
    table = marg.merge(kappa, on=["cell", "product"], how="left")
    table = table[
        [
            "cell",
            "product",
            "season",
            "cohen_kappa",
            "agreement_fraction",
            "sst_active_fraction",
            "wind_active_fraction",
            "sst_only",
            "wind_only",
            "coverage_artefact",
        ]
    ]
    table["cell"] = pd.Categorical(table["cell"], CELL_ORDER, ordered=True)
    table["product"] = pd.Categorical(table["product"], PRODUCT_ORDER, ordered=True)
    table["season"] = pd.Categorical(table["season"], SEASON_ORDER, ordered=True)
    table = table.sort_values(["cell", "product", "season"]).reset_index(drop=True)
    write_table(table, "table3_kappa_and_marginals")
    return table


def main() -> None:
    ensure_output_dirs()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_domain_map()
    figure_2_uibr_vs_uisst()
    figure_3_time_series()
    figure_4_xb_cdf()
    figure_5_four_regime()
    figure_6_tcu_scatter()
    t1 = table_1_summary()
    t2 = table_2_correlation_matrix()
    t3 = table_3_kappa_marginals()
    write_json(
        META_JSON,
        {
            "stage": "8",
            "script": "pipeline/08_make_figures.py",
            "figure_dir": str(FIG_DIR),
            "table_dir": str(TABLE_DIR),
            "figure_formats": ["pdf", "png"],
            "png_dpi": 300,
            "cell_color_map": CELL_COLORS,
            "tables": {
                "table1_rows": int(len(t1)),
                "table2_rows": int(len(t2)),
                "table3_rows": int(len(t3)),
            },
            "coverage_artefact_handling": "Figure 5 hatches bars where kappa_by_cell_product.coverage_artefact is true.",
        },
    )
    print(f"Wrote figures under {FIG_DIR}")
    print(f"Wrote tables under {TABLE_DIR}")


if __name__ == "__main__":
    main()
