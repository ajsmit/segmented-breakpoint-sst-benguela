#!/usr/bin/env python3
"""Stage 4d: compare SST breakpoint daily detection against ERA5 wind detection."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from _uidx_common import DIAG_DIR, OUTPUT_ROOT, ensure_output_dirs, write_json


SST_DAILY = OUTPUT_ROOT / "uibr_phenology" / "daily_classification.parquet"
SST_PHENO = OUTPUT_ROOT / "uibr_phenology" / "per_cell_product_year.parquet"
WIND_DAILY = OUTPUT_ROOT / "uiw_lamont" / "daily_cell_wind.parquet"
WIND_PHENO = OUTPUT_ROOT / "uiw_lamont" / "per_cell_per_year.parquet"
OUT_ROOT = OUTPUT_ROOT / "uibr_wind_cross_validation"
REGIME_OUT = OUT_ROOT / "daily_four_regimes.parquet"
REGIME_SUMMARY_OUT = OUT_ROOT / "four_regime_summary.parquet"
KAPPA_OUT = OUT_ROOT / "kappa_by_cell_product.parquet"
CORRELATION_OUT = OUT_ROOT / "annual_phenology_correlations.parquet"
REGIME_CSV = DIAG_DIR / "stage4d_four_regime_summary.csv"
KAPPA_CSV = DIAG_DIR / "stage4d_kappa_by_cell_product.csv"
CORRELATION_CSV = DIAG_DIR / "stage4d_annual_phenology_correlations.csv"
META_JSON = OUT_ROOT / "stage4d_meta.json"


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    tab = pd.crosstab(a.astype(bool), b.astype(bool)).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    n = tab.to_numpy().sum()
    if n == 0:
        return float("nan")
    po = (tab.loc[False, False] + tab.loc[True, True]) / n
    row = tab.sum(axis=1) / n
    col = tab.sum(axis=0) / n
    pe = row[False] * col[False] + row[True] * col[True]
    if math.isclose(1.0 - pe, 0.0):
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def corr_pair(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    valid = df[[x, y]].dropna()
    if len(valid) < 3:
        return float("nan")
    return float(valid[x].corr(valid[y], method=method))


def main() -> None:
    ensure_output_dirs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sst_daily = pd.read_parquet(SST_DAILY)
    wind_daily = pd.read_parquet(WIND_DAILY)
    sst_daily["date"] = pd.to_datetime(sst_daily["date"])
    wind_daily["date"] = pd.to_datetime(wind_daily["date"])
    wind_daily = wind_daily[["cell", "date", "season", "UI_w_m3_s_per_100m"]].copy()
    wind_daily["is_wind_active"] = wind_daily["UI_w_m3_s_per_100m"] > 0

    daily = sst_daily.merge(wind_daily, on=["cell", "date", "season"], how="inner")
    conditions = [
        daily["is_active"] & daily["is_wind_active"],
        ~daily["is_active"] & ~daily["is_wind_active"],
        daily["is_active"] & ~daily["is_wind_active"],
        ~daily["is_active"] & daily["is_wind_active"],
    ]
    choices = ["concordant_active", "concordant_quiet", "sst_only", "wind_only"]
    daily["regime"] = np.select(conditions, choices, default="unknown")
    daily.to_parquet(REGIME_OUT, index=False)

    regime_summary = (
        daily.groupby(["cell", "product", "season", "regime"], dropna=False)
        .size()
        .rename("n_days")
        .reset_index()
    )
    totals = regime_summary.groupby(["cell", "product", "season"])["n_days"].transform("sum")
    regime_summary["fraction"] = regime_summary["n_days"] / totals
    regime_summary.to_parquet(REGIME_SUMMARY_OUT, index=False)
    regime_summary.to_csv(REGIME_CSV, index=False)

    kappa = (
        daily.groupby(["cell", "product"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_days": len(g),
                    "sst_active_fraction": float(g["is_active"].mean()),
                    "wind_active_fraction": float(g["is_wind_active"].mean()),
                    "agreement_fraction": float((g["is_active"] == g["is_wind_active"]).mean()),
                    "cohen_kappa": cohen_kappa(g["is_active"], g["is_wind_active"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    kappa["coverage_artefact"] = (
        (kappa["sst_active_fraction"] < 0.05)
        & (kappa["wind_active_fraction"] > 0.30)
    )
    daily = daily.merge(
        kappa[["cell", "product", "agreement_fraction", "coverage_artefact"]],
        on=["cell", "product"],
        how="left",
    )
    daily.to_parquet(REGIME_OUT, index=False)
    kappa.to_parquet(KAPPA_OUT, index=False)
    kappa.to_csv(KAPPA_CSV, index=False)

    sst_pheno = pd.read_parquet(SST_PHENO)
    wind_pheno = pd.read_parquet(WIND_PHENO).rename(
        columns={
            "TCU_m3_s_per_100m_days": "TCU_w",
            "NUD": "NUD_w",
            "NUE": "NUE_w",
        }
    )
    merged = sst_pheno.merge(wind_pheno[["cell", "year", "TCU_w", "NUD_w", "NUE_w"]], on=["cell", "year"], how="inner")

    rows = []
    for (cell, product), group in merged.groupby(["cell", "product"], dropna=False):
        for metric_br, metric_w, applicable_filter in [
            ("NUD_BR", "NUD_w", pd.Series(True, index=group.index)),
            ("NUE_BR", "NUE_w", pd.Series(True, index=group.index)),
            ("TCU_BR", "TCU_w", group["tcu_br_applicable"].astype(bool)),
        ]:
            sub = group[applicable_filter]
            rows.append(
                {
                    "cell": cell,
                    "product": product,
                    "metric": metric_br.replace("_BR", ""),
                    "n_years": int(sub[[metric_br, metric_w]].dropna().shape[0]),
                    "pearson_r": corr_pair(sub, metric_br, metric_w, "pearson"),
                    "spearman_r": corr_pair(sub, metric_br, metric_w, "spearman"),
                    "skips_non_applicable_tcu": metric_br == "TCU_BR",
                }
            )
    correlations = pd.DataFrame(rows)
    correlations.to_parquet(CORRELATION_OUT, index=False)
    correlations.to_csv(CORRELATION_CSV, index=False)

    write_json(
        META_JSON,
        {
            "stage": "4d",
            "script": "pipeline/04d_cross_validate_wind_sst.py",
            "daily_regime_output": str(REGIME_OUT),
            "tcu_rule": "TCU_BR correlations filter tcu_br_applicable and drop NaN values; MW_OI remains in daily regime/kappa outputs.",
        },
    )
    print(f"Wrote {len(daily):,} daily regime rows -> {REGIME_OUT}")
    print(f"Wrote kappa diagnostics -> {KAPPA_CSV}")
    print(f"Wrote correlation diagnostics -> {CORRELATION_CSV}")


if __name__ == "__main__":
    main()
