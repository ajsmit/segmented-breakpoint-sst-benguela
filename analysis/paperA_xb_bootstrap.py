#!/usr/bin/env python3
"""paperA_xb_bootstrap.py -- per-profile bootstrap of x_b on a
stratified random sample of accepted profiles, used as a population-
level identifiability diagnostic.

Stratification:
  - Year: 2020 (matches the AR(1) fit year of Methods Section 2.3).
  - Product: OSTIA and OSPO (fine-resolution L4; the AR(1)-corrected
    accepted set on AVHRR / MW_OI is empty, so they are not included).
  - Season: DJF / MAM / JJA / SON.
  - Cell: the five operational Benguela UCs.
  - Per (cell, product, season) stratum: a random sample of
    N_PER_STRATUM accepted profiles, drawn with a fixed seed.
  - Each sampled profile receives 1000 nonparametric bootstrap
    resamples of (distance, SST), refitted with the candidate-grid
    breakpoint search, recording x_b.

Outputs:
  results/diagnostics/xb_bootstrap.parquet
    per-profile bootstrap results.
  results/figures/fig_s4_xb_bootstrap.{pdf,png}
    per-cell distribution of per-profile 90 % bootstrap CI widths,
    with OSTIA and OSPO overlaid.
  results/tables/table_s6_xb_bootstrap_diagnostic.tex
    per-cell summary table (median CI width, percentiles, fractions
    under 50 / 100 km, fraction bimodal, N).

Bimodality flag: a Gaussian KDE of the bootstrap distribution carries
two peaks at prominence >= 0.1 * max separated by more than 50 km.

Run from the release repository root:
    python3 analysis/paperA_xb_bootstrap.py
    python3 analysis/paperA_xb_bootstrap.py --smoke
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P

YEAR = 2020
PRODUCTS = ("OSTIA", "OSPO")
SEASONS = ("DJF", "MAM", "JJA", "SON")
N_BOOTSTRAP = 1000
N_PER_STRATUM = 100
SMOKE_N_PER_STRATUM = 20
BIMODAL_SEP_KM = 50.0
SEED = 20260526

OUT_PARQUET = P.PAPERA_ROOT / "diagnostics/xb_bootstrap.parquet"
OUT_FIG_PDF = P.PAPERA_ROOT / "figures/fig_s4_xb_bootstrap.pdf"
OUT_FIG_PNG = P.PAPERA_ROOT / "figures/fig_s4_xb_bootstrap.png"
OUT_TABLE_TEX = P.PAPERA_ROOT / "tables/table_s6_xb_bootstrap.tex"

_ufs_spec = importlib.util.spec_from_file_location(
    "ufs", P.ROOT / "upstream/build_feature_stack.py")
_ufs = importlib.util.module_from_spec(_ufs_spec)
_ufs_spec.loader.exec_module(_ufs)
sample_at_native = _ufs._sample_at_native_pixels
grid_search_xb = _ufs._grid_search_breakpoint

PRODUCT_STYLE = {"OSTIA": dict(color="#1f3b5b", ls="-",  lw=1.6),
                 "OSPO":  dict(color="#9b2c2c", ls="--", lw=1.4)}


def stratified_sample(product: str, season: str, n_per_stratum: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    fp = P.ROOT / f"data/outputs/uidx/paperA_model_evidence/{product}/{product}_{YEAR}_{season}.parquet"
    df = pd.read_parquet(fp)
    df = df[(df["bp_accepted"] == 1) & np.isfinite(df["delta_aic"])]
    picks = []
    for cell, sub in df.groupby("cell"):
        if cell not in P.CELLS:
            continue
        m = min(len(sub), n_per_stratum)
        idx = rng.choice(len(sub), size=m, replace=False)
        chosen = sub.iloc[idx].copy()
        chosen["product"] = product
        chosen["season"] = season
        picks.append(chosen)
    return pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()


def native_grain(product: str, season: str) -> float:
    fp = P.feature_root() / product / f"{product}_{YEAR}_{season}.parquet"
    meta = json.loads(fp.with_name(fp.stem + "_meta.json").read_text())
    return float(meta["native_grain_km"])


def bootstrap_one(dist: np.ndarray, sst: np.ndarray, native_grain_km: float,
                  rng: np.random.Generator) -> np.ndarray:
    dist_sub, sst_sub = sample_at_native(dist, sst, native_grain_km)
    n = len(dist_sub)
    if n < 6:
        return np.full(N_BOOTSTRAP, np.nan)
    order = np.argsort(dist_sub)
    dist_sorted = dist_sub[order]
    sst_sorted = sst_sub[order]
    out = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        try:
            xb, *_ = grid_search_xb(dist_sorted[idx], sst_sorted[idx])
            out[b] = float(xb)
        except ValueError:
            out[b] = np.nan
    return out


def detect_bimodal(samples: np.ndarray) -> bool:
    valid = samples[np.isfinite(samples)]
    if valid.size < 20 or np.ptp(valid) < 1.0:
        return False
    kde = gaussian_kde(valid, bw_method="silverman")
    grid = np.linspace(valid.min(), valid.max(), 400)
    density = kde(grid)
    peaks, _ = find_peaks(density, prominence=0.1 * density.max())
    if len(peaks) < 2:
        return False
    peak_positions = grid[peaks]
    gaps = np.diff(np.sort(peak_positions))
    return bool(gaps.max() > BIMODAL_SEP_KM)


def run(smoke: bool = False) -> None:
    n_per_stratum = SMOKE_N_PER_STRATUM if smoke else N_PER_STRATUM
    rng = np.random.default_rng(SEED)
    sample_frames = []
    for product in PRODUCTS:
        for season in SEASONS:
            sample_frames.append(stratified_sample(product, season, n_per_stratum, rng))
    picks = pd.concat(sample_frames, ignore_index=True)
    print(f"[sample] {len(picks)} profiles across {len(PRODUCTS)} products x "
          f"{len(SEASONS)} seasons x {len(P.CELLS)} cells "
          f"(target {n_per_stratum} per stratum)", flush=True)

    grain_cache: dict[tuple[str, str], float] = {}
    results = []
    boot_rng = np.random.default_rng(SEED + 1)
    t_start = time.monotonic()
    for i, p in picks.iterrows():
        product = str(p["product"])
        season = str(p["season"])
        if (product, season) not in grain_cache:
            grain_cache[(product, season)] = native_grain(product, season)
        grain_km = grain_cache[(product, season)]
        src_fp = P.feature_root() / product / f"{product}_{YEAR}_{season}.parquet"
        g = pq.read_table(src_fp,
                          columns=["col_id", "date", "distance_km", "sst_k", "x_b_km"],
                          filters=[("col_id", "in", [int(p["col_id"])]),
                                   ("date", "in", [p["date"]])]).to_pandas()
        if g.empty:
            continue
        dist = g["distance_km"].to_numpy(dtype=float)
        sst = g["sst_k"].to_numpy(dtype=float)
        xb_hat = float(g["x_b_km"].iloc[0])
        valid = np.isfinite(sst)
        boots = bootstrap_one(dist[valid], sst[valid], grain_km, boot_rng)
        vb = boots[np.isfinite(boots)]
        if vb.size:
            p05 = float(np.percentile(vb, 5))
            p50 = float(np.percentile(vb, 50))
            p95 = float(np.percentile(vb, 95))
            ci_w = p95 - p05
        else:
            p05 = p50 = p95 = ci_w = np.nan
        results.append({
            "cell": p["cell"], "product": product, "season": season,
            "date": p["date"], "col_id": int(p["col_id"]),
            "n_unique_pixels": int(p["n_eff"]),
            "delta_aic": float(p["delta_aic"]),
            "xb_hat": xb_hat,
            "xb_boot_p05": p05, "xb_boot_p50": p50, "xb_boot_p95": p95,
            "ci_width_km": ci_w,
            "bimodal": detect_bimodal(boots),
        })
        if (i + 1) % 200 == 0 or i + 1 == len(picks):
            elapsed = time.monotonic() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(picks) - i - 1) / rate
            print(f"  [{i+1:>5}/{len(picks)}] elapsed {elapsed:.0f}s, "
                  f"~{remaining:.0f}s remaining", flush=True)

    out_df = pd.DataFrame(results)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[output parquet] {OUT_PARQUET}")
    print(f"[summary] N = {len(out_df)}; "
          f"median CI width = {out_df['ci_width_km'].median():.1f} km; "
          f"fraction < 100 km = {(out_df['ci_width_km'] < 100).mean():.3f}; "
          f"fraction bimodal = {out_df['bimodal'].mean():.3f}")

    summarise_per_cell(out_df)
    plot_per_cell(out_df)
    emit_table(out_df)


def summarise_per_cell(df: pd.DataFrame) -> None:
    print("\n[per-cell summary]")
    rows = []
    for cell in P.CELLS:
        sub = df[df["cell"] == cell].dropna(subset=["ci_width_km"])
        if sub.empty:
            continue
        rows.append({
            "cell": cell, "n": len(sub),
            "median_ci_km": sub["ci_width_km"].median(),
            "p25_ci_km": sub["ci_width_km"].quantile(0.25),
            "p75_ci_km": sub["ci_width_km"].quantile(0.75),
            "frac_lt_50": (sub["ci_width_km"] < 50).mean(),
            "frac_lt_100": (sub["ci_width_km"] < 100).mean(),
            "frac_bimodal": sub["bimodal"].mean(),
        })
    pcs = pd.DataFrame(rows)
    print(pcs.to_string(index=False))


def plot_per_cell(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(P.CELLS), figsize=(3.0 * len(P.CELLS), 3.4),
                              constrained_layout=True, sharey=True)
    for ax, cell in zip(axes, P.CELLS):
        for product, style in PRODUCT_STYLE.items():
            sub = df[(df["cell"] == cell) & (df["product"] == product)].dropna(subset=["ci_width_km"])
            if sub.empty:
                continue
            vals = sub["ci_width_km"].to_numpy()
            xs = np.sort(vals)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, label=product, **style)
        ax.axvline(50, color="#888", lw=0.5)
        ax.axvline(100, color="#888", lw=0.5)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 1.02)
        ax.set_title(cell, fontsize=10)
        ax.set_xlabel(r"per-profile 90 % CI width on $\hat{x}_b$ (km)")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("empirical CDF over profiles")
    axes[-1].legend(fontsize=9, loc="lower right")
    OUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PDF, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG_PNG, dpi=200, bbox_inches="tight")
    print(f"[output fig] {OUT_FIG_PDF}")


def emit_table(df: pd.DataFrame) -> None:
    cells_latex = {"Luderitz": r"L\"uderitz"}
    rows = []
    for cell in P.CELLS:
        for product in PRODUCTS:
            sub = df[(df["cell"] == cell) & (df["product"] == product)].dropna(subset=["ci_width_km"])
            if sub.empty:
                continue
            rows.append({
                "cell": cells_latex.get(cell, cell),
                "product": product, "n": len(sub),
                "median_ci_km": sub["ci_width_km"].median(),
                "p25_ci_km": sub["ci_width_km"].quantile(0.25),
                "p75_ci_km": sub["ci_width_km"].quantile(0.75),
                "frac_lt_50": (sub["ci_width_km"] < 50).mean(),
                "frac_lt_100": (sub["ci_width_km"] < 100).mean(),
                "frac_bimodal": sub["bimodal"].mean(),
            })
    body_lines = []
    for r in rows:
        body_lines.append(
            f"{r['cell']} & {r['product']} & {r['n']} & "
            f"{r['median_ci_km']:.1f} & {r['p25_ci_km']:.1f}--{r['p75_ci_km']:.1f} & "
            f"{r['frac_lt_50']:.2f} & {r['frac_lt_100']:.2f} & {r['frac_bimodal']:.2f} \\\\"
        )
    tex = (
        r"\begin{center}" + "\n"
        r"\setlength{\tabcolsep}{4pt}" + "\n"
        r"\footnotesize" + "\n"
        r"\begin{longtable}{llrrrrrr}" + "\n"
        r"\caption{Per-profile bootstrap diagnostic of $\hat{x}_b$ identifiability, "
        r"computed on a stratified random sample of accepted 2020 profiles "
        rf"({N_PER_STRATUM} profiles per (UC, product, season), pooled across "
        r"the four austral seasons). Per (UC, product): the median width of the "
        r"per-profile 90\,\% bootstrap interval on $\hat{x}_b$, the 25th--75th "
        r"percentile range of that width, the fraction of profiles with width "
        r"below 50 and below 100\,km, and the fraction flagged bimodal "
        r"(Gaussian KDE peaks separated by more than 50\,km). $N$ is the per-"
        r"(UC, product) sample size summed across seasons.}\label{tab:supp_xb_bootstrap}\\" + "\n"
        r"\toprule" + "\n"
        r"UC & Product & $N$ & Median CI (km) & IQR (km) & frac $<50$ & frac $<100$ & frac bimodal \\" + "\n"
        r"\midrule" + "\n"
        r"\endfirsthead" + "\n"
        r"\toprule" + "\n"
        r"UC & Product & $N$ & Median CI (km) & IQR (km) & frac $<50$ & frac $<100$ & frac bimodal \\" + "\n"
        r"\midrule" + "\n"
        r"\endhead" + "\n"
        r"\bottomrule" + "\n"
        r"\endfoot" + "\n"
        + "\n".join(body_lines) + "\n"
        r"\end{longtable}" + "\n"
        r"\end{center}" + "\n"
    )
    OUT_TABLE_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE_TEX.write_text(tex)
    print(f"[output table] {OUT_TABLE_TEX}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"Use {SMOKE_N_PER_STRATUM} per stratum instead of {N_PER_STRATUM}.")
    args = parser.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
