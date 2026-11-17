#!/usr/bin/env python3
"""paperA_residuals_ar1.py -- Stage 3 Task 3.4: AR(1) fit to the OLS
residuals of accepted two-segment fits at the unique-pixel scale.

For each (cell, season) on the smoke subset (all 5 cells x 4 seasons
on OSTIA 2020), we reconstruct the residuals of every accepted
two-segment profile:

  - For each accepted (col_id, date), apply the upstream's
    _sample_at_native_pixels to (distance_km, sst_k) at the
    native_grain_km of the product to get the unique-pixel sample.
  - Read the fit parameters (alpha_inshore, beta_inshore,
    beta_offshore, x_b_km) from the parquet's first row of the
    (col_id, date) group.
  - Compute predictions T_hat(x) and residuals r = sst - T_hat at the
    unique-pixel positions.

The pooled lag-1 autocorrelation phi_hat is computed across all
profiles in the (cell, season), using only within-profile lag-1 pairs
(no artificial boundary autocorrelations). The 95 % CI uses the
standard SE(phi) approx sqrt((1 - phi^2) / N) with N the pooled
sample size.

Outputs:
  results/diagnostics/ar1_residuals.csv
    columns: cell, season, product, phi, phi_ci_lo, phi_ci_hi,
             n_profiles, n_residual_pairs

  results/figures/fig_s_ar1_lag.pdf
  results/figures/fig_s_ar1_lag.png
    one panel per illustrative profile (one per cell), showing the
    residual sequence and the lag-1 scatter.

Flags phi > 0.8 in the diagnostic CSV.

Run from the release repository root:
    python3 analysis/paperA_residuals_ar1.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paperA_common as P


SMOKE_PRODUCT = "OSTIA"
SMOKE_YEAR = 2020

OUT_CSV = P.PAPERA_ROOT / "diagnostics/ar1_residuals.csv"
OUT_FIG_PDF = P.PAPERA_ROOT / "figures/fig_s_ar1_lag.pdf"
OUT_FIG_PNG = P.PAPERA_ROOT / "figures/fig_s_ar1_lag.png"

# Pull the upstream _sample_at_native_pixels (read-only).
_ufs_spec = importlib.util.spec_from_file_location(
    "ufs", P.ROOT / "upstream/build_feature_stack.py")
_ufs = importlib.util.module_from_spec(_ufs_spec)
_ufs_spec.loader.exec_module(_ufs)
sample_at_native = _ufs._sample_at_native_pixels


def predict(x: np.ndarray, alpha_in: float, beta_in: float,
            beta_off: float, x_b: float) -> np.ndarray:
    """Continuity-constrained two-segment prediction at positions x."""
    return alpha_in + beta_in * np.minimum(x, x_b) + beta_off * np.maximum(0.0, x - x_b)


def residuals_per_profile(g: pd.DataFrame, native_grain_km: float) -> np.ndarray | None:
    first = g.iloc[0]
    if int(first["bp_accepted"]) != 1:
        return None
    alpha_in = first["alpha_inshore"]
    beta_in  = first["beta_inshore"]
    beta_off = first["beta_offshore"]
    x_b      = first["x_b_km"]
    if not np.all(np.isfinite([alpha_in, beta_in, beta_off, x_b])):
        return None
    dist = g["distance_km"].to_numpy(dtype=float)
    sst  = g["sst_k"].to_numpy(dtype=float)
    valid = np.isfinite(sst)
    if valid.sum() < 6:
        return None
    dist_sub, sst_sub = sample_at_native(dist[valid], sst[valid], native_grain_km)
    if len(dist_sub) < 6:
        return None
    yhat = predict(dist_sub, float(alpha_in), float(beta_in),
                   float(beta_off), float(x_b))
    return sst_sub - yhat


def fit_ar1_pooled(profiles: list[np.ndarray]) -> tuple[float, float, float, int, int]:
    """Pooled lag-1 estimator across profiles (within-profile lag-1 pairs only)."""
    num = 0.0
    den = 0.0
    n_pairs = 0
    for r in profiles:
        if len(r) < 2:
            continue
        r = r - r.mean()  # de-mean per-profile (the fit's residuals should sum to 0 already)
        num += float(np.sum(r[1:] * r[:-1]))
        den += float(np.sum(r * r))
        n_pairs += len(r) - 1
    if den <= 0.0 or n_pairs < 2:
        return (np.nan, np.nan, np.nan, len(profiles), n_pairs)
    phi = num / den
    se = float(np.sqrt(max(0.0, 1.0 - phi * phi) / n_pairs))
    return (phi, phi - 1.96 * se, phi + 1.96 * se, len(profiles), n_pairs)


def load_profiles(cell_col_ids: list[int], season: str) -> tuple[pd.DataFrame, float]:
    fp = P.feature_root() / SMOKE_PRODUCT / f"{SMOKE_PRODUCT}_{SMOKE_YEAR}_{season}.parquet"
    meta = json.loads(fp.with_name(fp.stem + "_meta.json").read_text())
    ngk = float(meta["native_grain_km"])
    df = pq.read_table(
        fp,
        columns=["col_id", "date", "distance_km", "sst_k",
                 "bp_accepted", "alpha_inshore", "beta_inshore",
                 "beta_offshore", "x_b_km"],
        filters=[("col_id", "in", cell_col_ids), ("bp_accepted", "=", 1)],
    ).to_pandas()
    return df, ngk


def main() -> None:
    membership = pd.read_csv(P.paperA_cell_transects_path())
    out_rows = []
    illustrative = {}  # cell -> one residual series for plotting

    for cell in P.CELLS:
        cell_col_ids = membership[membership["cell"] == cell]["col_id"].tolist()
        for season in P.SEASONS:
            df, ngk = load_profiles(cell_col_ids, season)
            if df.empty:
                continue
            profiles = []
            for _, g in df.groupby(["col_id", "date"], sort=False):
                r = residuals_per_profile(g, ngk)
                if r is not None and len(r) >= 2:
                    profiles.append(r)
            phi, lo, hi, n_prof, n_pairs = fit_ar1_pooled(profiles)
            out_rows.append({
                "cell": cell, "season": season, "product": SMOKE_PRODUCT,
                "phi": phi, "phi_ci_lo": lo, "phi_ci_hi": hi,
                "n_profiles": n_prof, "n_residual_pairs": n_pairs,
                "flag_phi_gt_0_8": bool(np.isfinite(phi) and phi > 0.8),
            })
            print(f"  {cell:18s} {season}: phi = {phi:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  n_profiles = {n_prof}, "
                  f"n_pairs = {n_pairs:,}",
                  flush=True)
            if season == "DJF" and cell not in illustrative and len(profiles) > 5:
                # Pick a mid-length profile as the illustrative
                lengths = [len(r) for r in profiles]
                med_idx = int(np.argsort(lengths)[len(lengths) // 2])
                illustrative[cell] = profiles[med_idx]

    out = pd.DataFrame(out_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"\n[output csv] {OUT_CSV}")

    # Figure: one panel per illustrative cell
    cells_with_illus = [c for c in P.CELLS if c in illustrative]
    fig, axes = plt.subplots(2, len(cells_with_illus),
                              figsize=(3.0 * len(cells_with_illus), 5.4),
                              constrained_layout=True)
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)
    for j, cell in enumerate(cells_with_illus):
        r = illustrative[cell]
        ax_top = axes[0, j]
        ax_top.plot(r, lw=0.85, color="#2b5d8b")
        ax_top.axhline(0, color="#888", lw=0.5)
        ax_top.set_title(f"{cell} DJF 2020\n(OSTIA, n_eff = {len(r)})", fontsize=9)
        ax_top.set_xlabel("unique-pixel index")
        ax_top.set_ylabel(r"residual $r_t$ (K)")
        ax_bot = axes[1, j]
        ax_bot.scatter(r[:-1], r[1:], s=8, alpha=0.55, color="#9b2c2c", edgecolors="none")
        # Per-profile phi for the illustrative
        phi_i, *_ = fit_ar1_pooled([r])
        ax_bot.set_title(rf"lag-1 scatter ($\hat\phi = {phi_i:+.2f}$)", fontsize=9)
        ax_bot.set_xlabel(r"$r_{t-1}$ (K)")
        ax_bot.set_ylabel(r"$r_t$ (K)")
        ax_bot.axhline(0, color="#888", lw=0.4)
        ax_bot.axvline(0, color="#888", lw=0.4)
        ax_bot.set_aspect("equal")
        lim = max(abs(r).max(), 1e-3) * 1.05
        ax_bot.set_xlim(-lim, lim)
        ax_bot.set_ylim(-lim, lim)
    OUT_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG_PDF, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_FIG_PNG, dpi=200, bbox_inches="tight")
    print(f"[output fig] {OUT_FIG_PDF}")
    print(f"[output fig] {OUT_FIG_PNG}")


if __name__ == "__main__":
    main()
