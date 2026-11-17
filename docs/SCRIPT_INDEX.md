# Script index

Run scripts from the repository root. Paths and outputs are described in each module docstring.

## analysis

- `paperA_across_transect_thin.py`: paperA_across_transect_thin.py -- Stage 2 Task 2.2: first-occurrence-wins pixel-stride deduplication across transects within each UC window.
- `paperA_aggregate_thinned.py`: paperA_aggregate_thinned.py -- Stage 2 Task 2.4: re-aggregate p_active on the across-transect-deduplicated set, compare against the v1 formula.
- `paperA_aic_confusion.py`: paperA_aic_confusion.py -- Stage 3 Task 3.3: pairwise 2x2 confusion matrices between AIC, AICc, and BIC at Delta >= 4 on the smoke subset.
- `paperA_ar1_headline_results.py`: paperA_ar1_headline_results.py -- AR(1)-corrected headline results.
- `paperA_block_bootstrap.py`: paperA_block_bootstrap.py -- Stage 6 Task 6.1: alongshore-block bootstrap 95 % intervals for the three headline statistics.
- `paperA_common.py`: paperA_common.py -- shared paths, constants, and helpers for paperA scripts.
- `paperA_din_thin_shift.py`: paperA_din_thin_shift.py -- Stage 2 Task 2.5 helper: climatological-median D̄_in shift on retained transects relative to all-accepted transects.
- `paperA_export_fig_s5_profiles.py`: Select and extract OSTIA profiles for Supplementary Figure S5.
- `paperA_fig1_study_region.py`: paperA_fig1_study_region.py -- Stage 8 Task 8.1: rebuild Figure 1 (study region with enlarged inset, CVD audit).
- `paperA_fig1_study_region_v3.py`: Add illustrative Benguela regions to Figure 1, preserving the v2 mesh.
- `paperA_fig2_method_walkthrough.py`: Create the illustrative five-panel breakpoint-method walk-through.
- `paperA_fig3_d0_scatter_v2.py`: Revised inshore-drawdown versus fixed-distance scatter figure.
- `paperA_fig3_xb_cdf.py`: paperA_fig3_xb_cdf.py -- per-cell ECDFs of x_b on the AR(1)-corrected OSTIA and OSPO accepted-profile set, with d_0 reference distances marked.
- `paperA_fig4_four_regime.py`: paperA_fig4_four_regime.py -- rebuild Figure 4 from the AR(1)-corrected four-regime fractions.
- `paperA_fig_s3_kappa_wind_marginal_v2.py`: Create the revised Figure S3 with an axis matched to observed kappa.
- `paperA_fig_s5_readable_panels.py`: Draw readable per-study-sector panels for Supplementary Figure S5.
- `paperA_model_evidence.py`: paperA_model_evidence.py -- Stage 3 Task 3.1: compute AICc and BIC alongside AIC for the one- and two-segment fits.
- `paperA_neff_acceptance.py`: paperA_neff_acceptance.py -- Stage 3 Task 3.5: convert AR(1) phi to n_eff* and recompute the per-profile AIC. Tabulate the change in acceptance rate per (cell, product, season) against the unique-pixel baseline.
- `paperA_paired_classifier.py`: paperA_paired_classifier.py -- Stage 5 Tasks 5.1 + 5.2 + 5.3: paired-classifier diagnostic statistics for the SST-structure detector vs the Bakun-Lamont wind detector.
- `paperA_phenology_filter.py`: paperA_phenology_filter.py -- Stage 7 Task 7.2: partial July-to-June-year exclusion helper for paperA phenology aggregations.
- `paperA_pixel_index.py`: paperA_pixel_index.py -- Stage 2 Task 2.1: map every accepted sigma-node observation in the recovered features_bp tree to its parent L4 pixel index.
- `paperA_residuals_ar1.py`: paperA_residuals_ar1.py -- Stage 3 Task 3.4: AR(1) fit to the OLS residuals of accepted two-segment fits at the unique-pixel scale.
- `paperA_retention_table.py`: paperA_retention_table.py -- Stage 2 Task 2.3: retention rate per (cell, product) on the deduplicated set.
- `paperA_sensitivity_figures.py`: paperA_sensitivity_figures.py -- Stage 4 Task 4.2: produce per-axis small-multiples figures and tables.
- `paperA_sensitivity_grid.py`: paperA_sensitivity_grid.py -- Stage 4 Task 4.1: six-dimensional threshold sensitivity grid.
- `paperA_sensitivity_survival.py`: paperA_sensitivity_survival.py -- Stage 4 Task 4.3: headline-finding survival summary across the six sensitivity axes.
- `paperA_smoke_aic_aicc_bic.py`: paperA_smoke_aic_aicc_bic.py -- Stage 3 Task 3.2: tabulate per-profile AIC, AICc, BIC on the smoke subset (Cape Columbine, all four products, DJF 2020). Verifies that the AICc correction scales correctly with n.
- `paperA_v1_table1_reproduction.py`: paperA_v1_table1_reproduction.py -- operational per-(cell, product) climatological-median table for paperA Table 1.
- `paperA_xb_bootstrap.py`: paperA_xb_bootstrap.py -- per-profile bootstrap of x_b on a stratified random sample of accepted profiles, used as a population- level identifiability diagnostic.

## pipeline

- `00_define_windows_verify_inputs.py`: Stage 0: define canonical uidx cell windows and verify inherited inputs.
- `01_construct_uibr.py`: Stage 1: construct UI_BR per canonical-cell transect-day.
- `02_construct_uisst_baseline.py`: Stage 2: construct fixed-distance UI_SST baselines.
- `03_construct_uiw_lamont.py`: Stage 3: construct Bakun-Lamont wind-index phenology from ERA5 winds.
- `04_aggregate_per_cell.py`: Stage 4: aggregate UI_BR and fixed-distance UI_SST to cell-season-year.
- `04b_aggregate_uibr_daily.py`: Stage 4b: aggregate transect-day UI_BR outputs to cell-day series.
- `04c_detect_uibr_events.py`: Stage 4c: detect SST breakpoint active days and annual UI_BR phenology.
- `04d_cross_validate_wind_sst.py`: Stage 4d: compare SST breakpoint daily detection against ERA5 wind detection.
- `05_compare_indices.py`: Stage 5: cross-index comparison and reconciliation.
- `08_make_figures.py`: Stage 8: make manuscript figures and tables for uidx.
- `_uidx_common.py`: Shared paths and helpers for the uidx active scripts.

## upstream

- `build_feature_stack.py`: 07_build_feature_stack.py — Stage 4: SST feature stack on the FMM mesh.
