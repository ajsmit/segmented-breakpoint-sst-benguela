# Reproducing the study

Run all commands from this repository's root. The release contains source
code and selected derived data sufficient for the verification example,
Table 1 range, headline daily correlations and representative profile plots.
A full 2015–2025 rerun requires the original daily NetCDF products and the
intermediate feature stack. Those large inputs are obtained separately.

## External inputs

Place daily GHRSST L4 files under `data/raw/OSTIA`, `data/raw/OSPO`,
`data/raw/AVHRR` and `data/raw/MW_OI`. The product variable is `analysed_sst`;
OSTIA and AVHRR filenames end in `.nc4`, OSPO and MW OI in `.nc`.
The filename timestamp parser expects the GHRSST YYYYMMDDhhmmss convention.
Obtain the products from NASA Earthdata/PO.DAAC, using the product identifiers
and versions cited in the article. Obtain ERA5 10 m winds from the Copernicus
Climate Data Store, following the variables and dimensions read by
`pipeline/03_construct_uiw_lamont.py`, and place them under `data/raw/era5`.
Provider authentication is handled outside these scripts.

The exact study mesh is supplied under
`data/derived/analysis_runs/fmm_uc2d_v2_outer500km_v2026_05_21_r2/mesh_curvilinear`.
It has 742 cross-shore transects. Paper-specific membership of 189 transects is
in `results/audit/paperA_cell_transects.csv`; the earlier pipeline window
membership is retained separately in `results/data_access`. Use the former
for the accepted paper's AR(1) analysis. They must not be interchanged.

## Scientific workflow

1. Run `upstream/build_feature_stack.py --product OSTIA --date 2020-07-15`
   as a single-day input check, then use `--year` and `--season` for the
   intended product/year blocks. Its output is
   `data/outputs/uc2d_v2/features_bp/PRODUCT/PRODUCT_YEAR_SEASON.parquet`.
   The retained mesh, smoothing, native-grain constants and fit criteria
   define the supplied implementation.
2. The numbered `pipeline/` scripts construct baseline SST contrasts,
   ERA5 transport and daily/annual summaries. Start with the supplied window
   definitions; inspect stage 00 before replacing them. Stage 08 produces
   supporting baseline plots and tables.
3. Run the paper-specific programs in this dependency order:
   `paperA_model_evidence.py`, `paperA_pixel_index.py`,
   `paperA_across_transect_thin.py`, `paperA_aggregate_thinned.py`,
   `paperA_sensitivity_grid.py`, `paperA_residuals_ar1.py`,
   `paperA_neff_acceptance.py`, and `paperA_ar1_headline_results.py`.
   Prefix each filename with `python analysis/`. The last program creates
   the accepted AR(1)-adjusted profile flags, fixed-distance summaries,
   Table 1 and paired SST/wind classifications.
4. Bootstrap and sensitivity programs operate on these intermediate files;
   consult their input/output docstrings before running. In particular,
   `paperA_xb_bootstrap.py` reports the 5th–95th percentile per-profile span,
   which is a central **90% interval**. Separate block-bootstrap intervals
   use 95% coverage and retain that label.
5. The `paperA_fig*` scripts draw the paper-specific figures. They use the
   derived tables/profiles and may need additional intermediate files from
   the full workflow. Generated figures go under `results/figures/`.

## Supplied derived data

`paperA_ar1/fixed_distance_profiles.parquet` retains the accepted profiles
used in the fixed-distance comparisons. `paperA_ar1/per_cell_product_day.parquet`
contains daily sector/product summaries. `diagnostics/paperA_fig_s5_profiles.parquet`
and its manifest contain the selected illustrative profiles.
`uibr_phenology/per_cell_product_year.parquet` and `uiw_lamont/per_cell_per_year.parquet`
are the baseline annual series displayed in supplementary Figure S1.
`results/tables/*.csv` contains the saved manuscript and diagnostic summaries;
filenames ending in `_ar1` identify the corrected acceptance analysis.
Units are explicit in column suffixes (`_km`, `_K`, `_C`) or the generating
program. A kelvin temperature difference equals a degree Celsius difference.
NaN marks an unavailable quantity and must not be interpreted as zero.

`paperA_v1_table1_reproduction.py` is retained because its numerical helpers
are imported by downstream scripts. Its standalone comparison routine
requires a baseline archive outside the bundled example and is not the
accepted Table 1 generator. `paperA_ar1_headline_results.py` supplies that table.

## Release checks and limits

The release changes file locations and machine-specific paths while preserving
numerical fitting functions and thresholds. Source and delivered hashes are
listed in `provenance/source_manifest.json`. Verification results are in
`provenance/verification.json`. The complete external-data pipeline has not
been rerun during packaging; the executable checks exercise the bundled
scientific examples and internal dependencies. The generated plots expose the
stored scientific values; the separately delivered journal production artwork
also contains typography and panel-label corrections.
