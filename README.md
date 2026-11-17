# Segmented-breakpoint SST structure in the Benguela

Code and derived data for **A segmented-breakpoint measure of cross-shore sea-surface-temperature structure in the Benguela system**, by Albertus J. Smit and Neville A. Sweijd, accepted by *Ocean Science*.

The analysis estimates a daily thermal breakpoint, an inshore SST contrast,
and the frequency with which profiles satisfy the specified SST-structure
criteria in five Benguela study sectors. The period is 2015–2025, with AVHRR
available from 2016. The four SST products are OSTIA, OSPO, AVHRR and MW OI.
Acceptance describes cross-shore SST structure; physical upwelling requires
independent oceanographic evidence.

## Quick verification

Use Python 3.12, from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python tests/verify_release.py
python examples/reproduce_summary.py
```

The first command checks the regression against an independent least-squares
calculation, refits a supplied real profile, checks interval coverage and
verifies file checksums. The example regenerates the reported breakpoint
range and daily correlations from the supplied derived data, and writes a
profile figure under `verification-output/`.

## Contents

| Directory | Contents |
|---|---|
| `analysis/` | Paper-specific model evidence, native-pixel checks, AR(1) acceptance, bootstrap, sensitivity and figure scripts |
| `upstream/` | Feature-stack construction and continuous segmented regression |
| `pipeline/` | SST contrasts, wind transport, daily aggregation and annual phenology |
| `results/` | Sector definitions, diagnostic outputs and manuscript table data |
| `data/derived/` | Study mesh and coastline |
| `data/outputs/uidx/` | Derived profile, daily and annual data used by the reproducible examples |
| `docs/` | Workflow, file meanings and external input requirements |
| `provenance/` | Source-to-release file mapping and checksums |

[Reproducibility instructions](docs/REPRODUCIBILITY.md) distinguish the bundled
examples from a complete rerun, which requires the external SST and wind
inputs. [Script index](docs/SCRIPT_INDEX.md) describes each retained program.
Historical identifiers such as `uidx`, `paperA`, and `A_in` remain in filenames
and stored column names to preserve provenance. The manuscript quantity
`D_in` corresponds to the stored `A_in` family where documented by the shared
reader in `analysis/paperA_common.py`.

## Citation and licensing

Version 1.0.0 is archived in Zenodo at
[doi:10.5281/zenodo.24306305](https://doi.org/10.5281/zenodo.24306305).
Citation metadata are in `CITATION.cff`. The record was published through
this repository's GitHub release integration. Code uses the MIT
licence. Derived tables, mesh and documentation use CC BY 4.0; see
[DATA_LICENSE.md](DATA_LICENSE.md).
