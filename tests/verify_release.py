"""Fast scientific and archive checks using only supplied data."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import paperA_model_evidence as evidence
spec = importlib.util.spec_from_file_location("features", ROOT / "upstream/build_feature_stack.py")
features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)


def brute_force(x, y):
    """Independent direct least squares at the same candidate positions."""
    solutions = []
    for xb in x[2:-1]:
        design = np.column_stack((np.ones(len(x)), np.minimum(x, xb), np.maximum(0, x-xb)))
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        rss = np.sum((y-design @ coef)**2)
        solutions.append((rss, xb, coef))
    return min(solutions, key=lambda row: row[0])


def main():
    checked = 0
    for item in json.loads((ROOT / "provenance/source_manifest.json").read_text()):
        actual = hashlib.sha256((ROOT / item["release"]).read_bytes()).hexdigest()
        assert actual == item["release_sha256"], item["release"]
        checked += 1
    x = np.arange(0., 501., 5.)
    rng = np.random.default_rng(20261117)
    y = 16 + .025*np.minimum(x, 180) + .002*np.maximum(x-180, 0) + rng.normal(0, .025, len(x))
    fit = features._grid_search_breakpoint(x, y)
    ref = brute_force(x, y)
    assert fit[0] == ref[1] == 180
    np.testing.assert_allclose(fit[1:4], ref[2], atol=1e-9)
    np.testing.assert_allclose(fit[4], ref[0], atol=1e-7)
    try:
        features._grid_search_breakpoint(x[:3], y[:3])
    except ValueError:
        pass
    else:
        raise AssertionError("Three-point fit must be rejected")
    assert np.isnan(evidence.aicc(np.array([1.]), np.array([5.]), 4)[0])
    profiles = pd.read_parquet(ROOT / "data/outputs/uidx/diagnostics/paperA_fig_s5_profiles.parquet")
    real = next(iter(profiles.groupby(["cell", "date", "col_id"])))[1].sort_values("distance_km")
    rx, ry = features._sample_at_native_pixels(real.distance_km.to_numpy(), real.sst_k.to_numpy(), 5.56)
    real_fit = features._grid_search_breakpoint(rx, ry)
    real_ref = brute_force(rx, ry)
    assert real_fit[0] == real_ref[1]
    np.testing.assert_allclose(real_fit[1:4], real_ref[2], atol=1e-7)
    np.testing.assert_allclose(real_fit[4], real_ref[0], atol=1e-5)
    boot = pd.read_parquet(ROOT / "results/diagnostics/xb_bootstrap.parquet")
    assert len(boot) == 4000
    np.testing.assert_allclose(boot.ci_width_km, boot.xb_boot_p95-boot.xb_boot_p05, atol=1e-10)
    membership = pd.read_csv(ROOT / "results/audit/paperA_cell_transects.csv")
    assert membership.col_id.nunique() == 189
    result = {"passed": True, "checksummed_source_files": checked, "real_profile_points":len(rx),
              "real_profile_breakpoint_km":real_fit[0], "bootstrap_profiles":len(boot),
              "per_profile_central_coverage_percent":90, "sector_transects":189}
    out = ROOT / "verification-output"
    out.mkdir(exist_ok=True)
    (out / "tests.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
