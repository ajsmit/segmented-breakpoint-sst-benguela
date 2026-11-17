"""Recompute headline statistics and draw a supplied representative SST profile."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import paperA_ar1_headline_results as study
import paperA_fig3_d0_scatter_v2 as scatter


def main():
    out = ROOT / "verification-output"
    out.mkdir(exist_ok=True)
    profiles = pd.read_parquet(ROOT / "data/outputs/uidx/paperA_ar1/fixed_distance_profiles.parquet")
    study.OUT_TABLE1 = out / "table1_recomputed.csv"
    table = study.table1_ar1(profiles)
    stored = pd.read_csv(ROOT / "results/tables/table1_per_cell_summary.csv")
    for col in ["D_in_C", "x_b_km", "beta_inshore_C_per_km"]:
        np.testing.assert_allclose(table[col], stored[col], rtol=5e-6, equal_nan=True)
    bounds = [float(table.x_b_km.min()), float(table.x_b_km.max())]
    daily = scatter.build_intermediate()
    correlations = {str(d):float(daily[["D_in_med",f"ui_d{d}"]].corr().iloc[0,1]) for d in [50,100,200,500]}
    assert [round(x) for x in bounds] == [114,270]
    assert round(correlations["50"],2)==.06 and round(correlations["500"],2)==.93
    example = pd.read_parquet(ROOT / "data/outputs/uidx/diagnostics/paperA_fig_s5_profiles.parquet")
    example = example[example.profile_status == "accepted"]
    key, profile = next(iter(example.groupby(["cell","date","col_id"])))
    profile = profile.sort_values("distance_km")
    p = profile.iloc[0]
    x = profile.distance_km.to_numpy()
    fitted = p.alpha_inshore+p.beta_inshore*np.minimum(x,p.x_b_km)+p.beta_offshore*np.maximum(0,x-p.x_b_km)-273.15
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    ax.plot(x,profile.sst_c,label="SST profile",color="#555555")
    ax.plot(x,fitted,label="Continuous segmented fit",color="#0072B2")
    ax.axvline(p.x_b_km,color="#D55E00",ls="--",label="Fitted breakpoint")
    ax.set(xlabel="Distance offshore (km)",ylabel="Sea-surface temperature (°C)",title=f"{key[0]}: {str(key[1])[:10]}")
    ax.legend(frameon=False)
    fig.savefig(out / "representative_profile.png",dpi=180)
    plt.close(fig)
    report={"accepted_profile_rows":len(profiles),"breakpoint_range_km":bounds,"OSTIA_daily_correlations":correlations,"table1_matches_saved_values":True}
    (out / "summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
