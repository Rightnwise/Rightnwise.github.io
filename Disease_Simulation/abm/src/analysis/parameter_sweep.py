"""
파라미터 민감도 분석(sensitivity sweep).
  python src/analysis/parameter_sweep.py
결과: result/sensitivity/
  · sensitivity_results.csv
  · beta_vs_peak_infected.png / beta_vs_total_deaths.png
  · commute_reduction_vs_total_infected.png
  · heatmap_beta_commute_vs_deaths.png
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src.models.nyc_metapop_sird import run_simulation
from src.visualization import plots
from src.utils.paths import RESULT_DIR, ensure_dir
from src.utils.geo import load_prepared_gdf

BETAS = [0.05, 0.075, 0.10, 0.125, 0.15]
COMMUTE_REDUCTIONS = [0.0, 0.25, 0.5, 0.75]
LATENT_DAYS = [2, 4, 6]


def _row(sweep, value, res):
    return {"sweep": sweep, "value": value, **res.summary}


def sweep():
    outdir = ensure_dir(os.path.join(RESULT_DIR, "sensitivity"))
    gdf = load_prepared_gdf(verbose=True)
    rows = []

    print("· beta 스윕")
    for b in BETAS:
        res = run_simulation({"scenario_name": f"beta_{b}", "beta": b},
                             gdf=gdf)
        rows.append(_row("beta", b, res))

    print("· commute_reduction 스윕 (lockdown_day=0)")
    for cr in COMMUTE_REDUCTIONS:
        res = run_simulation({"scenario_name": f"cr_{cr}",
                              "lockdown_day": 0, "commute_reduction": cr}, gdf=gdf)
        rows.append(_row("commute_reduction", cr, res))

    print("· latent_days 스윕")
    for ld in LATENT_DAYS:
        res = run_simulation({"scenario_name": f"latent_{ld}", "latent_days": ld},
                             gdf=gdf)
        rows.append(_row("latent_days", ld, res))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "sensitivity_results.csv"), index=False)

    # 라인 그래프
    bdf = df[df.sweep == "beta"]
    plots.plot_sweep_line(bdf, "value", "peak_infected",
                          os.path.join(outdir, "beta_vs_peak_infected.png"),
                          "β vs peak infected", "beta", "peak_infected")
    plots.plot_sweep_line(bdf, "value", "total_deaths",
                          os.path.join(outdir, "beta_vs_total_deaths.png"),
                          "β vs total deaths", "beta", "total_deaths")
    crdf = df[df.sweep == "commute_reduction"]
    plots.plot_sweep_line(crdf, "value", "total_infected",
                          os.path.join(outdir, "commute_reduction_vs_total_infected.png"),
                          "Commute reduction vs total infected", "commute_reduction", "total_infected")

    # 히트맵: beta × commute_reduction → total_deaths
    print("· beta × commute_reduction 히트맵")
    hrows = []
    for b in BETAS:
        for cr in COMMUTE_REDUCTIONS:
            res = run_simulation({"scenario_name": f"b{b}_cr{cr}", "beta": b,
                                  "lockdown_day": 0, "commute_reduction": cr}, gdf=gdf)
            hrows.append({"beta": b, "commute_reduction": cr,
                          "total_deaths": res.summary["total_deaths"]})
    hdf = pd.DataFrame(hrows)
    hdf.to_csv(os.path.join(outdir, "heatmap_data.csv"), index=False)
    plots.plot_heatmap(hdf, "beta", "commute_reduction", "total_deaths",
                       os.path.join(outdir, "heatmap_beta_commute_vs_deaths.png"),
                       "β × commute reduction → total deaths")
    print(f"저장: {outdir}/")
    return df


if __name__ == "__main__":
    sweep()
