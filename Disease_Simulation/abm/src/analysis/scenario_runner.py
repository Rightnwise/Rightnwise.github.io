"""
여러 시나리오 자동 실행 + 비교 그래프.
  python src/analysis/scenario_runner.py
결과: result/<시나리오>/ 각각 + result/comparison/ (비교 6종 + all_summary.csv)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src.config.scenarios import SCENARIOS, get_scenario_config
from src.models.nyc_metapop_sird import run_simulation
from src.analysis.metrics import save_run_outputs
from src.visualization import plots
from src.utils.paths import RESULT_DIR, ensure_dir
from src.utils.geo import load_prepared_gdf


def run_all(names=None):
    names = names or list(SCENARIOS)
    gdf = load_prepared_gdf(verbose=True)     # 캐시 워밍(1회 로드)
    results = {}
    for name in names:
        cfg = get_scenario_config(name)
        res = run_simulation(cfg, gdf=gdf)
        save_run_outputs(res, os.path.join(RESULT_DIR, name))
        results[name] = res
        s = res.summary
        print(f"[{name:24s}] 감염 {s['total_infected']:>6,} · 사망 {s['total_deaths']:>4,} · "
              f"정점 {s['peak_infected']:>5,}({s['peak_day']:>3}일) · "
              f"발병률 {s['final_attack_rate']*100:4.0f}%")

    # 비교 그래프 + 통합 요약
    cmpdir = ensure_dir(os.path.join(RESULT_DIR, "comparison"))
    plots.plot_scenario_comparison(results, cmpdir)
    pd.DataFrame([r.summary for r in results.values()]).to_csv(
        os.path.join(cmpdir, "all_summary.csv"), index=False)
    print(f"\n비교 그래프 저장: {cmpdir}/ (active_infected, cumulative_deaths, "
          f"new_infections, peak/total_deaths/attack_rate bar)")
    return results


if __name__ == "__main__":
    run_all()
