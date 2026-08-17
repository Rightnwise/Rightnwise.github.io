"""
단일 시나리오 실행 진입점.
  python run_scenario.py --scenario baseline
  python run_scenario.py --scenario early_lockdown
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.scenarios import SCENARIOS, get_scenario_config
from src.models.nyc_metapop_sird import run_simulation
from src.analysis.metrics import save_run_outputs
from src.utils.paths import RESULT_DIR
from src.utils.geo import load_prepared_gdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="baseline",
                    help=f"시나리오 이름 {list(SCENARIOS)}")
    ap.add_argument("--gif", action="store_true", help="감염 확산 애니메이션(GIF)도 생성")
    args = ap.parse_args()

    cfg = get_scenario_config(args.scenario)
    print(f"시나리오 실행: {args.scenario}  ({ {k:v for k,v in cfg.items() if k!='scenario_name'} })")
    gdf = load_prepared_gdf(verbose=True)
    result = run_simulation(cfg, gdf=gdf, capture_spatial=args.gif)

    outdir = os.path.join(RESULT_DIR, args.scenario)
    save_run_outputs(result, outdir, make_gif=args.gif)

    s = result.summary
    print(f"  총 감염 {s['total_infected']:,} / 총 사망 {s['total_deaths']:,} / "
          f"정점 {s['peak_infected']:,}({s['peak_day']}일) / "
          f"발병률 {s['final_attack_rate']*100:.1f}% / 종료 {s['epidemic_end_day']}일")
    print(f"저장: {outdir}/")


if __name__ == "__main__":
    main()
