"""
baseline 실행 진입점.
  python run_baseline.py
→ legacy 최종 모델과 동일한 baseline 을 돌리고 result/baseline/ 에 저장.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 프로젝트 루트

from src.models.nyc_metapop_sird import run_simulation
from src.analysis.metrics import save_run_outputs
from src.utils.paths import RESULT_DIR
from src.utils.geo import load_prepared_gdf


def main():
    print("=" * 60)
    print("BASELINE 시뮬레이션")
    gdf = load_prepared_gdf(verbose=True)
    result = run_simulation({"scenario_name": "baseline"}, gdf=gdf)

    outdir = os.path.join(RESULT_DIR, "baseline")
    save_run_outputs(result, outdir)

    s = result.summary
    print(f"  중심지(자동탐색): {result.center_name}")
    print(f"  총 감염 {s['total_infected']:,} / 총 사망 {s['total_deaths']:,}")
    print(f"  정점 {s['peak_infected']:,}명 ({s['peak_day']}일차)")
    print(f"  최종 발병률 {s['final_attack_rate']*100:.1f}% / "
          f"치명률 {s['final_fatality_rate']*100:.2f}% / 종료 {s['epidemic_end_day']}일")
    print(f"저장: {outdir}/  (timeseries.csv, summary.csv, *.png)")
    print("=" * 60)


if __name__ == "__main__":
    main()
