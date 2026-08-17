"""
결과 지표 계산 + 저장.
  · compute_summary(timeseries, name) -> dict
  · save_run_outputs(result, outdir): timeseries.csv / summary.csv / png 3종 저장
"""
import os
import pandas as pd

from src.utils.paths import ensure_dir


def compute_summary(ts: pd.DataFrame, scenario_name: str) -> dict:
    """일별 timeseries DataFrame 에서 요약 지표를 계산."""
    total_agents = int(ts["alive_population"].iloc[0])          # day0 = 전체(사망 0)
    # 면역 소실(SIRS)이 있으면 재감염이 생기므로 '고유 감염자'와 '감염 이벤트'를 구분
    unique_col = "cumulative_unique_infected" if "cumulative_unique_infected" in ts \
        else "cumulative_infections"
    total_infected = int(ts[unique_col].iloc[-1])               # 한 번이라도 감염된 사람 수
    total_infection_events = int(ts["cumulative_infections"].iloc[-1])  # 재감염 포함 이벤트
    total_deaths = int(ts["cumulative_deaths"].iloc[-1])
    peak_idx = int(ts["active_infected"].idxmax())
    peak_infected = int(ts["active_infected"].iloc[peak_idx])
    peak_day = int(ts["day"].iloc[peak_idx])

    return {
        "scenario_name": scenario_name,
        "total_infected": total_infected,
        "total_infection_events": total_infection_events,
        "total_deaths": total_deaths,
        "peak_infected": peak_infected,
        "peak_day": peak_day,
        "final_attack_rate": round(total_infected / total_agents, 4) if total_agents else 0.0,
        "final_fatality_rate": round(total_deaths / total_infection_events, 4) if total_infection_events else 0.0,
        "epidemic_end_day": int(ts["day"].iloc[-1]),
        "max_daily_new_infections": int(ts["new_infections"].max()),
    }


def save_run_outputs(result, outdir, make_plots=True, make_gif=False):
    """한 실행 결과를 outdir 에 저장(csv 2종 + png 3종, 선택적 GIF)."""
    ensure_dir(outdir)
    result.timeseries.to_csv(os.path.join(outdir, "timeseries.csv"), index=False)
    pd.DataFrame([result.summary]).to_csv(os.path.join(outdir, "summary.csv"), index=False)

    if make_plots or make_gif:
        # 지연 임포트(그래프 백엔드 로딩 분리)
        from src.visualization import plots
        if make_plots:
            plots.plot_epidemic_curve(result, os.path.join(outdir, "epidemic_curve.png"))
            plots.plot_cumulative_deaths(result, os.path.join(outdir, "cumulative_deaths.png"))
            plots.plot_final_map(result, os.path.join(outdir, "final_map.png"))
        if make_gif and result.spatial_frames:
            plots.build_animation(result, os.path.join(outdir, "animation.gif"))
    return outdir
