"""
파라미터 탐색 도구.

두 가지 방식으로 실행:
  1) CLI (기본, 의존성 없음):
       python parameter_explorer.py --beta 0.12 --lockdown_day 20 --commute_reduction 0.7
     → result/explorer/ 에 결과 저장 + 요약 출력
  2) Streamlit 대시보드(설치돼 있을 때):
       streamlit run parameter_explorer.py

조정 가능 값: beta, latent_days, recovery_days, fatality_rate, initial_infected,
             simulation_days, commute_probability, commute_reduction, lockdown_day,
             vaccination_rate, vaccination_strategy, random_seed
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.nyc_metapop_sird import run_simulation
from src.analysis.metrics import save_run_outputs
from src.utils.paths import RESULT_DIR
from src.utils.geo import load_prepared_gdf


def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# ==================================================================
# CLI
# ==================================================================
def run_cli():
    ap = argparse.ArgumentParser(description="NYC 메타포퓰레이션 SIRD 파라미터 탐색(CLI)")
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--latent_days", type=int, default=4)
    ap.add_argument("--recovery_days", type=int, default=14)
    ap.add_argument("--fatality_rate", type=float, default=0.02)
    ap.add_argument("--initial_infected", type=int, default=20)
    ap.add_argument("--simulation_days", type=int, default=300)
    ap.add_argument("--commute_probability", type=float, default=1.0)
    ap.add_argument("--commute_reduction", type=float, default=0.0)
    ap.add_argument("--lockdown_day", type=int, default=None)
    ap.add_argument("--vaccination_rate", type=float, default=0.0)
    ap.add_argument("--vaccination_strategy", default="random",
                    choices=["random", "high_density_first"])
    ap.add_argument("--isolation_effectiveness", type=float, default=1.0)
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--name", default="explorer")
    ap.add_argument("--animate", action="store_true",
                    help="감염 확산 애니메이션(GIF)도 생성")
    args = ap.parse_args()

    cfg = {k: v for k, v in vars(args).items() if k not in ("name", "animate")}
    cfg["scenario_name"] = args.name

    print("파라미터:", {k: v for k, v in cfg.items() if k != "scenario_name"})
    gdf = load_prepared_gdf(verbose=True)
    result = run_simulation(cfg, gdf=gdf, capture_spatial=args.animate)

    outdir = os.path.join(RESULT_DIR, "explorer")
    save_run_outputs(result, outdir)

    s = result.summary
    print("\n── 요약 지표 ─────────────────────────")
    for k, v in s.items():
        print(f"  {k:24s}: {v}")
    files = "timeseries.csv, summary.csv, epidemic_curve.png, cumulative_deaths.png, final_map.png"
    if args.animate:
        from src.visualization import plots
        gif = os.path.join(outdir, "animation.gif")
        print("애니메이션(GIF) 생성 중...")
        plots.build_animation(result, gif)
        files += ", animation.gif"
    print(f"\n저장: {outdir}/ ({files})")


# ==================================================================
# Streamlit 대시보드 (선택)
# ==================================================================
def run_streamlit():
    import streamlit as st
    import matplotlib.pyplot as plt
    from src.visualization import plots

    st.set_page_config(page_title="SIRD Explorer", layout="wide")
    st.title("NYC Metapopulation SIRD Parameter Explorer")

    with st.sidebar:
        st.header("Parameters")
        cfg = dict(
            beta=st.slider("beta", 0.0, 0.5, 0.10, 0.01),
            latent_days=st.slider("latent_days", 0, 14, 4),
            recovery_days=st.slider("recovery_days", 3, 30, 14),
            fatality_rate=st.slider("fatality_rate", 0.0, 0.2, 0.02, 0.005),
            initial_infected=st.slider("initial_infected", 1, 200, 20),
            simulation_days=st.slider("simulation_days", 30, 400, 300),
            commute_probability=st.slider("commute_probability", 0.0, 1.0, 1.0, 0.05),
            commute_reduction=st.slider("commute_reduction", 0.0, 1.0, 0.0, 0.05),
            lockdown_day=st.number_input("lockdown_day (-1=none)", -1, 400, -1),
            vaccination_rate=st.slider("vaccination_rate", 0.0, 1.0, 0.0, 0.05),
            vaccination_strategy=st.selectbox("vaccination_strategy",
                                              ["random", "high_density_first"]),
            random_seed=st.number_input("random_seed", 0, 9999, 42),
            scenario_name="explorer",
        )
        show_gif = st.checkbox("Generate infection spread animation (GIF) — slow", value=False)
    if cfg["lockdown_day"] < 0:
        cfg["lockdown_day"] = None

    gdf = load_prepared_gdf()
    result = run_simulation(cfg, gdf=gdf, capture_spatial=show_gif)
    ts = result.timeseries

    c1, c2, c3 = st.columns(3)
    c1.metric("Total infected", f"{result.summary['total_infected']:,}")
    c2.metric("Total deaths", f"{result.summary['total_deaths']:,}")
    c3.metric("Final attack rate", f"{result.summary['final_attack_rate']*100:.1f}%")

    st.subheader("S / L / I / R / D time series")
    st.line_chart(ts.set_index("day")[["S", "latent_infected", "I", "R", "D"]])
    cc1, cc2 = st.columns(2)
    cc1.subheader("Active infected")
    cc1.line_chart(ts.set_index("day")[["active_infected"]])
    cc2.subheader("Cumulative deaths")
    cc2.line_chart(ts.set_index("day")[["cumulative_deaths"]])

    st.subheader("Summary metrics")
    st.table({k: [v] for k, v in result.summary.items()})

    st.subheader("Final attack rate map")
    fig, ax = plt.subplots(figsize=(8, 8))
    result.final_node_states.plot(ax=ax, column="attack_rate", cmap="Reds",
                                  vmin=0, vmax=1, edgecolor="0.7", linewidth=0.3,
                                  legend=True)
    ax.set_axis_off(); ax.set_aspect("equal")
    st.pyplot(fig)

    if show_gif:
        st.subheader("Infection spread animation")
        gif_path = os.path.join(RESULT_DIR, "explorer", "animation.gif")
        os.makedirs(os.path.dirname(gif_path), exist_ok=True)
        with st.spinner("Generating GIF... (takes a few seconds)"):
            plots.build_animation(result, gif_path)
        st.image(gif_path)


if __name__ == "__main__":
    if _in_streamlit():
        run_streamlit()
    else:
        run_cli()
