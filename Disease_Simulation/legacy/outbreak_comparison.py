"""
outbreak_comparison.py
================================================================
초기 발생 방식 두 가지를 끝까지 시뮬레이션하고 비교한다.

  (1) 중심 발생      : 격자 중앙 한 곳에서 시작하는 단일 감염파
  (2) 다중 발생      : 서로 떨어진 네 곳에서 시작하는 여러 감염파

관찰 목표:
  · 감염파가 공간적으로 어떻게 퍼지는가 (I(x,y,t) 애니메이션)
  · 다중 발생에서 '파동이 만나면' 어떻게 되는가.
    → 반응-확산 감염파는 지나온 자리의 S(감염가능자)를 태워 없앤다.
      두 파동이 만나는 경계에는 이미 감염을 겪어 S 가 고갈된 사람들뿐이라
      더 감염시킬 대상이 없어 파동이 '서로를 소멸'시키며 성장을 멈춘다.
      (Fisher-KPP 파동의 상호 소멸 / front annihilation)

설계: initial_conditions 패키지로 초기 상태를 만들고, SpatialSIR 로 굴린다.
      시각화 코드만 여기서 간단히 자체 구현(전역 결합 최소화).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from initial_conditions import create_initial_conditions, summarize, print_summary
from sir_reaction_diffusion import (
    SpatialSIR, RESULT_DIR, DX, BETA, GAMMA, D_S, D_I, D_R,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 공통 설정 ----------------
NX = NY = 120          # 격자 (120 x 120 km)
DAYS = 250             # 파동이 만나 겹치는 걸 볼 만큼 충분히 길게
DT = 0.1               # 시간 간격 [일]
FRAME_EVERY = 25       # 2.5일마다 한 프레임 → 약 100프레임
BASE_DENSITY = 500.0   # 균일 인구밀도 [명/km²]


# ==================================================================
# 시뮬레이션 실행 (모델은 그대로 재사용)
# ==================================================================
def run_case(name, config):
    """config 로 초기조건 생성 → 모델 주입 → 끝까지 전진하며 기록."""
    S0, I0, R0, N = create_initial_conditions(config)
    print_summary(summarize(S0, I0, R0, N), title=f"{name} 초기 상태")

    model = SpatialSIR(NX, NY, DX, BETA, GAMMA, D_S, D_I, D_R, N)
    model.set_initial_state(S0, I0, R0, N)

    steps = int(DAYS / DT)
    times = np.zeros(steps + 1)
    tot_S = np.zeros(steps + 1)
    tot_I = np.zeros(steps + 1)
    tot_R = np.zeros(steps + 1)
    tot_S[0], tot_I[0], tot_R[0] = model.totals()

    frames, frame_times = [model.I.copy()], [0.0]
    for k in range(steps):
        model.step(DT)
        t = (k + 1) * DT
        times[k + 1] = t
        tot_S[k + 1], tot_I[k + 1], tot_R[k + 1] = model.totals()
        if (k + 1) % FRAME_EVERY == 0:
            frames.append(model.I.copy())
            frame_times.append(t)

    return {
        "name": name, "times": times,
        "S": tot_S, "I": tot_I, "R": tot_R,
        "frames": frames, "frame_times": frame_times,
    }


# ==================================================================
# 시각화
# ==================================================================
def animate_case(case, vmax, filename):
    """I(x,y,t) 감염파 확산 애니메이션 → GIF."""
    frames, ftimes = case["frames"], case["frame_times"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(frames[0], cmap="inferno", origin="lower",
                   vmin=0, vmax=vmax, extent=[0, NX * DX, 0, NY * DX])
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Infected density [people/km²]")
    title = ax.set_title("")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"{case['name']}  I(x, y, t)   t = {ftimes[i]:.0f} days")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = os.path.join(RESULT_DIR, filename)
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"저장: {out}")


def plot_totals(case, filename):
    """한 사례의 전체 S(t), I(t), R(t)."""
    t = case["times"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, case["S"], color="#2f6fd0", lw=2.2, label="S Susceptible")
    ax.plot(t, case["I"], color="#d1352c", lw=2.2, label="I Infected")
    ax.plot(t, case["R"], color="#2e8b57", lw=2.2, label="R Recovered")
    ax.fill_between(t, 0, case["I"], color="#d1352c", alpha=0.12)

    peak = int(np.argmax(case["I"]))
    ax.plot(t[peak], case["I"][peak], "o", color="#d1352c", ms=7, zorder=5)
    ax.annotate(f"Peak day {t[peak]:.0f}\n{case['I'][peak]:,.0f} people",
                xy=(t[peak], case["I"][peak]),
                xytext=(t[peak] + 10, case["I"][peak]),
                fontsize=10, color="#d1352c",
                arrowprops=dict(arrowstyle="->", color="#d1352c"))

    ax.set_title(f"{case['name']} — Total S/I/R Trajectory", fontsize=13)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total population")
    ax.set_xlim(0, DAYS)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, filename)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"저장: {out}")


def plot_comparison(case_c, case_m, filename):
    """두 사례를 겹쳐서 감염파 상호작용 효과를 한눈에 비교."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # (좌) 감염중 I(t) 비교
    ax1.plot(case_c["times"], case_c["I"], color="#2f6fd0", lw=2.2,
             label="Central Outbreak (1 site)")
    ax1.plot(case_m["times"], case_m["I"], color="#d1352c", lw=2.2,
             label="Multiple Outbreaks (4 sites)")
    ax1.set_title("Concurrent Infections I(t) Comparison", fontsize=12)
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Total infected")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.25)

    # (우) 감염가능 S(t) 비교 — 고갈 속도가 곧 파동의 '연료'
    ax2.plot(case_c["times"], case_c["S"], color="#2f6fd0", lw=2.2,
             label="Central Outbreak (1 site)")
    ax2.plot(case_m["times"], case_m["S"], color="#d1352c", lw=2.2,
             label="Multiple Outbreaks (4 sites)")
    ax2.set_title("Susceptible S(t) Comparison (Fuel of the Wave)", fontsize=12)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Total susceptible")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Initial Outbreak Pattern Comparison: multiple outbreaks spread faster, "
                 "but when waves meet they stop together as fuel (S) runs out", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULT_DIR, filename)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"저장: {out}")


# ==================================================================
# 메인
# ==================================================================
if __name__ == "__main__":
    # (1) 중심 발생 — 중앙 한 곳
    config_center = {
        "grid_size": (NY, NX),
        "population_type": "uniform",
        "base_density": BASE_DENSITY,
        "initial_condition_type": "center_outbreak",
        "initial_infected_fraction": 0.05,
        "outbreak_radius": 3,
    }

    # (2) 다중 발생 — 서로 떨어진 네 곳(좌표·반경·감염량 모두 설정 가능)
    config_multi = {
        "grid_size": (NY, NX),
        "population_type": "uniform",
        "base_density": BASE_DENSITY,
        "initial_condition_type": "multiple_outbreaks",
        "outbreak_seeds": [
            {"x": 35, "y": 35, "radius": 3, "fraction": 0.05},
            {"x": 85, "y": 35, "radius": 3, "fraction": 0.05},
            {"x": 35, "y": 85, "radius": 3, "fraction": 0.05},
            {"x": 85, "y": 85, "radius": 3, "fraction": 0.05},
        ],
    }

    print("=" * 60)
    case_c = run_case("Central Outbreak", config_center)
    print("\n" + "=" * 60)
    case_m = run_case("Multiple Outbreaks", config_multi)

    # 두 애니메이션의 색 스케일을 통일(공정 비교)
    vmax = max(max(f.max() for f in case_c["frames"]),
               max(f.max() for f in case_m["frames"]))

    print("\n애니메이션/그래프 생성 중...")
    animate_case(case_c, vmax, "outbreak_center.gif")
    animate_case(case_m, vmax, "outbreak_multiple.gif")
    plot_totals(case_c, "outbreak_center_totals.png")
    plot_totals(case_m, "outbreak_multiple_totals.png")
    plot_comparison(case_c, case_m, "outbreak_comparison.png")

    # 파동 상호작용 요약
    pc = int(np.argmax(case_c["I"]))
    pm = int(np.argmax(case_m["I"]))
    print("\n── 파동 상호작용 관찰 " + "─" * 30)
    print(f"  중심 발생: 정점 {case_c['times'][pc]:.0f}일, "
          f"최대 동시감염 {case_c['I'][pc]:,.0f}명")
    print(f"  다중 발생: 정점 {case_m['times'][pm]:.0f}일, "
          f"최대 동시감염 {case_m['I'][pm]:,.0f}명")
    print(f"  최종 미감염 S: 중심 {case_c['S'][-1]:,.0f}명  vs  "
          f"다중 {case_m['S'][-1]:,.0f}명")
    print("완료.")
