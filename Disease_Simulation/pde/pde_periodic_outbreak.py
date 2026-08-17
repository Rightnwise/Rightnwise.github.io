"""
pde_periodic_outbreak.py
================================================================
SIRS 반응-확산 모델 + '주기적 외부 outbreak'.

역학 (각 격자칸 = SIRS + 출생/사망 반응, 이웃과 확산으로 결합)
  계절/정현파 강제 없음. S 를 회복시키는 항은 딱 2개: 출생(μN) + 면역소실(ωR).
  ∂S/∂t = μN − βSI/N + ωR − μS + D∇²S   ← 출생 +μN,  면역소실(waning) +ωR
  ∂I/∂t =  βSI/N − γI − μI     + D∇²I
  ∂R/∂t =  γI − ωR − μR        + D∇²R    ← 면역소실(waning) −ωR
  (μ: 출생·사망률 — 새 생명이 S 로 태어나고, 각 구획에서 μ 비율로 사망)
  무유출(Neumann) 경계, 명시적 오일러.

주기적 outbreak
  · enable_periodic_outbreak == True 이면 outbreak_period 마다 외부 감염 유입.
  · 지정/랜덤 위치에서  I[region] += outbreak_strength,  S[region] -= outbreak_strength.
  · S 가 음수가 되지 않도록 clip (실제 감소량 = min(strength, S) 로 인구 보존).
  · outbreak_location: 'center' | 'random' | 'multiple_points'

동심원 파동 + 유행 소멸(핵심)
  · 각 outbreak 는 발생점에서 감염을 주입 → 확산으로 '동심원 파동'이 퍼져 이웃과 충돌.
  · 하지만 SIRS 는 R0>1 이면 감염이 안 죽고 풍토병 평형점에 갇혀, 첫 유행 이후로는
    동심원이 사라지고(배경이 계속 감염 상태) outbreak 가 배경 얼룩만 만든다.
  · 그래서 각 outbreak 후 CLEAR_DELAY 스텝(파동이 도메인을 한 번 쓸고 간 뒤) 잔여 I 를
    소멸시켜 도메인을 감염가능 상태로 리셋한다 → 출생+waning 으로 S 회복 →
    다음 outbreak 가 '새 동심원 파동'을 fresh 하게 퍼뜨리고, 위상궤도에 '반복 loop' 발생.

관찰 목표
  출생(μN)+waning(ωR) 로 감염가능자 S 가 회복되고, 주기적 outbreak 로 동심원 파동이
  반복해 퍼지며, 발생점 S–I 위상궤도에 '반복되는 loop(순환)'가 나타나는지 확인.

시각화 (result/)
  1) pde_periodic_outbreak_phase.png : (좌) 발생점의 S–I 위상궤도  (우) S/I/R 시계열
  2) pde_periodic_outbreak.gif       : I(x,y,t) 히트맵 + 발생점 위상궤도 애니메이션

실행:  python3 pde_periodic_outbreak.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ======================= 설정(옵션) =======================
# ---- 격자 / 시간 ----
NX = NY = 150
DX = 1.0
DT = 0.1
N = 1.0                       # 칸당 총 인구(정규화)
STEPS = 14400

# ---- SIRS 역학 파라미터 ----
BETA = 0.60                   # 감염력
GAMMA = 0.15                  # 회복률  (R0 = β/γ ≈ 4)
OMEGA = 0.010                 # 면역 소실률 R→S (waning)
MU = 0.015                    # 출생·사망률 (새 생명이 S 로 태어남 μN, 각 구획 사망 μ)
D = 0.60                      # 확산계수 (클수록 파동이 동심원으로 잘 퍼짐)

# 유행 소멸(fadeout): 각 outbreak 후 CLEAR_DELAY 스텝이 지나면(파동이 도메인을 한 번
# 쓸고 지나간 뒤) 잔여 I 를 소멸시켜 도메인을 감염가능 상태로 리셋한다. 그래야
#  ① 다음 outbreak 가 '새 동심원 파동'을 fresh 하게 퍼뜨리고(안 그러면 풍토병 배경에 묻힘)
#  ② 출생+waning 으로 S 가 회복되어 위상궤도에 '반복 loop' 가 생긴다.
# (전염병의 유행 간 소멸=stochastic fadeout 을 모사. 0 이면 비활성.)
CLEAR_DELAY = 700

# ---- 주기적 outbreak ----
ENABLE_PERIODIC_OUTBREAK = True
OUTBREAK_PERIOD = 1600        # 몇 스텝마다 outbreak (t = PERIOD*DT 마다)
OUTBREAK_STRENGTH = 0.30      # 유입 감염자 밀도 (I += , S -= )
OUTBREAK_RADIUS = 4           # 유입 지점 반지름(칸) — 작을수록 점源→깨끗한 동심원
OUTBREAK_LOCATION = "multiple_points"  # 'center' | 'random' | 'multiple_points'
RNG = np.random.default_rng(0)

# ---- 시각화 ----
FRAME_STRIDE = 48             # 히트맵 프레임 간격 → 약 300프레임(동심원 확산 포착)
# =========================================================

# 발생점 4곳(네 모서리 부근)
MULTI_POINTS = [(NX // 4, NY // 4), (3 * NX // 4, NY // 4),
                (NX // 4, 3 * NY // 4), (3 * NX // 4, 3 * NY // 4)]

# 위상궤도를 그릴 추적점 = 발생점
if OUTBREAK_LOCATION == "center":
    TRACK_POINTS = [(NX // 2, NY // 2)]
else:
    TRACK_POINTS = MULTI_POINTS
TRACK_COLORS = ["#2f6fd0", "#d1352c", "#2e8b57", "#8e44ad", "#e08a00"]


def laplacian(f):
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f) / (DX * DX)


def outbreak_centers():
    """이번 outbreak 의 중심점 리스트를 반환."""
    if OUTBREAK_LOCATION == "center":
        return [(NX // 2, NY // 2)]
    if OUTBREAK_LOCATION == "multiple_points":
        return MULTI_POINTS
    if OUTBREAK_LOCATION == "random":
        m = OUTBREAK_RADIUS + 1
        return [(int(RNG.integers(m, NX - m)), int(RNG.integers(m, NY - m)))]
    raise ValueError(f"알 수 없는 OUTBREAK_LOCATION: {OUTBREAK_LOCATION}")


def inject_outbreak(S, I, yy, xx):
    """지정 위치에 감염 유입: I += strength, S -= strength (S 음수 방지)."""
    for cx, cy in outbreak_centers():
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= OUTBREAK_RADIUS ** 2
        add = np.minimum(OUTBREAK_STRENGTH, S[mask])   # S 부족분 clip → 인구 보존
        I[mask] += add
        S[mask] -= add
    np.clip(S, 0.0, None, out=S)


def run():
    # 초기: 전원 감염가능(S=1). 첫 outbreak 가 유행을 촉발.
    S = np.full((NY, NX), N)
    I = np.zeros((NY, NX))
    R = np.zeros((NY, NX))
    yy, xx = np.ogrid[:NY, :NX]

    S_avg, I_avg, R_avg = [], [], []
    trk_S = [[] for _ in TRACK_POINTS]      # 발생점별 국소 S, I 기록
    trk_I = [[] for _ in TRACK_POINTS]
    frames_I, frame_t, outbreak_t = [], [], []

    t = 0.0
    for k in range(STEPS + 1):
        # ---- 주기적 outbreak 유입 ----
        if ENABLE_PERIODIC_OUTBREAK and k % OUTBREAK_PERIOD == 0:
            inject_outbreak(S, I, yy, xx)
            outbreak_t.append(t)
        # ---- 유행 소멸: outbreak 후 CLEAR_DELAY 스텝(파동이 쓸고 간 뒤) 잔여 I 리셋 ----
        if (ENABLE_PERIODIC_OUTBREAK and CLEAR_DELAY > 0
                and k % OUTBREAK_PERIOD == CLEAR_DELAY):
            I[:] = 0.0

        # ---- 기록(매 timestep 공간 평균 + 발생점 국소값) ----
        S_avg.append(S.mean()); I_avg.append(I.mean()); R_avg.append(R.mean())
        for p, (px, py) in enumerate(TRACK_POINTS):
            trk_S[p].append(S[py, px]); trk_I[p].append(I[py, px])
        if k % FRAME_STRIDE == 0:
            frames_I.append(I.copy()); frame_t.append(t)

        # ---- ① 반응(SIRS + 출생/사망 미분방정식 한 스텝) ----
        infection = BETA * S * I / N
        recovery = GAMMA * I
        waning = OMEGA * R
        S = S + DT * (MU * N - infection + waning - MU * S)   # +μN 출생, +ωR waning
        I = I + DT * (infection - recovery - MU * I)
        R = R + DT * (recovery - waning - MU * R)             # -ωR waning

        # ---- ② 확산(라플라시안) ----
        S = S + DT * D * laplacian(S)
        I = I + DT * D * laplacian(I)
        R = R + DT * D * laplacian(R)

        for a in (S, I, R):
            np.clip(a, 0.0, None, out=a)
        t += DT

    return dict(S=np.array(S_avg), I=np.array(I_avg), R=np.array(R_avg),
                tS=[np.array(s) for s in trk_S], tI=[np.array(i) for i in trk_I],
                frames=frames_I, ftime=frame_t, otime=outbreak_t)


def fig_static(d):
    tarr = np.arange(len(d["S"])) * DT
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))

    # (좌) 발생점들의 S–I 위상궤도 (각 발생점 국소값 → 반복 loop 확인)
    xmax = max(s.max() for s in d["tS"]); ymax = max(i.max() for i in d["tI"])
    for p in range(len(TRACK_POINTS)):
        ax[0].plot(d["tS"][p], d["tI"][p], "-", color=TRACK_COLORS[p], lw=0.9,
                   alpha=0.8, label=f"Site {p+1}")
    ax[0].set_xlim(-0.03, xmax * 1.05)
    ax[0].set_ylim(-0.02, ymax * 1.08)
    ax[0].set_xlabel("S (local susceptible at site)"); ax[0].set_ylabel("I (local infected at site)")
    ax[0].set_title("S-I phase orbit at outbreak sites (periodic outbreak -> repeating loop)", fontsize=12)
    ax[0].legend(fontsize=9); ax[0].grid(True, alpha=0.25)

    # (우) S/I/R 시계열 + outbreak 시점 표시
    ax[1].plot(tarr, d["S"], color="#2f6fd0", lw=1.8, label="S_avg")
    ax[1].plot(tarr, d["I"], color="#d1352c", lw=1.8, label="I_avg")
    ax[1].plot(tarr, d["R"], color="#2e8b57", lw=1.8, label="R_avg")
    for i, ot in enumerate(d["otime"]):
        ax[1].axvline(ot, color="0.6", ls="--", lw=0.8,
                      label="outbreak" if i == 0 else None)
    ax[1].set_xlabel("Time t"); ax[1].set_ylabel("Spatial mean density")
    ax[1].set_title("S/I/R time series (waning restores S -> outbreak re-raises I)", fontsize=11.5)
    ax[1].legend(fontsize=9, ncol=2); ax[1].grid(True, alpha=0.25)

    loc = {"center": "중심", "random": "랜덤", "multiple_points": "다중점"}[OUTBREAK_LOCATION]
    fig.suptitle(f"SIRS + births(μ={MU}) + waning(ω={OMEGA}) + periodic outbreak  "
                 f"(R0≈{BETA/GAMMA:.0f}, period={OUTBREAK_PERIOD*DT:.0f}, "
                 f"strength={OUTBREAK_STRENGTH}, {len(MULTI_POINTS)} sites)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULT_DIR, "pde_periodic_outbreak_phase.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"저장: {out}")


def animate(d):
    frames, ftime = d["frames"], d["ftime"]
    tS, tI = d["tS"], d["tI"]               # 발생점 국소 궤도
    vmax = np.percentile(np.concatenate([f.ravel() for f in frames]), 99.5) or 1.0
    orb_per_frame = FRAME_STRIDE
    xmax = max(s.max() for s in tS); ymax = max(i.max() for i in tI)

    fig, (axm, axp) = plt.subplots(1, 2, figsize=(13.5, 6))
    im = axm.imshow(frames[0], cmap="inferno", origin="lower", vmin=0, vmax=vmax)
    for (px, py) in TRACK_POINTS:
        axm.plot(px, py, "o", mfc="none", mec="cyan", ms=11, mew=1.8)
    axm.set_title("Infected I(x, y, t)  (outbreak sites marked)"); axm.set_xticks([]); axm.set_yticks([])
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.02)

    axp.set_xlim(-0.03, xmax * 1.05); axp.set_ylim(-0.02, ymax * 1.08)
    axp.set_xlabel("S (local at site)"); axp.set_ylabel("I (local at site)")
    axp.set_title("S-I phase orbit at outbreak sites"); axp.grid(True, alpha=0.25)
    lines, dots = [], []
    for p in range(len(TRACK_POINTS)):
        (ln,) = axp.plot([], [], "-", color=TRACK_COLORS[p], lw=0.9, alpha=0.8)
        (dt,) = axp.plot([], [], "o", color=TRACK_COLORS[p], ms=6)
        lines.append(ln); dots.append(dt)
    sup = fig.suptitle("", fontsize=13)

    def update(f):
        im.set_data(frames[f])
        n = min(f * orb_per_frame + 1, len(tS[0]))
        for p in range(len(TRACK_POINTS)):
            lines[p].set_data(tS[p][:n], tI[p][:n])
            dots[p].set_data([tS[p][n - 1]], [tI[p][n - 1]])
        sup.set_text(f"SIRS + births + waning + periodic outbreak   t = {ftime[f]:.0f}")
        return [im, *lines, *dots, sup]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=80)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULT_DIR, "pde_periodic_outbreak.gif")
    anim.save(out, writer=PillowWriter(fps=15)); plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    print(f"R0 = β/γ = {BETA/GAMMA:.1f}, 출생 μ={MU}, waning ω={OMEGA}, D={D}")
    print(f"주기 outbreak: {ENABLE_PERIODIC_OUTBREAK}, 주기={OUTBREAK_PERIOD}스텝, "
          f"강도={OUTBREAK_STRENGTH}, 위치={OUTBREAK_LOCATION}")
    d = run()
    print(f"outbreak {len(d['otime'])}회, 프레임 {len(d['frames'])}개, 기록 {len(d['S'])}스텝")
    fig_static(d)
    animate(d)
    print("완료.")
