"""
2D 반응-확산 SIR 모델 시뮬레이션
================================================================
공간을 격자(grid)로 나누고, 각 칸마다 S/I/R 인구 밀도를 둔다.
사람들이 이동(확산)하면서 병이 공간적으로 퍼지는 모습을 본다.

지배 방정식 (연속형):
  ∂S/∂t = -β·S·I/N + D_S·ΔS
  ∂I/∂t =  β·S·I/N - γ·I + D_I·ΔI
  ∂R/∂t =  γ·I           + D_R·ΔR

  β    : 감염 전파율
  γ    : 회복율  (1/γ = 평균 감염기간)Build a 2D reaction-diffusion SIR simulation.

Use a grid-based model where S, I, and R are 2D NumPy arrays.

Start with the equations:

∂S/∂t = -βSI/N + D_S ΔS
∂I/∂t = βSI/N - γI + D_I ΔI
∂R/∂t = γI + D_R ΔR

Use finite differences for the Laplacian.
Use explicit time stepping first.
Use no-flux boundary conditions.
Track total S, I, R over time.
Generate heatmaps and animations of I(x, y, t).
Keep the code modular so that later we can add quarantine zones, spatially varying β, and population density maps.
  D_x  : 각 집단의 확산계수 [km²/일]   (사람 이동성)
  Δ    : 라플라시안 ∂²/∂x² + ∂²/∂y²  (공간 확산)

수치기법:
  · 라플라시안 → 유한차분 (5점 스텐실)
  · 시간전진   → 명시적 오일러 (explicit Euler)  ※ 먼저 가장 단순하게
  · 경계조건   → 무유출(no-flux, Neumann): 격자 밖으로 사람이 새지 않음

모듈 구조:
  나중에 아래를 쉽게 끼워넣을 수 있도록 분리해 두었다.
    - 격리구역(quarantine zone)      → self.mobility 마스크
    - 공간적으로 변하는 β             → self.beta 를 2D 배열로
    - 인구밀도 지도(density map)      → self.N 을 2D 배열로
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 모든 결과 png/gif 는 result/ 폴더에 저장한다
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
os.makedirs(RESULT_DIR, exist_ok=True)


# ==================================================================
# 1. 수치 도구  (공간 미분)
# ==================================================================
def laplacian(f, dx):
    """5점 스텐실 라플라시안 + 무유출(Neumann) 경계.

    np.pad(mode="edge") 로 경계 바깥에 '유령칸'을 만들되 경계값을 그대로 복사한다.
    → 경계를 가로지르는 기울기가 0 이 되어 사람이 밖으로 새지 않는다(no-flux).
    """
    fp = np.pad(f, 1, mode="edge")
    lap = (
        fp[:-2, 1:-1]   # 위
        + fp[2:, 1:-1]  # 아래
        + fp[1:-1, :-2] # 왼쪽
        + fp[1:-1, 2:]  # 오른쪽
        - 4.0 * f
    ) / (dx * dx)
    return lap


# ==================================================================
# 2. 모델 본체  (모듈화된 클래스)
# ==================================================================
class SpatialSIR:
    """격자 위의 반응-확산 SIR 모델.

    S, I, R 은 (ny, nx) 2D 배열(각 칸의 인구 밀도).
    beta, N, mobility 는 스칼라 또는 2D 배열 모두 받을 수 있어
    공간적으로 변하는 β·인구밀도·격리구역을 그대로 표현한다.
    """

    def __init__(self, nx, ny, dx, beta, gamma, D_S, D_I, D_R, N):
        self.nx, self.ny, self.dx = nx, ny, dx
        self.gamma = gamma
        self.D_S, self.D_I, self.D_R = D_S, D_I, D_R

        # 스칼라를 넣어도 2D 배열로 승격 → 나중에 지도로 바꾸기 쉽다
        self.beta = np.broadcast_to(np.asarray(beta, float), (ny, nx)).copy()
        self.N = np.broadcast_to(np.asarray(N, float), (ny, nx)).copy()
        self.N_safe = np.where(self.N > 0, self.N, 1.0)  # 0 나눗셈 방지

        # 이동성 마스크: 1=자유 이동, 0=완전 봉쇄(격리구역 자리)
        self.mobility = np.ones((ny, nx))

        # 초기 상태: 모두 감염가능(S), 감염/회복 0
        self.S = self.N.copy()
        self.I = np.zeros((ny, nx))
        self.R = np.zeros((ny, nx))

    # ---- 초기 조건 시스템(initial_conditions)에서 만든 상태 주입 ----
    def set_initial_state(self, S, I, R, N):
        """create_initial_conditions() 가 반환한 (S, I, R, N) 을 그대로 싣는다.

        모델 격자 크기와 배열 모양이 일치해야 한다. 인구밀도 N 도 함께
        갱신하므로 공간적으로 변하는 인구분포를 그대로 반영한다."""
        assert S.shape == (self.ny, self.nx), "격자 크기 불일치"
        self.S = S.astype(float).copy()
        self.I = I.astype(float).copy()
        self.R = R.astype(float).copy()
        self.N = N.astype(float).copy()
        self.N_safe = np.where(self.N > 0, self.N, 1.0)
        return self

    # ---- 초기 감염 씨앗 ----
    def seed_infection(self, cx, cy, amount=1.0, radius=2):
        """(cx, cy) 중심 반경 radius 안의 칸에 감염자를 심는다(S→I)."""
        yy, xx = np.ogrid[: self.ny, : self.nx]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        seed = np.where(mask, amount, 0.0)
        seed = np.minimum(seed, self.S)  # S 재고보다 많이 감염시키지 않음
        self.I += seed
        self.S -= seed

    # ---- 확산계수에 이동성 마스크 적용 (격리구역 훅) ----
    def _diffuse(self, f, D):
        # mobility 로 국소 확산을 줄일 수 있다(0 이면 그 칸으로/에서 이동 없음)
        return D * self.mobility * laplacian(f, self.dx)

    # ---- 명시적 오일러 한 스텝 ----
    def step(self, dt):
        infection = self.beta * self.S * self.I / self.N_safe
        recovery = self.gamma * self.I

        dS = -infection + self._diffuse(self.S, self.D_S)
        dI = infection - recovery + self._diffuse(self.I, self.D_I)
        dR = recovery + self._diffuse(self.R, self.D_R)

        self.S += dt * dS
        self.I += dt * dI
        self.R += dt * dR

        # 수치오차로 인한 음수 방지
        np.clip(self.S, 0.0, None, out=self.S)
        np.clip(self.I, 0.0, None, out=self.I)
        np.clip(self.R, 0.0, None, out=self.R)

    def totals(self):
        return self.S.sum(), self.I.sum(), self.R.sum()

    # ---- CFL 안정성 점검 (명시적 확산의 필요조건) ----
    def stability_dt(self):
        Dmax = max(self.D_S, self.D_I, self.D_R)
        return self.dx * self.dx / (4.0 * Dmax)


# ==================================================================
# 3. 시뮬레이션 설정 & 실행
# ==================================================================
# ---- 격자 / 시간 ----
NX = NY = 120        # 격자 칸 수 (120 x 120 km)
DX = 1.0             # 칸 크기 [km]
DAYS = 250.          # 시뮬레이션 기간 [일]
DT = 0.1             # 시간 간격 [일]

# ---- 역학 파라미터 ----
BETA = 0.35          # 감염 전파율 β
GAMMA = 0.10         # 회복율 γ (평균 감염기간 10일)  → R0 = β/γ = 3.5
N_PER_CELL = 500.0   # 칸당 인구밀도 [명/km²]

# ---- 확산계수 [km²/일] (요구값) ----
D_S = 0.4
D_I = 0.02           # 감염자는 덜 움직인다고 가정 → 훨씬 느린 확산
D_R = 0.4


def run():
    model = SpatialSIR(NX, NY, DX, BETA, GAMMA, D_S, D_I, D_R, N_PER_CELL)

    # 안정성 확인
    dt_max = model.stability_dt()
    print(f"CFL 안정조건: dt ≤ {dt_max:.3f} 일  (사용 dt = {DT} 일) "
          f"→ {'안정 OK' if DT <= dt_max else '불안정 위험!'}")

    # 도시 한복판에 최초 감염 발생
    model.seed_infection(cx=NX // 2, cy=NY // 2, amount=5.0, radius=2)

    steps = int(DAYS / DT)
    times = np.zeros(steps + 1)
    tot_S = np.zeros(steps + 1)
    tot_I = np.zeros(steps + 1)
    tot_R = np.zeros(steps + 1)
    tot_S[0], tot_I[0], tot_R[0] = model.totals()

    # 애니메이션/스냅샷용 I(x,y) 프레임 저장
    frame_every = 20                 # 20스텝(=2일)마다 프레임 저장
    frames, frame_times = [model.I.copy()], [0.0]

    for k in range(steps):
        model.step(DT)
        t = (k + 1) * DT
        times[k + 1] = t
        tot_S[k + 1], tot_I[k + 1], tot_R[k + 1] = model.totals()
        if (k + 1) % frame_every == 0:
            frames.append(model.I.copy())
            frame_times.append(t)

    return model, times, tot_S, tot_I, tot_R, frames, frame_times


# ==================================================================
# 4. 시각화  (결과는 모두 result/ 에 저장)
# ==================================================================
def plot_totals(times, tot_S, tot_I, tot_R):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, tot_S, color="#2f6fd0", lw=2.2, label="S Susceptible")
    ax.plot(times, tot_I, color="#d1352c", lw=2.2, label="I Infected")
    ax.plot(times, tot_R, color="#2e8b57", lw=2.2, label="R Recovered")
    ax.fill_between(times, 0, tot_I, color="#d1352c", alpha=0.12)

    peak = int(np.argmax(tot_I))
    ax.plot(times[peak], tot_I[peak], "o", color="#d1352c", ms=7, zorder=5)
    ax.annotate(f"Peak day {times[peak]:.0f}\n{tot_I[peak]:,.0f} people",
                xy=(times[peak], tot_I[peak]),
                xytext=(times[peak] + 8, tot_I[peak]),
                fontsize=10, color="#d1352c",
                arrowprops=dict(arrowstyle="->", color="#d1352c"))

    ax.set_title(f"2D Reaction-Diffusion SIR Total Population "
                 f"(R0={BETA/GAMMA:.1f}, D_I={D_I})", fontsize=13)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Total population")
    ax.set_xlim(0, DAYS)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "sir2d_totals.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"저장: {out}")


def plot_heatmaps(frames, frame_times, vmax):
    """여러 시점의 I(x,y) 스냅샷을 한 장에."""
    want_days = [0, 10, 25, 45, 70, DAYS]
    picks = [min(range(len(frame_times)),
                 key=lambda i: abs(frame_times[i] - d)) for d in want_days]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for ax, idx in zip(axes.ravel(), picks):
        im = ax.imshow(frames[idx], cmap="inferno", origin="lower",
                       vmin=0, vmax=vmax,
                       extent=[0, NX * DX, 0, NY * DX])
        ax.set_title(f"t = {frame_times[idx]:.0f} days", fontsize=11)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
    fig.suptitle("Infected Density I(x, y, t) Snapshots Over Time", fontsize=14)
    fig.subplots_adjust(right=0.9)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Infected density [people/km²]")
    out = os.path.join(RESULT_DIR, "sir2d_infected_heatmaps.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"저장: {out}")


def make_animation(frames, frame_times, vmax):
    """I(x,y,t) 확산 애니메이션 → GIF."""
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
        title.set_text(f"Infection Spread  I(x, y, t)    t = {frame_times[i]:.0f} days")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out = os.path.join(RESULT_DIR, "sir2d_infection.gif")
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"저장: {out}")


# ==================================================================
# 5. 메인
# ==================================================================
if __name__ == "__main__":
    model, times, tot_S, tot_I, tot_R, frames, frame_times = run()

    # 요약 수치
    R0 = BETA / GAMMA
    peak = int(np.argmax(tot_I))
    print(f"기초감염재생산수 R0 = {R0:.2f}")
    print(f"감염 정점: {times[peak]:.1f}일차, 동시감염 {tot_I[peak]:,.0f}명")
    print(f"최종 누적 감염(=R): {tot_R[-1]:,.0f}명 "
          f"(전체의 {tot_R[-1]/(tot_S[-1]+tot_R[-1]+tot_I[-1])*100:.1f}%)")

    # 색상 스케일은 전체 프레임 최댓값 기준으로 통일
    vmax = max(f.max() for f in frames)

    plot_totals(times, tot_S, tot_I, tot_R)
    plot_heatmaps(frames, frame_times, vmax)
    make_animation(frames, frame_times, vmax)
    print("완료.")
