"""
pde_recurrent_sir.py
================================================================
2D 반응-확산 SIR (넓은 격자) + '면역 소실 후 재발하는 유행' + 4점의 S-I 위상궤도.

구성
  1) 4개의 발생점에서 감염(I)이 시작.
  2) 매 타임스텝마다 '2단계 연산':
        ① 반응(reaction): 각 칸에서 SIR 미분방정식 한 스텝
             dS = -βSI/N ,  dI = βSI/N - γI ,  dR = γI
        ② 확산(diffusion): 라플라시안으로 S/I/R 을 퍼뜨림 (무유출 경계)
       → "미방 연산 후 laplacian 확산" 순서의 연산자 분리(operator splitting).
  3) 재발 메커니즘(면역 소실):
        · 유행이 잦아들면(전체 I 낮아짐) 격자의 잔여 I 를 제거하고 '면역 소실기' 진입.
        · 면역 소실기 동안 R 이 정해진 기간(WANING_PERIOD)에 걸쳐 서서히 S 로 복귀.
        · R 이 거의 다 S 로 돌아가면(모두 감염가능) 비로소 '새 질병'이 4점에서 재발.
       → 각 재발은 유행 1 과 똑같은 깨끗한 4점에서 같은 속도로 출발.
  4) 시각화:
        (좌) I(x, y, t) 확산 히트맵 + 추적점 4개 표시
        (우) 추적점 4개의 (S, I) 위상궤도 — 시간축이 아니라 S-I 평면의 궤도.
             면역 소실기엔 I≈0 인 채 S 가 오른쪽으로 회복(R→S) → 궤도가 닫히며 순환.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- 격자 / 시간 ----------------
NX = NY = 160
DX = 1.0
DT = 0.15
N0 = 1000.0

# ---------------- 역학 파라미터 ----------------
BETA = 0.6
GAMMA = 0.12           # R0 = β/γ = 5

# ---------------- 확산계수 ----------------
D_S = 0.5
D_I = 0.15
D_R = 0.5

# ---------------- 면역 소실 / 재발 ----------------
MAX_OUTBREAKS = 4
SEED_AMOUNT = 60.0
SEED_RADIUS = 3
PEAK_FRAC = 0.02       # 전체 I 비율이 이 위면 '유행이 제대로 붙음'
LOW_FRAC = 0.001       # 이 아래면 '유행 종료' → 면역 소실기 진입
WANING_PERIOD = 12.0   # R 이 S 로 돌아가는 기간(e-folding, 일). 이 기간에 걸쳐 서서히 복귀
R_LOW_FRAC = 0.03      # R 비율이 이 아래면 '모두 S 로 복귀' → 새 질병 발병
OMEGA = 1.0 / WANING_PERIOD

MAX_STEPS = 16000
FRAME_STRIDE = 30

# 4개 발생점(좌하, 우하, 좌상, 우상)
SEEDS = [(NX // 2, NY // 2)]          # 단일 발생점(중심)

# 추적점 = 발생점(중심)
TRACK_POINTS = [
    {"xy": SEEDS[0], "c": "#22d3ee", "label": "Outbreak site (center)"},
]


def laplacian(f):
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f) / (DX * DX)


class RecurrentSIR:
    def __init__(self):
        self.S = np.full((NY, NX), N0)
        self.I = np.zeros((NY, NX))
        self.R = np.zeros((NY, NX))
        self.t = 0.0
        self.outbreaks = 1
        self.active = False      # 유행이 피크를 넘겼는지
        self.waiting = False     # 면역 소실기(R→S 진행 중)
        self.seed_points()

    def seed_points(self):
        yy, xx = np.ogrid[:NY, :NX]
        for cx, cy in SEEDS:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SEED_RADIUS ** 2
            add = np.where(mask, SEED_AMOUNT, 0.0)
            add = np.minimum(add, self.S)
            self.I += add
            self.S -= add

    def status(self):
        if self.waiting:
            return "Immunity waning (R->S)"
        return f"Outbreak #{self.outbreaks} spreading"

    def step(self):
        S, I, R = self.S, self.I, self.R
        N = S + I + R
        Nsafe = np.where(N > 0, N, 1.0)

        # ① 반응(미방)
        infection = BETA * S * I / Nsafe
        recovery = GAMMA * I
        S = S + DT * (-infection)
        I = I + DT * (infection - recovery)
        R = R + DT * (recovery)

        # 면역 소실기: R 이 정해진 기간에 걸쳐 S 로 복귀 (유행 중엔 면역 유지)
        if self.waiting:
            wf = DT * OMEGA * R
            S = S + wf
            R = R - wf

        # ② 확산(라플라시안)
        S = S + DT * D_S * laplacian(S)
        I = I + DT * D_I * laplacian(I)
        R = R + DT * D_R * laplacian(R)

        np.clip(S, 0.0, None, out=S)
        np.clip(I, 0.0, None, out=I)
        np.clip(R, 0.0, None, out=R)
        self.S, self.I, self.R = S, I, R
        self.t += DT

        # ---- 상태 전이 ----
        totN = (S + I + R).sum()
        fracI = I.sum() / totN
        fracR = R.sum() / totN

        if not self.active and not self.waiting and fracI > PEAK_FRAC:
            self.active = True
        if self.active and fracI < LOW_FRAC:
            # 유행 종료: 잔여 I 를 S 로 정리하고 면역 소실기 시작
            self.S += self.I
            self.I[:] = 0.0
            self.active = False
            self.waiting = True
        if self.waiting and fracR < R_LOW_FRAC and self.outbreaks < MAX_OUTBREAKS:
            # 모두 S 로 복귀 → 새 질병 발병
            self.seed_points()
            self.outbreaks += 1
            self.waiting = False


def run():
    m = RecurrentSIR()
    n_pts = len(TRACK_POINTS)
    trk_S = [[] for _ in range(n_pts)]
    trk_I = [[] for _ in range(n_pts)]
    frames_I, frame_t, frame_status = [], [], []

    def record_track():
        for k, p in enumerate(TRACK_POINTS):
            x, y = p["xy"]
            trk_S[k].append(m.S[y, x])
            trk_I[k].append(m.I[y, x])

    record_track()
    frames_I.append(m.I.copy()); frame_t.append(0.0); frame_status.append(m.status())

    step = 0
    while step < MAX_STEPS:
        m.step(); step += 1
        record_track()
        if step % FRAME_STRIDE == 0:
            frames_I.append(m.I.copy()); frame_t.append(m.t); frame_status.append(m.status())
        # 마지막 유행이 끝나 면역 소실기에 들어가면 종료
        if m.outbreaks >= MAX_OUTBREAKS and m.waiting:
            frames_I.append(m.I.copy()); frame_t.append(m.t); frame_status.append(m.status())
            break

    print(f"총 {step}스텝(t={m.t:.0f}), 유행 {m.outbreaks}회, 프레임 {len(frames_I)}개")
    return (frames_I, frame_t, frame_status,
            [np.array(s) for s in trk_S], [np.array(i) for i in trk_I])


def animate(frames_I, frame_t, frame_status, trk_S, trk_I):
    vmax = max(f.max() for f in frames_I) * 0.9
    smax = max(s.max() for s in trk_S)
    imax = max(i.max() for i in trk_I)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
    axm = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])

    # (좌) I 히트맵 + 추적점 4개
    im = axm.imshow(frames_I[0], cmap="inferno", origin="lower",
                    vmin=0, vmax=vmax, extent=[0, NX, 0, NY])
    for p in TRACK_POINTS:
        x, y = p["xy"]
        axm.plot(x, y, "o", mfc="none", mec=p["c"], ms=13, mew=2.2)
    axm.set_xlabel("x"); axm.set_ylabel("y")
    fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04).set_label("Infected density I")
    map_title = axm.set_title("")

    # (우) 4점의 (S, I) 위상궤도
    axp.set_xlim(0, max(smax, N0) * 1.05)
    axp.set_ylim(0, imax * 1.1)
    axp.set_xlabel("S (susceptible)"); axp.set_ylabel("I (infected)")
    axp.set_title("S-I phase orbit at outbreak site (center)")
    axp.grid(True, alpha=0.25)
    trails, dots = [], []
    for p in TRACK_POINTS:
        (tr,) = axp.plot([], [], "-", color=p["c"], lw=1.3, alpha=0.8, label=p["label"])
        (dt_,) = axp.plot([], [], "o", color=p["c"], ms=9)
        trails.append(tr); dots.append(dt_)
    axp.legend(fontsize=9, loc="upper right")

    def update(i):
        im.set_data(frames_I[i])
        map_title.set_text(f"I(x, y)   t = {frame_t[i]:.0f}   {frame_status[i]}")
        n = min(i * FRAME_STRIDE + 1, len(trk_S[0]))
        for k in range(len(TRACK_POINTS)):
            trails[k].set_data(trk_S[k][:n], trk_I[k][:n])
            dots[k].set_data([trk_S[k][n - 1]], [trk_I[k][n - 1]])
        return [im, map_title, *trails, *dots]

    anim = FuncAnimation(fig, update, frames=len(frames_I), blit=False, interval=120)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "pde_recurrent_sir.gif")
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"저장: {out}")

    # 최종 위상궤도 정지 이미지
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    for k, p in enumerate(TRACK_POINTS):
        ax2.plot(trk_S[k], trk_I[k], "-", color=p["c"], lw=1.0, alpha=0.8, label=p["label"])
    ax2.set_xlabel("S (susceptible)"); ax2.set_ylabel("I (infected)")
    ax2.set_title("S-I phase orbit at outbreak site (center) (waning -> recurrence cycle)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    out2 = os.path.join(RESULT_DIR, "pde_recurrent_phase.png")
    fig2.savefig(out2, dpi=140)
    plt.close(fig2)
    print(f"저장: {out2}")


if __name__ == "__main__":
    data = run()
    animate(*data)
    print("완료.")
