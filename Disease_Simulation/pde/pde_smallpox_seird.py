"""
pde_smallpox_seird.py  (SEIQR — 명시적 격리구획 Q, 사망은 '제거항'으로만)
================================================================
천연두(smallpox)를 2D 반응-확산 모델로 재발(recurrent) 시뮬레이션.
천연두는 증상(발진)이 뚜렷 → 발현자의 상당수가 발견되어 '격리'된다.
격리를 유효 β 감소로 뭉뚱그리지 않고, 격리구획 Q 를 명시적으로 둔다.
사망은 별도 D 구획을 두지 않고 '제거항'으로만 반영한다(종료자 중 CFR 은 인구에서 사라짐).

구획  S → E → I → (Q) → R   (+ 사망은 종료 시 CFR 만큼 인구에서 제거)
  S : 감염가능
  E : 잠복기(감염됐으나 전염 X). 평균 8일 뒤 증상 발현
  I : 증상 감염(전염 O) — 아직 격리 안 됨
  Q : 격리됨 (전염 X 또는 아주 약함). 여전히 앓는 중 → 이후 종료
  R : 회복/면역

전이 규칙 (일 단위)
  · S→E : 감염력 β0·S·(I + ISO_LEAK·Q)/N   (전염원 = 격리 안 된 I, 격리자는 β 거의 0)
  · E→I : 평균 8일(σ = 1/8)
  · I→Q : 격리율 ISO_RATE. 증상 뚜렷 → 발현자의 ISO_FRAC(=80%)가 결국 격리된다.
          경쟁률(격리 vs 자연종료)로 ISO_FRAC 달성:  ISO_RATE = γ·ISO_FRAC/(1-ISO_FRAC)
          → 평균 ISO_DELAY(≈3.5일) 만에 격리, 그 사이에만 전염.
  · I,Q 종료 : 평균 14일(γ = 1/14). 종료자 중 (1-CFR) 만 R 로, CFR(30%) 은 사망→인구에서 제거.

격리 효과
  격리 전(I) 에만 전염 → 격리로 I 가 빨리 빠져나가 '유효 감염기간'이
  1/γ(14일) → 1/(ISO_RATE+γ)(≈2.8일) 로 줄어 실효 재생산수가 급감한다:
      R0(무대책) = β0/γ = 21
      R0(격리)   = β0/(ISO_RATE+γ) + β0·ISO_LEAK/γ ≈ 4.2

수치기법(매 스텝 2단계)
  ① 반응: 각 칸의 SEIQR 미분방정식 한 스텝(명시적 오일러). 사망은 R 로 안 가는 제거항.
  ② 확산: 라플라시안(무유출 경계). Q(격리자)는 거의 안 움직임.

재발: 유행이 끝나면(E+I+Q 소멸) 인구를 초기화(새 발병)하고 중심 재파종.
시각화: (좌) E/I/Q/R 구획 지도 / (우) 그리드 중앙의 S–I 위상궤도.
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

# ---------------- 천연두 역학 파라미터(일 단위) ----------------
BETA0 = 1.5            # 기본 감염력(격리 안 된 I)
INCUBATION = 8.0       # 잠복기(E) 평균 일수
SYMPTOM_DUR = 14.0     # 증상(I/Q) 지속 평균 일수(발현 후)
CFR = 0.30             # 치사율: 종료 시 30% 사망(제거), 70% 회복(R)
SIGMA = 1.0 / INCUBATION
GAMMA = 1.0 / SYMPTOM_DUR

# ---------------- 격리(quarantine) 파라미터 ----------------
ISO_FRAC = 0.80        # 증상 발현자 중 결국 격리되는 비율(증상 뚜렷 → 높음)
ISO_RATE = GAMMA * ISO_FRAC / (1.0 - ISO_FRAC)   # I→Q 격리율(경쟁률로 ISO_FRAC 달성)
ISO_DELAY = 1.0 / ISO_RATE                        # 평균 격리까지 걸리는 일수(≈3.5일)
ISO_LEAK = 0.0         # 격리자의 잔여 전염력 배수(0=완전격리, 예: 0.05=불완전). β_Q=β0·ISO_LEAK

# 실효 재생산수(참고)
R0_RAW = BETA0 / GAMMA
R0_ISO = BETA0 / (ISO_RATE + GAMMA) + BETA0 * ISO_LEAK / GAMMA

# ---------------- 확산계수 (Q(격리자)는 이동 없음) ----------------
D_S = 0.5
D_E = 0.5
D_I = 0.15
D_R = 0.5

# ---------------- 재발 ----------------
MAX_OUTBREAKS = 2
SEED_AMOUNT = 40.0
SEED_RADIUS = 3
PEAK_FRAC = 0.01
LOW_FRAC = 0.0004

MAX_STEPS = 20000
FRAME_STRIDE = 55

SEEDS = [(NX // 2, NY // 2)]          # 단일 발생점(중심)
TRACK = (NX // 2, NY // 2)            # 추적점 = 그리드 중앙

COMP_COLOR = {"S": "#2f6fd0", "E": "#e58a1a", "I": "#d1352c",
              "Q": "#8e44ad", "R": "#2e8b57"}


def laplacian(f):
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f) / (DX * DX)


class SmallpoxSEIQR:
    def __init__(self):
        self.reset_population()
        self.t = 0.0
        self.outbreaks = 1
        self.active = False
        self.seed_points()

    def reset_population(self):
        self.S = np.full((NY, NX), N0)
        self.E = np.zeros((NY, NX))
        self.I = np.zeros((NY, NX))
        self.Q = np.zeros((NY, NX))
        self.R = np.zeros((NY, NX))

    def seed_points(self):
        yy, xx = np.ogrid[:NY, :NX]
        for cx, cy in SEEDS:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SEED_RADIUS ** 2
            add = np.minimum(np.where(mask, SEED_AMOUNT, 0.0), self.S)
            self.I += add
            self.S -= add

    def step(self):
        S, E, I, Q, R = self.S, self.E, self.I, self.Q, self.R
        N = S + E + I + Q + R                    # 살아있는 인구(격리자 포함)
        Nsafe = np.where(N > 0, N, 1.0)

        # ① 반응(SEIQR). 사망은 R 로 안 가는 제거항(종료자 중 CFR 은 인구에서 사라짐)
        force = BETA0 * S * (I + ISO_LEAK * Q) / Nsafe   # 전염원 = 격리 안 된 I(+Q 누출)
        onset = SIGMA * E                        # E→I
        iso = ISO_RATE * I                       # I→Q (격리)
        end_I = GAMMA * I                        # I 종료(격리 전)
        end_Q = GAMMA * Q                        # Q 종료(격리 중)

        S = S + DT * (-force)
        E = E + DT * (force - onset)
        I = I + DT * (onset - iso - end_I)
        Q = Q + DT * (iso - end_Q)
        R = R + DT * ((1.0 - CFR) * (end_I + end_Q))     # 생존자만 R (CFR 은 사망=제거)

        # ② 확산(무유출 경계). Q(격리자)는 이동 없음 → 확산 안 함
        S = S + DT * D_S * laplacian(S)
        E = E + DT * D_E * laplacian(E)
        I = I + DT * D_I * laplacian(I)
        R = R + DT * D_R * laplacian(R)

        for arr in (S, E, I, Q, R):
            np.clip(arr, 0.0, None, out=arr)
        self.S, self.E, self.I, self.Q, self.R = S, E, I, Q, R
        self.t += DT

        # ---- 재발 판정: 감염군(E+I+Q) 소멸 시 새 발병 ----
        totN = (S + E + I + Q + R).sum()
        frac_inf = (E + I + Q).sum() / totN
        if not self.active and frac_inf > PEAK_FRAC:
            self.active = True
        if self.active and frac_inf < LOW_FRAC and self.outbreaks < MAX_OUTBREAKS:
            self.reset_population()
            self.seed_points()
            self.outbreaks += 1
            self.active = False


def run():
    m = SmallpoxSEIQR()
    frames = {k: [] for k in "EIQR"}
    frame_t = []
    trk_S, trk_I = [], []
    totals = {k: [] for k in "SEIQR"}

    def record():
        trk_S.append(m.S[TRACK[1], TRACK[0]]); trk_I.append(m.I[TRACK[1], TRACK[0]])
        for k, arr in zip("SEIQR", (m.S, m.E, m.I, m.Q, m.R)):
            totals[k].append(arr.sum())

    def snap():
        for k, arr in zip("EIQR", (m.E, m.I, m.Q, m.R)):
            frames[k].append(arr.copy())
        frame_t.append(m.t)

    record(); snap()
    step = 0
    while step < MAX_STEPS:
        m.step(); step += 1
        record()
        if step % FRAME_STRIDE == 0:
            snap()
        totN = (m.S + m.E + m.I + m.Q + m.R).sum()
        if (m.outbreaks >= MAX_OUTBREAKS and m.active
                and (m.E + m.I + m.Q).sum() / totN < LOW_FRAC):
            snap(); break

    print(f"총 {step}스텝(t={m.t:.0f}일), 유행 {m.outbreaks}회, 프레임 {len(frame_t)}개")
    print(f"격리: 발현자의 {ISO_FRAC*100:.0f}% 가 평균 {ISO_DELAY:.1f}일 뒤 격리(β_Q=β0·{ISO_LEAK})")
    print(f"실효 R0: 무대책 {R0_RAW:.0f} → 격리 {R0_ISO:.1f}")
    return frames, frame_t, np.array(trk_S), np.array(trk_I), totals


def plot_totals(totals):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(totals["S"])) * DT
    for k, lab in [("S", "S susceptible"), ("E", "E exposed (non-infectious)"), ("I", "I symptomatic (infectious)"),
                   ("Q", "Q quarantined (non-infectious)"), ("R", "R recovered")]:
        ax.plot(x, totals[k], color=COMP_COLOR[k], lw=2, label=lab)
    living = sum(np.array(totals[k]) for k in "SEIQR")
    ax.plot(x, living, color="0.5", lw=1.3, ls="--", label="Living population (S+E+I+Q+R)")
    ax.set_title(f"Smallpox SEIQR totals — quarantine {ISO_FRAC*100:.0f}% (avg {ISO_DELAY:.1f} days later), "
                 f"CFR {CFR*100:.0f}% (removed)  |  effective R0 {R0_RAW:.0f}->{R0_ISO:.1f}")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Population")
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "pde_smallpox_seird_totals.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"저장: {out}")


def animate(frames, frame_t, trk_S, trk_I, totals):
    comps = [("I", "I symptomatic (infectious)", "inferno"), ("Q", "Q quarantined (non-infectious, static)", "Purples")]
    nframe = len(frame_t)
    smax = trk_S.max(); imax = trk_I.max() if trk_I.max() > 0 else 1

    fig = plt.figure(figsize=(16, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
    axes = {"I": fig.add_subplot(gs[0, 0]), "Q": fig.add_subplot(gs[0, 1])}
    axp = fig.add_subplot(gs[0, 2])

    ims = {}
    for key, title, cmap in comps:
        vm = max((f.max() for f in frames[key]), default=1) * 0.9 or 1.0
        ims[key] = axes[key].imshow(frames[key][0], cmap=cmap, origin="lower",
                                    vmin=0, vmax=vm, extent=[0, NX, 0, NY])
        axes[key].set_title(title, fontsize=12)
        axes[key].set_xticks([]); axes[key].set_yticks([])
        fig.colorbar(ims[key], ax=axes[key], fraction=0.046, pad=0.02)
    axes["I"].plot(TRACK[0], TRACK[1], "o", mfc="none", mec="#22d3ee", ms=10, mew=1.8)
    suptitle = fig.suptitle("", fontsize=13)

    axp.set_xlim(0, max(smax, N0) * 1.05); axp.set_ylim(0, imax * 1.1)
    axp.set_xlabel("S (susceptible)"); axp.set_ylabel("I (symptomatic infected)")
    axp.set_title("S-I phase orbit at grid center")
    axp.grid(True, alpha=0.25)
    (trail,) = axp.plot([], [], "-", color="#22d3ee", lw=1.4, alpha=0.85)
    (dot,) = axp.plot([], [], "o", color="#22d3ee", ms=9)

    def update(i):
        for key, _, _ in comps:
            ims[key].set_data(frames[key][i])
        suptitle.set_text(f"Smallpox SEIQR   t = {frame_t[i]:.0f} days   "
                          f"(quarantine {ISO_FRAC*100:.0f}%, effective R0 {R0_RAW:.0f}->{R0_ISO:.1f})")
        n = min(i * FRAME_STRIDE + 1, len(trk_S))
        trail.set_data(trk_S[:n], trk_I[:n]); dot.set_data([trk_S[n - 1]], [trk_I[n - 1]])
        return [*ims.values(), suptitle, trail, dot]

    anim = FuncAnimation(fig, update, frames=nframe, blit=False, interval=120)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULT_DIR, "pde_smallpox_seird.gif")
    anim.save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    frames, frame_t, trk_S, trk_I, totals = run()
    plot_totals(totals)
    animate(frames, frame_t, trk_S, trk_I, totals)
    qmax = max(totals["Q"]) if totals["Q"] else 0
    print(f"격리자(Q) 최대 {qmax:.0f}명  (사망은 종료자의 {CFR*100:.0f}% 로 인구에서 제거됨)")
    print("완료.")
