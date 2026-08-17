"""
pde_polio_saiv.py
================================================================
소아마비(polio) 2D 반응-확산 SAIRV 모델 + '증상 발생 반응성 대량 백신'.

구획  S → (A 또는 I) → R,  그리고 V(백신)
  S : 감염가능
  A : 무증상 감염(asymptomatic). 전염 O. 무증상이라 평소처럼 이동 (D_A = D_S)
  I : 증상 감염(symptomatic/마비). 전염 O. 증상으로 잘 못 움직임 (D_I ≪ D_S)
  R : 회복/면역
  V : 백신 접종(면역)

감염 분리(폴리오 특성)
  새 감염 중 SYMPTOMATIC_FRAC(=0.05, 5%)만 증상(I)으로 나타나고, 95%는 무증상(A).
  → 대부분의 전파가 '조용히'(A) 일어나고, 증상(I)은 드물게 수면 위로 드러나는 신호.

반응성 대량 백신  ★핵심
  증상 감염 I 가 나타난 칸 '근방(RESPONSE_RADIUS)'의 S 를 매 스텝 대량으로 V 로.
  = 마비 사례가 보고될 때마다 그 주변에 긴급 집단 접종을 퍼붓는 실제 폴리오 대응.
  단, A·I 를 직접 건드리지 않고 S→V 만 한다(감염자를 지우지 않음).

수치기법: 매 스텝 ① 반응 ② 확산 ③ 반응성 백신. 인구 보존·전이 clamp.
시각화: A / I / V 히트맵(S·R 은 생략) + 총량 그래프.
"""

import os
from collections import deque
import numpy as np
from scipy.ndimage import distance_transform_edt
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
BETA = 0.6                 # 감염 전파율(전염원 = A + I)
GAMMA = 0.12               # 회복율 (R0 = β/γ = 5)
SYMPTOMATIC_FRAC = 0.05    # 새 감염 중 증상(I) 비율(폴리오 ≈ 5%)

# ---------------- 확산계수 ----------------
D_S = 0.3
D_A = 0.3                  # = D_S (무증상은 평소처럼 이동)
D_I = 0.05                 # 증상(마비)은 이동 거의 못 함
D_R = 0.3
D_V = 0.0                  # 접종된 곳은 고정(패치가 번지지 않게)

# ---------------- 반응성 대량 백신 ----------------
I_DETECT = 1.0             # 이 이상 I 가 있으면 '증상 사례 발견'
RESPONSE_RADIUS = 7        # 발견 지점 근방 이 반경(칸) 안에 접종
VAX_RESPONSE_RATE = 0.10   # 반응 구역에서 매 스텝 S 의 이 비율을 V 로(수일에 걸쳐 대량)
RESPONSE_DELAY_DAYS = 5.0  # ★증상 발견 후 백신 배포까지 지연(일). 0 이면 즉시 대응
RESPONSE_DELAY_STEPS = int(round(RESPONSE_DELAY_DAYS / DT))

# ---------------- 발생점 ----------------
SEEDS = [(NX // 2, NY // 2)]          # 단일 발생점(중심)
SEED_A = 50.0
SEED_RADIUS = 2

MAX_STEPS = 1600
FRAME_STRIDE = 10
COMP_COLOR = {"A": "#e58a1a", "I": "#d1352c", "V": "#8e44ad", "total": "0.4"}


def laplacian(f):
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f) / (DX * DX)


class PolioSAIV:
    def __init__(self):
        self.S = np.full((NY, NX), N0)
        self.A = np.zeros((NY, NX))
        self.I = np.zeros((NY, NX))
        self.R = np.zeros((NY, NX))
        self.V = np.zeros((NY, NX))
        self.t = 0.0
        # 증상 감지 마스크 이력(지연 대응용): 길이 = 지연 스텝 + 1
        self.detect_buffer = deque(maxlen=RESPONSE_DELAY_STEPS + 1)
        self.seed_points()

    def seed_points(self):
        yy, xx = np.ogrid[:NY, :NX]
        for cx, cy in SEEDS:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SEED_RADIUS ** 2
            add = np.minimum(np.where(mask, SEED_A, 0.0), self.S)   # 초기엔 무증상 A 로 시작
            self.A += add
            self.S -= add

    def reactive_vaccination(self):
        """증상 감염 I 가 나타난 칸 근방의 S 를 대량으로 V 로(반응성 접종).
        단, 발견 후 RESPONSE_DELAY_DAYS(일) '지연' 뒤에 배포한다.
        → 지연 동안 조용한 A 가 더 퍼져 방역이 뒤처진다.
        A·I 는 건드리지 않는다(감염자 제거 금지). 오직 S↓, V↑."""
        # 현재 증상 감지 마스크를 이력에 저장
        self.detect_buffer.append(self.I > I_DETECT)
        # 지연 스텝만큼 이력이 쌓여야 대응 시작
        if len(self.detect_buffer) <= RESPONSE_DELAY_STEPS:
            return
        past_mask = self.detect_buffer[0]                # RESPONSE_DELAY_STEPS 스텝 전 감지
        if not past_mask.any():
            return
        dist = distance_transform_edt(~past_mask)        # 그때 발견된 지점 기준
        zone = dist <= RESPONSE_RADIUS
        vax = np.where(zone, VAX_RESPONSE_RATE * self.S, 0.0)
        vax = np.minimum(vax, self.S)                    # 전이량 ≤ S
        self.V += vax
        self.S -= vax

    def step(self):
        S, A, I, R, V = self.S, self.A, self.I, self.R, self.V
        N = S + A + I + R + V
        Nsafe = np.where(N > 0, N, 1.0)

        # ① 반응: 전염원 = A + I, 새 감염을 5% I / 95% A 로 분리
        new_inf = BETA * S * (A + I) / Nsafe
        new_inf = np.minimum(new_inf, S / DT)            # clamp: 감염 ≤ S
        rec_A = np.minimum(GAMMA * A, A / DT)
        rec_I = np.minimum(GAMMA * I, I / DT)
        S = S + DT * (-new_inf)
        A = A + DT * ((1 - SYMPTOMATIC_FRAC) * new_inf - rec_A)
        I = I + DT * (SYMPTOMATIC_FRAC * new_inf - rec_I)
        R = R + DT * (rec_A + rec_I)

        # ② 확산 (A 는 S 와 동일, I 는 느림)
        S = S + DT * D_S * laplacian(S)
        A = A + DT * D_A * laplacian(A)
        I = I + DT * D_I * laplacian(I)
        R = R + DT * D_R * laplacian(R)
        V = V + DT * D_V * laplacian(V)

        for arr in (S, A, I, R, V):
            np.clip(arr, 0.0, None, out=arr)
        self.S, self.A, self.I, self.R, self.V = S, A, I, R, V

        # ③ 반응성 대량 백신(증상 발생 근방)
        self.reactive_vaccination()
        self.t += DT

    def total(self):
        return (self.S + self.A + self.I + self.R + self.V).sum()


def run():
    m = PolioSAIV()
    frames = {"A": [], "I": [], "V": []}
    frame_t = []
    totals = {k: [] for k in ("S", "A", "I", "R", "V")}

    def rec():
        for k, arr in zip(("S", "A", "I", "R", "V"), (m.S, m.A, m.I, m.R, m.V)):
            totals[k].append(arr.sum())

    def snap():
        for k, arr in zip(("A", "I", "V"), (m.A, m.I, m.V)):
            frames[k].append(arr.copy())
        frame_t.append(m.t)

    rec(); snap()
    peaked = False
    for step in range(1, MAX_STEPS + 1):
        m.step(); rec()
        if step % FRAME_STRIDE == 0:
            snap()
        frac = (m.A + m.I).sum() / m.total()
        if frac > 0.002:
            peaked = True
        if peaked and frac < 3e-4:        # 봉쇄되면(감염 <0.03%) 종료
            break
    snap()
    N = totals["S"][0] + totals["A"][0] + totals["I"][0] + totals["R"][0] + totals["V"][0]
    attack = (totals["R"][-1] + totals["A"][-1] + totals["I"][-1]) / N
    print(f"총 {step}스텝(t={m.t:.0f}), 프레임 {len(frame_t)}개")
    print(f"증상 비율 {SYMPTOMATIC_FRAC*100:.0f}% · 반응성 백신(반경 {RESPONSE_RADIUS}) · "
          f"배포 지연 {RESPONSE_DELAY_DAYS:.0f}일")
    print(f"최종 감염경험률 {attack*100:.1f}% · 최종 백신 {totals['V'][-1]/N*100:.1f}%")
    return frames, frame_t, totals


def plot_totals(totals):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(totals["A"])) * DT
    ax.plot(x, totals["A"], color=COMP_COLOR["A"], lw=2, label="A asymptomatic infected")
    ax.plot(x, totals["I"], color=COMP_COLOR["I"], lw=2, label="I symptomatic infected")
    ax.plot(x, totals["V"], color=COMP_COLOR["V"], lw=2, label="V vaccinated")
    tot = sum(np.array(totals[k]) for k in ("S", "A", "I", "R", "V"))
    ax.plot(x, tot, color="0.4", lw=1.3, ls="--", label="Total pop (conserved)")
    ax.set_title(f"Polio SAIRV — symptomatic {SYMPTOMATIC_FRAC*100:.0f}% + mass vaccination near symptom sites")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Population")
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "pde_polio_saiv_totals.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"저장: {out}")


def animate(frames, frame_t, totals):
    comps = [("A", "A asymptomatic infected (spreader)", "inferno"),
             ("I", "I symptomatic infected (surveillance signal)", "Reds"),
             ("V", "V vaccinated (mass rollout near symptoms)", "Purples")]
    nframe = len(frame_t)
    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    axes = {c: fig.add_subplot(gs[0, i]) for i, (c, _, _) in enumerate(comps)}
    axc = fig.add_subplot(gs[0, 3])

    ims = {}
    for key, title, cmap in comps:
        vm = max((f.max() for f in frames[key]), default=1) * 0.9 or 1.0
        ims[key] = axes[key].imshow(frames[key][0], cmap=cmap, origin="lower",
                                    vmin=0, vmax=vm, extent=[0, NX, 0, NY])
        axes[key].set_title(title, fontsize=10)
        axes[key].set_xticks([]); axes[key].set_yticks([])
        fig.colorbar(ims[key], ax=axes[key], fraction=0.046, pad=0.02)
    suptitle = fig.suptitle("", fontsize=13)

    x = np.arange(len(totals["A"])) * DT
    lines = {}
    for k in ("A", "I", "V"):
        (ln,) = axc.plot([], [], color=COMP_COLOR[k], lw=2,
                         label={"A": "A asymptomatic", "I": "I symptomatic", "V": "V vaccinated"}[k])
        lines[k] = ln
    axc.set_xlim(0, x[-1] if len(x) else 1)
    axc.set_ylim(0, max(max(totals["A"]), max(totals["V"]), 1) * 1.05)
    axc.set_xlabel("Time (days)"); axc.set_ylabel("Population")
    axc.set_title("A / I / V trend"); axc.legend(fontsize=9); axc.grid(True, alpha=0.25)

    def update(i):
        for key, _, _ in comps:
            ims[key].set_data(frames[key][i])
        suptitle.set_text(f"Polio SAIRV + symptom-reactive vaccination (rollout delay {RESPONSE_DELAY_DAYS:.0f} days)"
                          f"   t = {frame_t[i]:.0f} days")
        n = min(i * FRAME_STRIDE + 1, len(x))
        for k in ("A", "I", "V"):
            lines[k].set_data(x[:n], totals[k][:n])
        return [*ims.values(), suptitle, *lines.values()]

    anim = FuncAnimation(fig, update, frames=nframe, blit=False, interval=120)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULT_DIR, "pde_polio_saiv.gif")
    anim.save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    frames, frame_t, totals = run()
    plot_totals(totals)
    animate(frames, frame_t, totals)
    print("완료.")
