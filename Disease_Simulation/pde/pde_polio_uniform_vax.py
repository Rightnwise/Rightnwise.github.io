"""
pde_polio_uniform_vax.py
================================================================
소아마비 SARV 모델 + '전역 균일 백신'.

목적: 백신을 (링이 아니라) 전체 인구의 일정 비율에 균일하게 접종했을 때,
      집단면역선(herd immunity threshold) 기준으로 A 가 퍼지는지/막히는지 보인다.

집단면역선  H = 1 - 1/R0.   여기서 R0 = β/γ = 5  →  H = 80%.
  · 접종률 < 80% : 실효 재생산수 R_eff = R0·(1-접종률) > 1  → A 가 퍼진다.
  · 접종률 ≥ 80% : R_eff ≤ 1                              → A 가 못 퍼지고 소멸.

기본 시나리오: 전체의 75% 균일 접종(< 80%) → 아직 퍼진다.

모델(SARV, 사망 없음)
  S→A : β·S·A/N   (전염원은 A 뿐)
  A→R : γ·A
  S→V : 시작 시 전체 S 의 VAX_RATE 를 균일하게 V 로(재발항 없음)
  확산: S·A·R·V 라플라시안(무유출 경계). 전이량 clamp, 인구 보존.
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
GAMMA = 0.12                      # R0 = β/γ = 5 → 집단면역선 H = 1-1/R0 = 0.8
HERD_THRESHOLD = 1.0 - GAMMA / BETA

# ---------------- 확산계수 (A 는 무증상 → S 와 동일) ----------------
D_S = 0.3
D_A = 0.3
D_R = 0.3
D_V = 0.3

# ---------------- 백신 ----------------
VAX_RATE = 0.75                   # 전역 균일 접종 비율 (80% 임계 '미만' → 퍼짐)

# ---------------- 발생점 ----------------
SEEDS = [(NX // 2, NY // 2)]          # 단일 발생점(중심)
SEED_A = 50.0
SEED_RADIUS = 2

MAX_STEPS = 6000
FRAME_STRIDE = 30
COMP_COLOR = {"S": "#2f6fd0", "A": "#d1352c", "R": "#2e8b57", "V": "#8e44ad"}


def laplacian(f):
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f) / (DX * DX)


class PolioSARV:
    def __init__(self, vax_rate=VAX_RATE):
        self.S = np.full((NY, NX), N0)
        self.A = np.zeros((NY, NX))
        self.R = np.zeros((NY, NX))
        self.V = np.zeros((NY, NX))
        self.t = 0.0
        self.apply_uniform_vaccination(vax_rate)   # 전역 균일 백신 먼저
        self.seed_points()                          # 그 다음 감염 씨앗

    def apply_uniform_vaccination(self, rate):
        """전체 S 의 rate 만큼을 균일하게 V 로(S→V). A 는 안 건드림."""
        vax = np.minimum(rate * self.S, self.S)
        self.V += vax
        self.S -= vax

    def seed_points(self):
        yy, xx = np.ogrid[:NY, :NX]
        for cx, cy in SEEDS:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SEED_RADIUS ** 2
            add = np.minimum(np.where(mask, SEED_A, 0.0), self.S)
            self.A += add
            self.S -= add

    def step(self):
        S, A, R, V = self.S, self.A, self.R, self.V
        N = S + A + R + V
        Nsafe = np.where(N > 0, N, 1.0)

        infection = np.minimum(BETA * S * A / Nsafe, S / DT)   # S→A (A만), clamp
        recovery = np.minimum(GAMMA * A, A / DT)               # A→R, clamp
        S = S + DT * (-infection)
        A = A + DT * (infection - recovery)
        R = R + DT * recovery

        S = S + DT * D_S * laplacian(S)
        A = A + DT * D_A * laplacian(A)
        R = R + DT * D_R * laplacian(R)
        V = V + DT * D_V * laplacian(V)

        for arr in (S, A, R, V):
            np.clip(arr, 0.0, None, out=arr)
        self.S, self.A, self.R, self.V = S, A, R, V
        self.t += DT


def simulate(vax_rate, steps=MAX_STEPS, capture=False):
    m = PolioSARV(vax_rate)
    totals = {k: [] for k in "SARV"}
    frames = {"S": [], "A": [], "R": [], "V": []} if capture else None
    frame_t = []

    def rec():
        for k, arr in zip("SARV", (m.S, m.A, m.R, m.V)):
            totals[k].append(arr.sum())

    def snap():
        for k, arr in zip("SARV", (m.S, m.A, m.R, m.V)):
            frames[k].append(arr.copy())
        frame_t.append(m.t)

    rec()
    if capture:
        snap()
    peaked = False
    for step in range(1, steps + 1):
        m.step(); rec()
        if capture and step % FRAME_STRIDE == 0:
            snap()
        fa = m.A.sum() / (m.S + m.A + m.R + m.V).sum()
        if fa > 0.002:
            peaked = True
        if peaked and fa < 1e-5:
            break
    if capture:
        snap()
    return m, totals, frames, frame_t


def plot_totals(totals, vax_rate):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(totals["S"])) * DT
    for k, lab in [("S", "S susceptible"), ("A", "A asymptomatic infected"),
                   ("R", "R recovered (had infection)"), ("V", "V vaccinated")]:
        ax.plot(x, totals[k], color=COMP_COLOR[k], lw=2, label=lab)
    tot = sum(np.array(totals[k]) for k in "SARV")
    ax.plot(x, tot, color="0.4", lw=1.5, ls="--", label="Total pop S+A+R+V (conserved)")
    N = tot[0]
    attack = totals["R"][-1] / N
    reff = (BETA / GAMMA) * (1 - vax_rate)
    ax.set_title(f"Uniform vaccination {vax_rate*100:.0f}% (threshold {HERD_THRESHOLD*100:.0f}%) "
                 f"— R_eff={reff:.2f}, final attack rate {attack*100:.1f}%")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Population")
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "pde_polio_uniform_vax_totals.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"저장: {out}")


def plot_threshold_comparison(rates=(0.65, 0.70, 0.75, 0.80, 0.85)):
    """여러 접종률의 A(t) 를 겹쳐 80% 임계 경계를 보인다."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for r in rates:
        _, totals, _, _ = simulate(r, steps=3000, capture=False)
        x = np.arange(len(totals["A"])) * DT
        reff = (BETA / GAMMA) * (1 - r)
        spread = "spread" if reff > 1 else "die-out"
        ls = "-" if reff > 1 else "--"
        ax.plot(x, totals["A"], lw=2, ls=ls,
                label=f"{r*100:.0f}%  R_eff={reff:.2f} ({spread})")
    ax.set_title(f"Total infected A(t) by coverage — herd immunity threshold {HERD_THRESHOLD*100:.0f}%")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Total infected A")
    ax.legend(fontsize=9, title="Coverage"); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "pde_polio_uniform_threshold.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"저장: {out}")


def animate(frames, frame_t, totals, vax_rate):
    comps = [("S", "S susceptible", "Blues"), ("A", "A asymptomatic infected", "inferno"),
             ("R", "R recovered", "Greens"), ("V", "V vaccinated (uniform)", "Purples")]
    nframe = len(frame_t)
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
    axes = {"S": fig.add_subplot(gs[0, 0]), "A": fig.add_subplot(gs[0, 1]),
            "R": fig.add_subplot(gs[1, 0]), "V": fig.add_subplot(gs[1, 1])}
    axc = fig.add_subplot(gs[:, 2])
    ims = {}
    for key, title, cmap in comps:
        vm = max((f.max() for f in frames[key]), default=1) * 0.9 or 1.0
        ims[key] = axes[key].imshow(frames[key][0], cmap=cmap, origin="lower",
                                    vmin=0, vmax=vm, extent=[0, NX, 0, NY])
        axes[key].set_title(title, fontsize=11)
        axes[key].set_xticks([]); axes[key].set_yticks([])
        fig.colorbar(ims[key], ax=axes[key], fraction=0.046, pad=0.02)
    suptitle = fig.suptitle("", fontsize=13)
    x = np.arange(len(totals["S"])) * DT
    lines = {k: axc.plot([], [], color=COMP_COLOR[k], lw=2, label=k)[0] for k in "SARV"}
    axc.set_xlim(0, x[-1] if len(x) else 1); axc.set_ylim(0, max(totals["S"]) * 1.05)
    axc.set_xlabel("Time (days)"); axc.set_ylabel("Population")
    axc.set_title("Total SARV trend"); axc.legend(fontsize=9); axc.grid(True, alpha=0.25)

    def update(i):
        for key, _, _ in comps:
            ims[key].set_data(frames[key][i])
        reff = (BETA / GAMMA) * (1 - vax_rate)
        suptitle.set_text(f"Uniform vaccination {vax_rate*100:.0f}% (< threshold {HERD_THRESHOLD*100:.0f}%) "
                          f"→ R_eff={reff:.2f} → A spreads   t = {frame_t[i]:.0f} days")
        n = min(i * FRAME_STRIDE + 1, len(x))
        for k in "SARV":
            lines[k].set_data(x[:n], totals[k][:n])
        return [*ims.values(), suptitle, *lines.values()]

    anim = FuncAnimation(fig, update, frames=nframe, blit=False, interval=120)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULT_DIR, "pde_polio_uniform_vax.gif")
    anim.save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    print(f"집단면역선 H = 1 - 1/R0 = {HERD_THRESHOLD*100:.0f}%  (R0 = {BETA/GAMMA:.0f})")
    m, totals, frames, frame_t = simulate(VAX_RATE, capture=True)
    N = sum(totals[k][0] for k in "SARV")
    reff = (BETA / GAMMA) * (1 - VAX_RATE)
    print(f"전역 균일 접종 {VAX_RATE*100:.0f}% → R_eff = {reff:.2f} "
          f"({'>1 이므로 퍼짐' if reff > 1 else '≤1 이므로 소멸'})")
    print(f"최종 감염률: {totals['R'][-1]/N*100:.1f}%  (백신 없으면 ~99%)")
    plot_totals(totals, VAX_RATE)
    plot_threshold_comparison()
    animate(frames, frame_t, totals, VAX_RATE)
    print("완료.")
