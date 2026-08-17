"""
pde_polio_vax_strategy.py
================================================================
폴리오 SAIRV 모델에서 백신 '접종 전략' 비교:

  · 상시 접종(constant): 매일 전 인구에게 균일하게 조금씩 (S→V, 전역·상시)
  · 펄스 접종(pulse)   : 증상(I) 발견 후 5일 지연으로 그 근방에 집중 대량 (S→V, 반응성)
  · 혼합(50:50)        : 예산을 반으로 나눠 절반은 상시, 절반은 펄스(5일 지연)

무증상(A 95%)이 조용히 퍼지고 증상(I 5%)만 드러나는 폴리오에서,
'미리 상시로 까느냐' vs '증상 뜨면 몰아서 까느냐'의 효과를 본다.

모델(SAIRV): S→(95%A / 5%I)→R, S→V. 인구 보존·전이 clamp. (앞 파일과 동일)
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

# ---------------- 격자/시간 ----------------
NX = NY = 160
DX = 1.0
DT = 0.15
N0 = 1000.0

# ---------------- 역학 ----------------
BETA = 0.6
GAMMA = 0.12
SYMPTOMATIC_FRAC = 0.05
D_S = D_A = D_R = 0.3
D_I = 0.05
D_V = 0.0

# ---------------- 접종 전략 파라미터 ----------------
# 백신 예산(낮출수록 전략별 차이가 극적으로 벌어짐)
CONSTANT_FULL = 0.01       # 상시 접종 '풀' 강도: 하루 전체 S 의 1% 를 균일 접종
PULSE_FULL = 0.04          # 펄스 접종 '풀' 강도: 반응 구역에서 매 스텝 S 의 4%
I_DETECT = 1.0
RESPONSE_RADIUS = 7
RESPONSE_DELAY_DAYS = 2.0   # 펄스는 증상 발견 후 이만큼 지연 뒤 배포
RESPONSE_DELAY_STEPS = int(round(RESPONSE_DELAY_DAYS / DT))

SEEDS = [(NX // 2, NY // 2)]          # 단일 발생점(중심)
SEED_A = 50.0
SEED_RADIUS = 2
MAX_STEPS = 2000
FRAME_STRIDE = 12
COMP_COLOR = {"A": "#e58a1a", "I": "#d1352c", "V": "#8e44ad"}


def laplacian(f):
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f) / (DX * DX)


class PolioVaxStrategy:
    def __init__(self, constant_rate=0.0, pulse_rate=0.0):
        self.S = np.full((NY, NX), N0)
        self.A = np.zeros((NY, NX))
        self.I = np.zeros((NY, NX))
        self.R = np.zeros((NY, NX))
        self.V = np.zeros((NY, NX))
        self.t = 0.0
        self.constant_rate = constant_rate    # 하루 균일 접종 비율
        self.pulse_rate = pulse_rate          # 반응 구역 스텝당 접종 비율
        self.detect_buffer = deque(maxlen=RESPONSE_DELAY_STEPS + 1)
        self.seed_points()

    def seed_points(self):
        yy, xx = np.ogrid[:NY, :NX]
        for cx, cy in SEEDS:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SEED_RADIUS ** 2
            add = np.minimum(np.where(mask, SEED_A, 0.0), self.S)
            self.A += add
            self.S -= add

    def constant_vaccination(self):
        """상시 접종: 매 스텝(=하루 rate) 전 격자 S 의 일부를 균일하게 V 로."""
        if self.constant_rate <= 0:
            return
        vax = np.minimum(self.constant_rate * DT * self.S, self.S)
        self.V += vax
        self.S -= vax

    def pulse_vaccination(self):
        """펄스(반응성) 접종: 증상 발견 후 RESPONSE_DELAY 뒤, 그 근방 S 를 대량 V 로."""
        self.detect_buffer.append(self.I > I_DETECT)
        if self.pulse_rate <= 0 or len(self.detect_buffer) <= RESPONSE_DELAY_STEPS:
            return
        past = self.detect_buffer[0]                     # 지연 스텝 전 감지
        if not past.any():
            return
        zone = distance_transform_edt(~past) <= RESPONSE_RADIUS
        vax = np.minimum(np.where(zone, self.pulse_rate * self.S, 0.0), self.S)
        self.V += vax
        self.S -= vax

    def step(self):
        S, A, I, R, V = self.S, self.A, self.I, self.R, self.V
        N = S + A + I + R + V
        Nsafe = np.where(N > 0, N, 1.0)
        new_inf = np.minimum(BETA * S * (A + I) / Nsafe, S / DT)
        rec_A = np.minimum(GAMMA * A, A / DT)
        rec_I = np.minimum(GAMMA * I, I / DT)
        S = S + DT * (-new_inf)
        A = A + DT * ((1 - SYMPTOMATIC_FRAC) * new_inf - rec_A)
        I = I + DT * (SYMPTOMATIC_FRAC * new_inf - rec_I)
        R = R + DT * (rec_A + rec_I)
        S = S + DT * D_S * laplacian(S)
        A = A + DT * D_A * laplacian(A)
        I = I + DT * D_I * laplacian(I)
        R = R + DT * D_R * laplacian(R)
        V = V + DT * D_V * laplacian(V)
        for arr in (S, A, I, R, V):
            np.clip(arr, 0.0, None, out=arr)
        self.S, self.A, self.I, self.R, self.V = S, A, I, R, V
        self.constant_vaccination()      # 상시
        self.pulse_vaccination()         # 펄스(지연)
        self.t += DT

    def total(self):
        return (self.S + self.A + self.I + self.R + self.V).sum()


def simulate(constant_rate, pulse_rate, capture=False):
    m = PolioVaxStrategy(constant_rate, pulse_rate)
    ts = {"t": [], "A": [], "I": [], "V": [], "active": []}
    frames = {"A": [], "I": [], "V": []} if capture else None
    ft = []

    def rec():
        ts["t"].append(m.t); ts["A"].append(m.A.sum()); ts["I"].append(m.I.sum())
        ts["V"].append(m.V.sum()); ts["active"].append((m.A + m.I).sum())

    def snap():
        for k, arr in zip(("A", "I", "V"), (m.A, m.I, m.V)):
            frames[k].append(arr.copy())
        ft.append(m.t)

    rec()
    if capture:
        snap()
    peaked = False
    for step in range(1, MAX_STEPS + 1):
        m.step(); rec()
        if capture and step % FRAME_STRIDE == 0:
            snap()
        frac = (m.A + m.I).sum() / m.total()
        if frac > 0.002:
            peaked = True
        if peaked and frac < 3e-4:
            break
    if capture:
        snap()
    N = m.total()
    attack = (m.R.sum() + m.A.sum() + m.I.sum()) / N * 100
    vax = m.V.sum() / N * 100
    peak_active = max(ts["active"]) / N * 100
    return dict(ts=ts, frames=frames, ft=ft, attack=attack, vax=vax,
                peak=peak_active, t_end=m.t)


def compare():
    dlab = f"{RESPONSE_DELAY_DAYS:.0f}-day delay"
    clab = f"Constant only ({CONSTANT_FULL*100:.1f}%/day)"
    plab = f"Pulse only ({dlab})"
    mlab = f"Mixed 50:50 (constant {CONSTANT_FULL/2*100:.1f}%+pulse)"
    scen = {
        clab: (CONSTANT_FULL, 0.0),
        plab: (0.0, PULSE_FULL),
        mlab: (CONSTANT_FULL / 2, PULSE_FULL / 2),
    }
    colors = {clab: "#2e8b57", plab: "#d1352c", mlab: "#2f6fd0"}
    results = {}
    for name, (c, p) in scen.items():
        results[name] = simulate(c, p)
        r = results[name]
        print(f"[{name:26s}] 최종감염 {r['attack']:5.1f}% · 정점 {r['peak']:4.1f}% · "
              f"백신 {r['vax']:4.0f}% · 유행 {r['t_end']:.0f}일")

    # (1) 활성 감염 A+I 시계열 비교
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    for name, r in results.items():
        ax1.plot(r["ts"]["t"], np.array(r["ts"]["active"]) / 1e6,
                 color=colors[name], lw=2.2, label=name)
    ax1.set_title("Active infected A+I(t) by strategy")
    ax1.set_xlabel("Time (days)"); ax1.set_ylabel("Active infected (millions)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.25)

    # (2) 최종 감염률 막대
    names = list(results)
    attacks = [results[n]["attack"] for n in names]
    ax2.bar(range(len(names)), attacks, color=[colors[n] for n in names])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(["Constant", "Pulse", "Mixed"], fontsize=10)
    ax2.set_ylabel("Final attack rate (%)")
    ax2.set_title("Final attack rate by strategy")
    for i, a in enumerate(attacks):
        ax2.text(i, a, f"{a:.1f}%", ha="center", va="bottom", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Polio vaccine strategy comparison: constant vs pulse ({dlab}) vs mixed", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(RESULT_DIR, "pde_polio_strategy_compare.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"저장: {out}")
    return results


def animate_mix():
    """혼합 전략(상시1.5% + 펄스5일지연)의 A/I/V 확산 애니메이션."""
    r = simulate(CONSTANT_FULL / 2, PULSE_FULL / 2, capture=True)
    frames, ft, ts = r["frames"], r["ft"], r["ts"]
    comps = [("A", "A asymptomatic (spreader)", "inferno"), ("I", "I symptomatic (signal)", "Reds"),
             ("V", "V vaccinated (constant+pulse)", "Purples")]
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
    sup = fig.suptitle("", fontsize=13)
    x = ts["t"]
    lines = {k: axc.plot([], [], color=COMP_COLOR[k], lw=2,
                         label={"A": "A", "I": "I", "V": "V"}[k])[0] for k in "AIV"}
    axc.set_xlim(0, x[-1] if x else 1)
    axc.set_ylim(0, max(max(ts["A"]), max(ts["V"]), 1) * 1.05)
    axc.set_xlabel("Time (days)"); axc.set_title("A / I / V trend")
    axc.legend(fontsize=9); axc.grid(True, alpha=0.25)

    def update(i):
        for key, _, _ in comps:
            ims[key].set_data(frames[key][i])
        sup.set_text(f"Mixed strategy (constant {CONSTANT_FULL/2*100:.1f}%/day + pulse "
                     f"{RESPONSE_DELAY_DAYS:.0f}-day delay)   t = {ft[i]:.0f} days")
        n = min(i * FRAME_STRIDE + 1, len(x))
        for k in "AIV":
            lines[k].set_data(x[:n], ts[k][:n])
        return [*ims.values(), sup, *lines.values()]

    anim = FuncAnimation(fig, update, frames=len(ft), blit=False, interval=120)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(RESULT_DIR, "pde_polio_strategy_mix.gif")
    anim.save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    compare()
    animate_mix()
    print("완료.")
