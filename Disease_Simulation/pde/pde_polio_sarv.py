"""
pde_polio_sarv.py
================================================================
소아마비(polio)를 2D 반응-확산 SARV 모델로 '2웨이브' 시뮬레이션 + 링 백신 비교.

구획  S → A → R,  그리고 V(백신)
  S : 감염가능
  A : 무증상 감염(전염 O). 무증상이라 평소처럼 이동 → D_A = D_S
  R : 회복/면역
  V : 백신 접종(면역)

시나리오 (비교)
  · 1차 웨이브: 백신 없음 → A 가 격자 전역으로 퍼지는 대유행
  · (유행 종료 후 인구 초기화 = 새 코호트/재유입)
  · 2차 웨이브: "한 번 겪은 뒤 정부가 백신 도입" → 발생 4점 둘레에 링 백신.
      단, 접종률을 집단면역선(1-1/R0=80%) '아래'로 두어 완전봉쇄는 아니고,
      A 는 여전히 퍼지되 감염자 수가 줄어드는 '적당한 억제'.

수치기법: 매 스텝 ① 반응 → ② 확산(라플라시안, 무유출 경계). V·(사망없음).
시각화: S/A/R/V 지도 + 전체 SARV 시계열(웨이브 경계·최종 감염률 비교).
"""

import os
import sys
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
BETA = 0.6             # 감염 전파율(전염원 A)
GAMMA = 0.12           # 회복율 (R0 = β/γ = 5)

# ---------------- 확산계수 (A 는 무증상 → S 와 동일) ----------------
D_S = 0.3
D_A = 0.3              # = D_S
D_R = 0.3
D_V = 0.0             # 백신 링은 고정

# ---------------- 발생점 & 링 백신(2차 웨이브에만) ----------------
SEEDS = [(NX // 2, NY // 2)]          # 단일 발생점(중심)
SEED_A = 50.0
SEED_RADIUS = 2
# 링을 얇게 하면 전역 접종률이 너무 낮아 억제가 거의 안 됨.
# 눈에 띄는 '부분 억제'를 위해 링 면적을 키우되, 접종률은 집단면역선(80%) '아래'로
# 두어 완전봉쇄는 아니고 A 가 고리 사이 틈으로 퍼지되 총 감염이 줄게 함.
RING_IN = 4            # 발생점 둘레 링(칸)
RING_OUT = 26
VAX_COVERAGE = 0.78    # 80% 아래 → 완전봉쇄 X (전역 접종은 ~23%)

MAX_WAVES = 2
PEAK_FRAC = 0.002      # 유행 발생 인정
LOW_FRAC = 1e-4        # 유행 종료 판정
MAX_STEPS = 16000
FRAME_STRIDE = 30


def laplacian(f):
    fp = np.pad(f, 1, mode="edge")
    return (fp[:-2, 1:-1] + fp[2:, 1:-1] + fp[1:-1, :-2] + fp[1:-1, 2:]
            - 4.0 * f) / (DX * DX)


class PolioSARV:
    def __init__(self):
        self.reset_population()
        self.t = 0.0
        self.wave = 1
        self.active = False
        self.finished = False
        self.wave_attack = []          # 웨이브별 최종 감염률
        self.seed_points()             # 1차: 백신 없음

    def reset_population(self):
        self.S = np.full((NY, NX), N0)
        self.A = np.zeros((NY, NX))
        self.R = np.zeros((NY, NX))
        self.V = np.zeros((NY, NX))

    def apply_ring_vaccination(self):
        """지정된 발생점(SEEDS) 둘레 링(RING_IN~RING_OUT) 안의 S 일부를 V 로.
        ★ A 를 직접 건드리지 않는다(삭제·0화 금지). 오직 S↓, V↑ 만 한다(rule 6).
        → 며칠 뒤 그 지역에서 '새로 생길' A 가 줄어드는 방식."""
        yy, xx = np.ogrid[:NY, :NX]
        ring = np.zeros((NY, NX), dtype=bool)
        for cx, cy in SEEDS:                         # 지정된 지점 둘레에 고정 링
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            ring |= (d2 >= RING_IN ** 2) & (d2 <= RING_OUT ** 2)
        vax = np.where(ring, VAX_COVERAGE * self.S, 0.0)
        vax = np.minimum(vax, self.S)                # 전이량 ≤ S (rule 8, 음수 방지)
        self.V += vax
        self.S -= vax

    def seed_points(self):
        yy, xx = np.ogrid[:NY, :NX]
        for cx, cy in SEEDS:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= SEED_RADIUS ** 2
            add = np.minimum(np.where(mask, SEED_A, 0.0), self.S)
            self.A += add
            self.S -= add

    def status(self):
        return f"Wave {self.wave} · {'no vaccine' if self.wave == 1 else 'ring vaccination'}"

    def step(self):
        S, A, R, V = self.S, self.A, self.R, self.V
        N = S + A + R + V
        Nsafe = np.where(N > 0, N, 1.0)

        # ① 반응(SARV) — 감염은 A 만, 전이량은 원 구획을 넘지 않게 clamp(rule 8)
        infection = BETA * S * A / Nsafe          # S→A (전염원 A 뿐)
        recovery = GAMMA * A                       # A→R
        infection = np.minimum(infection, S / DT)  # 감염량 ≤ S
        recovery = np.minimum(recovery, A / DT)    # 회복량 ≤ A
        S = S + DT * (-infection)                  # S 는 감염(+백신)으로만 감소
        A = A + DT * (infection - recovery)        # A 는 감염으로만 증가, 회복으로만 감소
        R = R + DT * recovery                      # R 은 회복으로만 증가
        # V 는 반응에서 안 변함(백신으로만 증가, 재발항 없음)

        # ② 확산 (A 는 S 와 동일 속도)
        S = S + DT * D_S * laplacian(S)
        A = A + DT * D_A * laplacian(A)
        R = R + DT * D_R * laplacian(R)
        V = V + DT * D_V * laplacian(V)

        for arr in (S, A, R, V):
            np.clip(arr, 0.0, None, out=arr)
        self.S, self.A, self.R, self.V = S, A, R, V
        self.t += DT

        # ---- 웨이브 전환 ----
        totN = (S + A + R + V).sum()
        fracA = A.sum() / totN
        if not self.active and fracA > PEAK_FRAC:
            self.active = True
        if self.active and fracA < LOW_FRAC:
            self.wave_attack.append(self.R.sum() / totN)     # 이번 웨이브 감염률 기록
            if self.wave < MAX_WAVES:
                self.reset_population()
                self.wave += 1
                self.apply_ring_vaccination()                # 2차: 정부 링 백신 도입
                self.seed_points()
                self.active = False
            else:
                self.finished = True


def run():
    m = PolioSARV()
    frames = {"S": [], "A": [], "R": [], "V": []}
    frame_t, frame_status = [], []
    totals = {k: [] for k in "SARV"}

    def record_totals():
        for k, arr in zip("SARV", (m.S, m.A, m.R, m.V)):
            totals[k].append(arr.sum())

    def snap():
        for k, arr in zip("SARV", (m.S, m.A, m.R, m.V)):
            frames[k].append(arr.copy())
        frame_t.append(m.t); frame_status.append(m.status())

    record_totals(); snap()
    step = 0
    while step < MAX_STEPS and not m.finished:
        m.step(); step += 1
        record_totals()
        if step % FRAME_STRIDE == 0:
            snap()
    snap()
    print(f"총 {step}스텝(t={m.t:.0f}), 프레임 {len(frame_t)}개")
    for i, a in enumerate(m.wave_attack, 1):
        tag = "백신 없음" if i == 1 else f"링 백신({VAX_COVERAGE*100:.0f}%)"
        print(f"  {i}차 웨이브({tag}) 최종 감염률: {a*100:.1f}%")
    return frames, frame_t, frame_status, totals, m.wave_attack


def plot_totals(totals, wave_attack):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(totals["S"])) * DT
    for k, lab in [("S", "S susceptible"), ("A", "A asymptomatic infected"),
                   ("R", "R recovered (had infection)"), ("V", "V vaccinated")]:
        ax.plot(x, totals[k], color=COMP_COLOR[k], lw=2, label=lab)
    # 총인구 보존 확인선 (rule 7, 10): S+A+R+V 가 일정해야 함
    total_pop = np.array(totals["S"]) + np.array(totals["A"]) \
        + np.array(totals["R"]) + np.array(totals["V"])
    ax.plot(x, total_pop, color="0.4", lw=1.5, ls="--", label="Total pop S+A+R+V (conserved)")
    txt = "  vs  ".join(
        f"Wave {i} {'(no vax)' if i == 1 else '(ring vax)'} {a*100:.0f}%"
        for i, a in enumerate(wave_attack, 1))
    ax.set_title(f"Polio SARV 2-wave comparison — final attack rate  {txt}")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Population")
    ax.legend(fontsize=9, ncol=2); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "pde_polio_sarv_totals.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"저장: {out}")


COMP_COLOR = {"S": "#2f6fd0", "A": "#d1352c", "R": "#2e8b57", "V": "#8e44ad"}


def animate(frames, frame_t, frame_status, totals):
    comps = [("S", "S susceptible", "Blues"), ("A", "A asymptomatic infected", "inferno"),
             ("R", "R recovered", "Greens"), ("V", "V vaccinated (ring)", "Purples")]
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
    lines = {}
    for k in "SARV":
        (ln,) = axc.plot([], [], color=COMP_COLOR[k], lw=2,
                         label={"S": "S", "A": "A asymptomatic", "R": "R", "V": "V vaccinated"}[k])
        lines[k] = ln
    axc.set_xlim(0, x[-1] if len(x) else 1)
    axc.set_ylim(0, max(totals["S"]) * 1.05)
    axc.set_xlabel("Time (days)"); axc.set_ylabel("Population")
    axc.set_title("Total SARV trend (wave 1 no vax / wave 2 ring vax)")
    axc.legend(fontsize=9); axc.grid(True, alpha=0.25)

    def update(i):
        for key, _, _ in comps:
            ims[key].set_data(frames[key][i])
        suptitle.set_text(f"Polio SARV   t = {frame_t[i]:.0f} days   {frame_status[i]}")
        n = min(i * FRAME_STRIDE + 1, len(x))
        for k in "SARV":
            lines[k].set_data(x[:n], totals[k][:n])
        return [*ims.values(), suptitle, *lines.values()]

    anim = FuncAnimation(fig, update, frames=nframe, blit=False, interval=120)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULT_DIR, "pde_polio_sarv.gif")
    anim.save(out, writer=PillowWriter(fps=12)); plt.close(fig)
    print(f"저장: {out}")


# ==================================================================
# 규칙 검증 테스트 (rule 9)
# ==================================================================
def _simulate(steps=1200, apply_vax=False, uniform_vax=0.0, **ov):
    """파라미터를 임시로 바꿔 단일 웨이브를 돌리고 지표 반환.
    uniform_vax: 시작 시 전체 S 중 이 비율을 균일하게 V 로(전역 백신 원리 테스트)."""
    ov.setdefault("MAX_WAVES", 1)                 # 재발 없이 1웨이브
    saved = {k: globals()[k] for k in ov}
    globals().update(ov)
    try:
        m = PolioSARV()
        if uniform_vax > 0:                        # 전역 균일 백신 S→V
            v = uniform_vax * m.S
            m.V += v
            m.S -= v
        if apply_vax:
            m.apply_ring_vaccination()
        tot0 = (m.S + m.A + m.R + m.V).sum()
        a0 = m.A.sum()
        peakA = a0
        maxdev = 0.0
        negs = False
        for _ in range(steps):
            m.step()
            aS = m.A.sum()
            peakA = max(peakA, aS)
            maxdev = max(maxdev, abs((m.S + m.A + m.R + m.V).sum() - tot0) / tot0)
            negs = negs or any((arr < 0).any() for arr in (m.S, m.A, m.R, m.V))
            if m.finished:
                break
        Ntot = (m.S + m.A + m.R + m.V).sum()
        return dict(a0=a0, peakA=peakA, Afinal=m.A.sum(),
                    attack=m.R.sum() / Ntot, maxdev=maxdev, negs=negs,
                    A_after1=None)
    finally:
        globals().update(saved)


def run_tests():
    print("=" * 60)
    print("SARV 규칙 검증 테스트")
    print("=" * 60)
    tag = lambda b: "PASS ✅" if b else "FAIL ❌"

    # 1) beta=0 → 새 감염 없음(A 는 회복으로 감소만)
    r = _simulate(steps=300, BETA=0.0)
    t1 = r["peakA"] <= r["a0"] * 1.0001 and r["Afinal"] < r["a0"]
    print(f"1) beta=0 → A 성장 없음      : 초기 {r['a0']:.0f} · 정점 {r['peakA']:.0f} · 최종 {r['Afinal']:.0f}  [{tag(t1)}]")

    # 2) initial_A=0 → 감염 시작 안 함
    r = _simulate(steps=200, SEED_A=0.0)
    t2 = r["peakA"] < 1e-9 and r["attack"] < 1e-9
    print(f"2) initial_A=0 → 감염 없음   : 정점A {r['peakA']:.2e} · 감염률 {r['attack']*100:.2f}%  [{tag(t2)}]")

    # 3) 백신률↑ → A 정점↓ (일반 백신 원리: 균일 접종 비율을 올리며 확인)
    p0 = _simulate(steps=1200, uniform_vax=0.0)["peakA"]
    p1 = _simulate(steps=1200, uniform_vax=0.3)["peakA"]
    p2 = _simulate(steps=1200, uniform_vax=0.6)["peakA"]
    t3 = p0 >= p1 >= p2
    print(f"3) 백신률↑ → A정점↓          : 0% {p0:.0f} ≥ 30% {p1:.0f} ≥ 60% {p2:.0f}  [{tag(t3)}]")

    # 4) 링 반경↑ → 확산 약화(최종 감염률↓)
    a_small = _simulate(steps=1500, apply_vax=True, RING_IN=4, RING_OUT=12, VAX_COVERAGE=0.7)["attack"]
    a_big = _simulate(steps=1500, apply_vax=True, RING_IN=4, RING_OUT=26, VAX_COVERAGE=0.7)["attack"]
    t4 = a_big <= a_small
    print(f"4) 링반경↑ → 감염률↓         : 작은링 {a_small*100:.0f}% ≥ 큰링 {a_big*100:.0f}%  [{tag(t4)}]")

    # 5) V/R↑(접종↑) → 새 감염↓ (최종 감염률↓)
    v_none = _simulate(steps=1500, apply_vax=False)["attack"]
    v_more = _simulate(steps=1500, apply_vax=True, RING_IN=4, RING_OUT=26, VAX_COVERAGE=0.75)["attack"]
    t5 = v_more < v_none
    print(f"5) 백신↑ → 감염률↓           : 백신X {v_none*100:.0f}% > 백신O {v_more*100:.0f}%  [{tag(t5)}]")

    # 6) A→R 점진적(즉시 소멸 아님): beta=0 에서 1스텝 뒤 A ≈ a0·(1-γ·dt)
    saved = {k: globals()[k] for k in ("BETA",)}
    globals()["BETA"] = 0.0
    m = PolioSARV(); a0 = m.A.sum(); m.step(); a1 = m.A.sum()
    globals().update(saved)
    expected = a0 * (1 - GAMMA * DT)
    t6 = a1 > a0 * 0.5 and abs(a1 - expected) / a0 < 0.02
    print(f"6) A→R 점진적(즉시X)         : 1스텝 A {a0:.0f}→{a1:.0f} (기대 {expected:.0f})  [{tag(t6)}]")

    # 7) 보존 & 8) 음수
    r = _simulate(steps=800, apply_vax=True, RING_IN=4, RING_OUT=20, VAX_COVERAGE=0.7)
    t7 = r["maxdev"] < 1e-9
    t8 = not r["negs"]
    print(f"7) 총인구 보존(S+A+R+V일정)  : 최대편차 {r['maxdev']:.1e}  [{tag(t7)}]")
    print(f"8) 음수 없음                 : 음수발생 {r['negs']}  [{tag(t8)}]")

    allok = all([t1, t2, t3, t4, t5, t6, t7, t8])
    print("=" * 60)
    print(f"전체 결과: {'모두 PASS ✅' if allok else '일부 FAIL ❌'}")
    print("=" * 60)
    return allok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        frames, frame_t, frame_status, totals, wave_attack = run()
        plot_totals(totals, wave_attack)
        animate(frames, frame_t, frame_status, totals)
        print("완료.")
