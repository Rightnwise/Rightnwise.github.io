"""
================================================================
이중슬릿 / 회절격자 종합 정리 (영어 발표용 - 글 최소화)
================================================================

밝기 = | sin(N·φ/2) / (N·sin(φ/2)) |^2  ×  | sinc(β/2) |^2
        └── 간섭 인자 : N, d 담당 ──┘        └ 회절 인자 : w 담당 ┘
    φ = 2π d sinθ / λ ,   β = 2π w sinθ / λ

세 요소는 독립:
    N → 봉우리 날카로움(부극대 N-2개)  |  w → 회절 포락선 너비  |  d → 봉우리 간격

생성 파일 (글자 없는 깨끗한 그림, 텍스트는 발표에서 직접 추가):
    00_setup.png       - 실험 상황 2D 개념도
    01_sweep_N.png     - 광원 개수만 변화
    02_sweep_w.png     - 슬릿 폭만 변화
    03_sweep_d.png     - 광원 간격만 변화
    04_decompose.png   - 밝기 = 간섭 × 회절 분해
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 공통 파라미터 ----------------
LAM = 0.5                        # 파장 λ
BASE = dict(N=4, d=8.0, w=2.0)   # 기준값 (한 요소만 바꿀 때 나머지는 이 값 고정)
U = np.linspace(-0.4, 0.4, 60000)   # u = sinθ

COL = {"N": "#1f3b73", "w": "#1f7a4d", "d": "#c25a1a"}
ENVCOL = "#d12e5a"


# ---------------- 물리 계산 ----------------
def envelope(w):
    return np.sinc(w * U / LAM) ** 2


def pattern(N, d, w):
    phi = 2 * np.pi * d * U / LAM
    den = np.sin(phi / 2)
    grating = np.where(np.abs(den) < 1e-9, N**2, (np.sin(N * phi / 2) / den) ** 2) / N**2
    return grating * envelope(w)


def draw_panel(ax, N, d, w, color):
    I = pattern(N, d, w)
    ax.plot(U, I, color=color, lw=1.3)
    ax.fill_between(U, 0, I, color=color, alpha=0.18)
    ax.plot(U, envelope(w), "--", color=ENVCOL, lw=1.3, alpha=0.85)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(U.min(), U.max())
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_alpha(0.4)


# ================================================================
# 0. 실험 상황 2D 개념도 (셋업)
# ================================================================
def setup_figure(fname):
    fig, ax = plt.subplots(figsize=(11, 6))

    L = 10.0          # 슬릿면 -> 스크린 거리
    yP = 3.5          # 스크린 위 관찰점 높이
    theta = np.arctan2(yP, L)
    dir_ = np.array([np.cos(theta), np.sin(theta)])

    # 슬릿 (광원) — 원점 근처 2개
    N, dsp, wsp = 2, 1.35, 0.6
    centers = dsp * (np.arange(N) - (N - 1) / 2)
    ybar = 5.2

    # 장벽 (슬릿 사이 불투명 부분)
    edges = [-ybar]
    for c in centers:
        edges += [c - wsp / 2, c + wsp / 2]
    edges += [ybar]
    for i in range(0, len(edges), 2):
        y0, y1 = edges[i], edges[i + 1]
        ax.add_patch(Rectangle((-0.12, y0), 0.24, y1 - y0,
                               facecolor="0.25", edgecolor="none", zorder=3))

    # 들어오는 평면파 (같은 위상)
    for xw in [-2.7, -2.2, -1.7]:
        ax.plot([xw, xw], [-4, 4], color="#4a90d9", lw=1.6, alpha=0.55)
    for yw in [-3, -1.6, 1.6, 3]:
        ax.annotate("", xy=(-0.55, yw), xytext=(-1.45, yw),
                    arrowprops=dict(arrowstyle="->", color="#3a80c9", lw=1.2, alpha=0.8))
    ax.text(-2.2, 4.6, "incoming wave\n(in phase)", ha="center", va="bottom",
            fontsize=10, color="#2a6bb0")

    # 슬릿 -> 관찰점 P 로 가는 광선
    for c in centers:
        ax.plot([0, L], [c, yP], color="0.62", lw=0.8, zorder=1)

    # 중심축(점선) + P 로 가는 기준선(빨강) + 각도 θ
    ax.plot([0, L], [0, 0], "--", color="0.5", lw=1)
    ax.plot([0, L], [0, yP], color="crimson", lw=1.3, zorder=2)
    ax.add_patch(Arc((0, 0), 3.2, 3.2, angle=0, theta1=0,
                     theta2=np.degrees(theta), color="crimson", lw=1.6))
    ax.text(1.25, 0.16, r"$\theta$", fontsize=15, color="crimson")

    # 스크린
    ax.plot([L, L], [-5, 5], color="black", lw=4)
    ax.text(L + 0.25, 4.6, "screen", rotation=90, va="top", ha="left", fontsize=11)
    ax.plot(L, yP, "o", color="crimson", ms=8, zorder=4)
    ax.text(L + 0.25, yP, "P", fontsize=14, color="crimson", va="center")

    # 거리 L
    ax.annotate("", xy=(L, -5.9), xytext=(0, -5.9),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax.text(L / 2, -6.45, "L", ha="center", fontsize=13)

    # 간격 d (위 두 슬릿 중심 사이)
    ct, ct2 = centers[-1], centers[-2]
    xbr = -0.95
    ax.annotate("", xy=(xbr, ct), xytext=(xbr, ct2),
                arrowprops=dict(arrowstyle="<->", color=COL["d"], lw=1.8))
    ax.text(xbr - 0.35, (ct + ct2) / 2, "d", ha="right", va="center",
            fontsize=14, color=COL["d"])

    # 폭 w (맨 아래 슬릿) — 왼쪽 틈 옆 세로 화살표 (광선과 안 겹치게)
    cb, xbw = centers[0], -0.45
    ax.annotate("", xy=(xbw, cb - wsp / 2), xytext=(xbw, cb + wsp / 2),
                arrowprops=dict(arrowstyle="<->", color=COL["w"], lw=1.8,
                                mutation_scale=11))
    ax.plot([-0.12, xbw], [cb - wsp / 2] * 2, color=COL["w"], lw=0.8)
    ax.plot([-0.12, xbw], [cb + wsp / 2] * 2, color=COL["w"], lw=0.8)
    ax.text(xbw - 0.16, cb, "w", ha="right", va="center", fontsize=13, color=COL["w"])

    ax.text(0, 5.7, "slits (sources)", ha="center", fontsize=11)

    ax.set_xlim(-3.3, 12)
    ax.set_ylim(-7, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


setup_figure("00_setup.png")


# ================================================================
# 1~3. 요소별 실험 (글자 없이 값 표시만)
# ================================================================
def sweep_figure(fname, key, values, base=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), sharex=True, sharey=True)
    for ax, val in zip(axes, values):
        p = dict(base if base is not None else BASE)
        p[key] = val
        draw_panel(ax, p["N"], p["d"], p["w"], COL[key])
        ax.set_title(f"{key} = {val:g}", fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


sweep_figure("01_sweep_N.png", "N", [2, 4, 8])
sweep_figure("02_sweep_w.png", "w", [1, 2, 4], base={**BASE, "N": 2})
sweep_figure("03_sweep_d.png", "d", [4, 8, 16], base={**BASE, "N": 2})


# ================================================================
# 4. 분해 그림 (글자 없이)
# ================================================================
def decompose_figure(fname):
    N, d, w = BASE["N"], BASE["d"], BASE["w"]
    phi = 2 * np.pi * d * U / LAM
    den = np.sin(phi / 2)
    grating = np.where(np.abs(den) < 1e-9, N**2, (np.sin(N * phi / 2) / den) ** 2) / N**2
    env = envelope(w)
    total = grating * env

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    series = [(grating, COL["d"], None),
              (env, ENVCOL, None),
              (total, COL["N"], env)]
    for ax, (y, c, dashed) in zip(axes, series):
        ax.plot(U, y, color=c, lw=1.4)
        ax.fill_between(U, 0, y, color=c, alpha=0.18)
        if dashed is not None:
            ax.plot(U, dashed, "--", color=ENVCOL, lw=1.2, alpha=0.8)
        ax.set_ylim(0, 1.08)
        ax.set_xlim(U.min(), U.max())
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_alpha(0.4)
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


decompose_figure("04_decompose.png")

print("\n완료! 생성 파일: 00_setup, 01_sweep_N, 02_sweep_w, 03_sweep_d, 04_decompose (.png)")
