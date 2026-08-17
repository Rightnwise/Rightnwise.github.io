"""
Sweep 별 실험 셋업 개념도 (00_setup 스타일 재사용)
  - setup_N.png : 광원 개수 N = 4
  - setup_w.png : 슬릿 폭 w 가 (기준보다) 큼
  - setup_d.png : 광원 간격 d 가 (기준보다) 큼
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

COL = {"d": "#c25a1a", "w": "#1f7a4d"}


def draw_setup(N, dsp, wsp, title, fname, emph=None):
    """emph='d'/'w'/'N' 이면 해당 요소를 굵게 강조."""
    fig, ax = plt.subplots(figsize=(11, 6))
    L, yP = 10.0, 3.5
    theta = np.arctan2(yP, L)
    centers = dsp * (np.arange(N) - (N - 1) / 2)
    ybar = 5.2

    # 장벽 (슬릿 사이 불투명)
    edges = [-ybar]
    for c in centers:
        edges += [c - wsp / 2, c + wsp / 2]
    edges += [ybar]
    for i in range(0, len(edges), 2):
        y0, y1 = edges[i], edges[i + 1]
        ax.add_patch(Rectangle((-0.12, y0), 0.24, y1 - y0,
                               facecolor="0.25", edgecolor="none", zorder=3))

    # 들어오는 평면파
    for xw in [-2.7, -2.2, -1.7]:
        ax.plot([xw, xw], [-4, 4], color="#4a90d9", lw=1.6, alpha=0.55)
    for yw in [-3, -1.6, 1.6, 3]:
        ax.annotate("", xy=(-0.55, yw), xytext=(-1.45, yw),
                    arrowprops=dict(arrowstyle="->", color="#3a80c9", lw=1.2, alpha=0.8))
    ax.text(-2.2, 4.6, "incoming wave\n(in phase)", ha="center", va="bottom",
            fontsize=10, color="#2a6bb0")

    # 광선
    for c in centers:
        ax.plot([0, L], [c, yP], color="0.62", lw=0.8, zorder=1)

    # 중심축 + P 기준선 + θ
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

    # 간격 d (위 두 광원)
    dlw = 3.2 if emph == "d" else 1.8
    ct, ct2 = centers[-1], centers[-2]
    xbr = -0.95
    ax.annotate("", xy=(xbr, ct), xytext=(xbr, ct2),
                arrowprops=dict(arrowstyle="<->", color=COL["d"], lw=dlw))
    ax.text(xbr - 0.35, (ct + ct2) / 2, "d", ha="right", va="center",
            fontsize=15 if emph == "d" else 14, color=COL["d"],
            fontweight="bold" if emph == "d" else "normal")

    # 폭 w (맨 아래 슬릿, 왼쪽)
    wlw = 3.2 if emph == "w" else 1.8
    cb, xbw = centers[0], -0.45
    ax.annotate("", xy=(xbw, cb - wsp / 2), xytext=(xbw, cb + wsp / 2),
                arrowprops=dict(arrowstyle="<->", color=COL["w"], lw=wlw, mutation_scale=12))
    ax.plot([-0.12, xbw], [cb - wsp / 2] * 2, color=COL["w"], lw=0.8)
    ax.plot([-0.12, xbw], [cb + wsp / 2] * 2, color=COL["w"], lw=0.8)
    ax.text(xbw - 0.16, cb, "w", ha="right", va="center",
            fontsize=15 if emph == "w" else 13, color=COL["w"],
            fontweight="bold" if emph == "w" else "normal")

    ax.text(0, 5.7, "slits (sources)", ha="center", fontsize=11)

    ax.set_xlim(-3.3, 12)
    ax.set_ylim(-7, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


# 기준: N=2, dsp=1.4, wsp=0.55
draw_setup(4, 1.4, 0.55, "Sweep N  —  sources N = 4", "setup_N.png", emph="N")
draw_setup(2, 1.4, 1.20, "Sweep w  —  larger slit width w", "setup_w.png", emph="w")
draw_setup(2, 2.8, 0.55, "Sweep d  —  larger spacing d", "setup_d.png", emph="d")
