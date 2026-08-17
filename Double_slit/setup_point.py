"""
실험 상황 2D 개념도 — '점광원(point source)' 버전
(00_setup.png 과 똑같은 형태, 폭 w 인 슬릿 대신 점광원으로만 바뀜)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

COL_D = "#c25a1a"     # d 색 (간격)


def setup_point_figure(fname):
    fig, ax = plt.subplots(figsize=(11, 6))

    L = 10.0          # 광원면 -> 스크린 거리
    yP = 3.5          # 스크린 위 관찰점 높이
    theta = np.arctan2(yP, L)
    dir_ = np.array([np.cos(theta), np.sin(theta)])

    # 점광원 위치 (폭 없음, 간격 d) — 원점 근처 2개
    N, dsp = 2, 1.35
    centers = dsp * (np.arange(N) - (N - 1) / 2)

    # 들어오는 평면파 (같은 위상)
    for xw in [-2.7, -2.2, -1.7]:
        ax.plot([xw, xw], [-4, 4], color="#4a90d9", lw=1.6, alpha=0.55)
    for yw in [-3, -1.6, 1.6, 3]:
        ax.annotate("", xy=(-0.55, yw), xytext=(-1.45, yw),
                    arrowprops=dict(arrowstyle="->", color="#3a80c9", lw=1.2, alpha=0.8))
    ax.text(-2.2, 4.6, "incoming wave\n(in phase)", ha="center", va="bottom",
            fontsize=10, color="#2a6bb0")

    # 점광원 -> 관찰점 P 로 가는 광선
    for c in centers:
        ax.plot([0, L], [c, yP], color="0.62", lw=0.8, zorder=1)

    # 점광원 (구가 아니라 '점')
    ax.plot(np.zeros(N), centers, "o", color="#e8622a", ms=13,
            markeredgecolor="0.2", markeredgewidth=0.8, zorder=4)

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

    # 간격 d (위 두 점광원 사이)
    ct, ct2 = centers[-1], centers[-2]
    xbr = -0.95
    ax.annotate("", xy=(xbr, ct), xytext=(xbr, ct2),
                arrowprops=dict(arrowstyle="<->", color=COL_D, lw=1.8))
    ax.text(xbr - 0.35, (ct + ct2) / 2, "d", ha="right", va="center",
            fontsize=14, color=COL_D)

    ax.text(0, 5.7, "point sources", ha="center", fontsize=11)

    ax.set_xlim(-3.3, 12)
    ax.set_ylim(-7, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


setup_point_figure("00_setup_point.png")
