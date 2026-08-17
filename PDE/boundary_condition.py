"""
경계조건(boundary condition) 직관 그림.

양 끝이 벽에 고정된 줄: 가운데는 자유롭게 출렁이지만, 두 끝은 항상 u=0.
    u(0, t) = 0,   u(L, t) = 0     (fixed ends)
여러 시각의 줄 모양을 겹쳐 그려, '끝은 안 움직이고 가운데만 움직인다' 를 보여준다.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    L = 1.0
    x = np.linspace(0, L, 400)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")

    # 여러 시각 스냅샷: 같은 모드가 진폭을 바꿔가며 진동 (끝은 항상 0)
    phases = np.linspace(0, np.pi, 9)
    cmap = plt.cm.coolwarm
    for i, ph in enumerate(phases):
        u = np.cos(ph) * np.sin(np.pi * x / L)
        ax.plot(x, u, color=cmap(i / (len(phases) - 1)), lw=1.6, alpha=0.8)

    ax.axhline(0, color="0.6", lw=1)

    # 양 끝 '고정' 표시: 벽(해치) + 큰 점(핀)
    for xe in (0, L):
        ax.plot([xe, xe], [-1.25, 1.25], color="0.35", lw=6, solid_capstyle="butt")
        ax.plot(xe, 0, "o", color="black", ms=13, zorder=5)

    # 끝은 못 움직임 (고정), 가운데는 자유
    ax.annotate("fixed:  u(0,t) = 0", xy=(0, 0), xytext=(0.13, 0.95),
                fontsize=13, color="black",
                arrowprops=dict(arrowstyle="-|>", color="black"))
    ax.annotate("fixed:  u(L,t) = 0", xy=(L, 0), xytext=(0.62, -0.95),
                fontsize=13, color="black", ha="left",
                arrowprops=dict(arrowstyle="-|>", color="black"))

    # 가운데는 위아래로 자유롭게 움직인다는 양방향 화살표
    ax.annotate("", xy=(0.5, 1.05), xytext=(0.5, -1.05),
                arrowprops=dict(arrowstyle="<|-|>", color="0.5", lw=1.5))
    ax.text(0.53, 0.0, "free to move", color="0.4", fontsize=12, va="center")

    ax.set_xlim(-0.05, L + 0.05)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_xticks([0, L])
    ax.set_xticklabels(["0", "L"])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("boundary_condition.png", dpi=130, facecolor="white")
    print("저장: boundary_condition.png")
    plt.show()


if __name__ == "__main__":
    main()
