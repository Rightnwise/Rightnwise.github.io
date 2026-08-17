"""
초기 위치 함수 f(x) = u(x,0) : 가운데에서 뜯은 대칭 삼각형.

    f(x) = (2h/L) x        0 <= x <= L/2
    f(x) = (2h/L)(L-x)     L/2 <= x <= L
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    L = 1.0
    h = L / 2                                   # 꼭짓점 높이 = L/2 (기울기 ±1)
    x = np.linspace(0, L, 400)
    f = np.where(x <= L / 2, (2 * h / L) * x, (2 * h / L) * (L - x))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

    ax.plot(x, f, color="crimson", lw=3, zorder=3)
    ax.fill_between(x, f, 0, color="crimson", alpha=0.08)
    ax.axhline(0, color="0.6", lw=1)

    # 양 끝 고정 + 꼭짓점 표시
    ax.plot([0, L], [0, 0], "o", color="black", ms=10, zorder=5)
    ax.plot(L / 2, h, "o", color="crimson", ms=9, zorder=5)

    # 보조 점선 (꼭짓점 높이 h, 위치 L/2)
    ax.plot([L / 2, L / 2], [0, h], "--", color="0.6", lw=1)
    ax.plot([0, L / 2], [h, h], "--", color="0.6", lw=1)

    # 라벨
    ax.text(L / 2, h + 0.03, "peak height  L/2", ha="center",
            color="crimson", fontsize=13)
    ax.text(L / 4, (h / 2) + 0.02, r"slope $= 1$",
            ha="center", color="0.3", fontsize=12, rotation=38)
    ax.text(3 * L / 4, (h / 2) + 0.02, r"slope $= -1$",
            ha="center", color="0.3", fontsize=12, rotation=-38)

    ax.set_xlim(-0.05, L + 0.05)
    ax.set_ylim(-0.1, h + 0.15)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x) = u(x, 0)")
    ax.set_xticks([0, L / 2, L])
    ax.set_xticklabels(["0", "L/2", "L"])
    ax.set_yticks([0, h])
    ax.set_yticklabels(["0", "L/2"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig("initial_triangle.png", dpi=130, facecolor="white")
    print("저장: initial_triangle.png")
    plt.show()


if __name__ == "__main__":
    main()
