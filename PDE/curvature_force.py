"""
곡률(볼록함)이 복원력을 만든다 — 파동방정식 우변 u_xx 의 직관 그림.

파동방정식  u_tt = c^2 u_xx  에서, 어떤 점의 (세로) 가속도는 그 점의
'휘어짐(곡률)' u_xx 에 비례한다. 화살표로 그 힘의 방향을 보여준다.

    위로 볼록 (∩, concave down) : u_xx < 0  ->  힘/가속도 아래로
    아래로 볼록(∪, concave up ) : u_xx > 0  ->  힘/가속도 위로

즉 줄은 항상 '직선으로 펴지려는' 방향(축 쪽)으로 당겨진다.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(0, 2 * np.pi, 500)
    u = np.sin(x)                 # 봉우리 하나 + 골 하나
    u_xx = -np.sin(x)             # 2차 미분(곡률) = 가속도 방향 (∝ u_xx)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")

    # 축(평형 위치)과 줄 모양
    ax.axhline(0, color="0.6", lw=1)
    ax.plot(x, u, color="black", lw=3, zorder=3, label="string  u(x)")

    # 볼록/오목 영역 음영
    ax.fill_between(x, u, 0, where=(u > 0), color="crimson", alpha=0.08)
    ax.fill_between(x, u, 0, where=(u < 0), color="royalblue", alpha=0.08)

    # 곡률에 비례하는 힘 화살표 (방향 = sign(u_xx), 길이 ∝ |u_xx|)
    xa = np.linspace(0, 2 * np.pi, 21)[1:-1]     # 끝점 제외
    ua = np.sin(xa)
    fa = -np.sin(xa)                              # ∝ u_xx (가속도 방향)
    scale = 0.55
    for xi, ui, fi in zip(xa, ua, fa):
        if abs(fi) < 1e-2:                        # 변곡점(곡률 0)은 화살표 없음
            continue
        color = "crimson" if fi < 0 else "royalblue"
        ax.annotate("", xy=(xi, ui + scale * fi), xytext=(xi, ui),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    # 라벨
    ax.text(np.pi / 2, 1.28, "concave down (∩)\n$u_{xx}<0$  →  force DOWN",
            ha="center", va="bottom", color="crimson", fontsize=13)
    ax.text(3 * np.pi / 2, -1.28,
            "concave up (∪)\n$u_{xx}>0$  →  force UP",
            ha="center", va="top", color="royalblue", fontsize=13)

    ax.set_title(r"Why the right-hand side is $u_{xx}$:  "
                 r"curvature sets the restoring force   "
                 r"($u_{tt}=c^{2}u_{xx}$)", fontsize=14)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xlabel("x")
    ax.set_ylabel("u  (transverse displacement)")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.grid(True, alpha=0.15)
    ax.legend(loc="lower left", fontsize=11)

    fig.tight_layout()
    fig.savefig("curvature_force.png", dpi=130, facecolor="white")
    print("저장: curvature_force.png")
    plt.show()


if __name__ == "__main__":
    main()
