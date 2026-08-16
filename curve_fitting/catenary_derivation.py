"""
현수선(catenary) 방정식 유도용 자유물체도 (free-body diagram)

바닥점 (0,0) 에서 오른쪽의 한 점 P 까지의 케이블 토막에 작용하는 힘:

    - (0,0) 에서의 수평 장력  T0  (왼쪽 방향)
    - P 에서의 장력: 수평성분 = T0 (수평 장력은 일정),
                     수직성분 = T0 tan(theta0) = T0 dy/dx
      (둘의 합 = 접선 방향 장력 T)
    - 토막의 무게 W (아래 방향, 중력)

힘 평형:
    수평:  T cos(theta0) = T0
    수직:  T sin(theta0) = W
    => tan(theta0) = dy/dx = W / T0      (현수선 방정식의 출발점)

그림 안 텍스트는 영어만 사용.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def arrow(ax, tail, tip, color, lw=2.2):
    """tail -> tip 로 향하는 힘 화살표."""
    ax.annotate("", xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                mutation_scale=20))


def main():
    # 현수선 y = cosh(x) - 1  (바닥점이 원점, 거기서 접선이 수평)
    y = lambda x: np.cosh(x) - 1.0
    x_p = 1.0
    yp = y(x_p)
    dydx = np.sinh(x_p)            # P 에서의 기울기 = tan(theta0)
    theta = np.arctan(dydx)        # theta0
    theta_deg = np.degrees(theta)

    O = (0.0, 0.0)
    P = (x_p, yp)

    # 힘 화살표 크기 (데이터 단위)
    L = 0.6                        # 수평 장력 T0 길이
    Vlen = L * dydx                # 수직성분 길이 = T0 * dy/dx

    fig, ax = plt.subplots(figsize=(11, 8))

    # ── 케이블 곡선 ─────────────────────────────────────
    xf = np.linspace(-1.3, 1.3, 400)
    ax.plot(xf, y(xf), color="0.75", lw=1.5, zorder=1)          # 전체(옅게)
    xs = np.linspace(0, x_p, 200)
    ax.plot(xs, y(xs), color="black", lw=3, zorder=2,
            label="cable segment (analyzed)")                   # 분석 토막(굵게)

    # 점 표시
    ax.plot(*O, "ko", ms=7, zorder=5)
    ax.plot(*P, "ko", ms=7, zorder=5)
    ax.annotate("O = (0, 0)", O, textcoords="offset points",
                xytext=(-6, -16), fontsize=11)
    ax.annotate("P = (x, y)", P, textcoords="offset points",
                xytext=(8, -4), fontsize=11)

    # ── (1) O 에서의 수평 장력 T0 (왼쪽) ────────────────
    arrow(ax, O, (O[0] - L, O[1]), color="tab:blue")
    ax.text(O[0] - L - 0.05, O[1] + 0.04, r"$T_0$",
            color="tab:blue", fontsize=14, ha="right")
    ax.text(0.02, -0.14, "tangent is horizontal at the lowest point",
            fontsize=9, color="0.4")

    # ── 접선 (점선) 과 수평 기준선 ──────────────────────
    tx = np.array([x_p - 0.55, x_p + 0.8])
    ax.plot(tx, yp + dydx * (tx - x_p), ls=":", color="0.5", lw=1.5,
            zorder=3)
    ax.plot([x_p, x_p + L + 0.05], [yp, yp], ls="--", color="0.6", lw=1.0,
            zorder=3)

    # ── (2) P 에서의 장력 성분 ─────────────────────────
    # 수평성분 = T0 (오른쪽)
    arrow(ax, P, (P[0] + L, P[1]), color="tab:blue")
    ax.text(P[0] + L + 0.03, P[1] - 0.02, r"$T_0$",
            color="tab:blue", fontsize=14, va="top")
    # 수직성분 = T0 tan(theta0) = T0 dy/dx (위)
    arrow(ax, P, (P[0], P[1] + Vlen), color="tab:green")
    ax.text(P[0] - 0.04, P[1] + Vlen + 0.02,
            r"$T_0\tan\theta_0 = T_0\,\dfrac{dy}{dx}$",
            color="tab:green", fontsize=13, ha="right")
    # 합력 = 접선 방향 장력 T
    arrow(ax, P, (P[0] + L, P[1] + Vlen), color="tab:red", lw=2.6)
    ax.text(P[0] + L + 0.03, P[1] + Vlen + 0.02, r"$T$",
            color="tab:red", fontsize=15)

    # ── 각도 theta0 (수평선과 접선 사이) ────────────────
    arc = Arc(P, 0.55, 0.55, angle=0, theta1=0, theta2=theta_deg,
              color="black", lw=1.4)
    ax.add_patch(arc)
    la = theta / 2
    ax.text(P[0] + 0.36 * np.cos(la), P[1] + 0.36 * np.sin(la),
            r"$\theta_0$", fontsize=14)

    # ── (3) 무게 W (아래 방향, 중력) ───────────────────
    xm = 0.5
    M = (xm, y(xm))
    arrow(ax, M, (M[0], M[1] - 0.5), color="tab:purple")
    ax.text(M[0] + 0.03, M[1] - 0.5, r"$W = m g$  (weight)",
            color="tab:purple", fontsize=13, va="top")

    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 2.0)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Free-body diagram for deriving the catenary equation")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig("catenary_derivation.png", dpi=150)
    print("그래프를 catenary_derivation.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
