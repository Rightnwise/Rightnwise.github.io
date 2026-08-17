"""
lotka_volterra_allee.py
================================================================
로트카-볼테라(포식-피식) 모델에 알리 효과(Allee effect)를 더한 위상궤도.

고전 로트카-볼테라
  dx/dt = a x - b x y            (피식자 x)
  dy/dt = c x y - d y            (포식자 y)
  → 평형점 주위로 '중립 폐곡선(closed orbits)'만 돈다(중심).

알리 효과 추가 (강한 알리, 피식자 성장에)
  dx/dt = r x (1 - x/K)(x/A - 1) - b x y
  dy/dt = c x y - m y
  · x < A  : 성장률 < 0  → 피식자가 임계밀도 A 아래면 스스로 멸종(알리)
  · A<x<K : 성장률 > 0
  → 결과: (0,0) '멸종'이 끌개가 되고, 공존 평형점과 멸종 사이에
    분리선(separatrix)/멸종 분지가 생긴다. 초기값에 따라 공존 vs 공멸.

위상평면(x=피식자, y=포식자)에 벡터장·널클라인·평형점·여러 궤도를 그린다.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- 파라미터 ----------------
# 고전 LV
A_LV, B_LV, C_LV, D_LV = 1.0, 1.0, 1.0, 0.7      # dx=ax-bxy, dy=cxy-dy
# 알리 모델 (공존 평형점이 널클라인 봉우리 오른쪽=안정쪽에 오도록 M 설정)
R, K, ALLEE = 1.0, 1.0, 0.15     # 피식자 성장률 / 수용력 / 알리 임계밀도
B, C, M = 1.0, 1.0, 0.7          # 포식률 / 전환율 / 포식자 사망률 (x*=m/c=0.7 > 봉우리 0.575)

XMAX, YMAX = 1.25, 1.5


def f_classic(t, z):
    x, y = z
    return [A_LV * x - B_LV * x * y, C_LV * x * y - D_LV * y]


def f_allee(t, z):
    x, y = z
    dx = R * x * (1 - x / K) * (x / ALLEE - 1) - B * x * y
    dy = C * x * y - M * y
    return [dx, dy]


def trajectory(f, z0, tmax=120, n=6000):
    sol = solve_ivp(f, (0, tmax), z0, t_eval=np.linspace(0, tmax, n),
                    rtol=1e-8, atol=1e-10, max_step=0.05)
    x, y = sol.y
    return x, y


def draw_phase(ax, f, title, nullclines, fixed_pts, inits, allee_line=None):
    # 벡터장(streamplot)
    gx, gy = np.meshgrid(np.linspace(1e-3, XMAX, 32), np.linspace(1e-3, YMAX, 32))
    U, V = f(0, [gx, gy])
    ax.streamplot(gx, gy, U, V, color="0.78", density=1.1, linewidth=0.7,
                  arrowsize=0.8, zorder=1)

    # 널클라인
    for xs, ys, c, lab in nullclines:
        ax.plot(xs, ys, color=c, lw=2, ls="--", zorder=3, label=lab)

    # 궤도
    for z0 in inits:
        x, y = trajectory(f, z0)
        # 결과에 따라 색: 멸종(피식자→0) vs 생존
        extinct = x[-1] < 1e-2 and y[-1] < 1e-2
        col = "#d1352c" if extinct else "#2f6fd0"
        ax.plot(x, y, color=col, lw=1.3, alpha=0.9, zorder=4)
        ax.plot(z0[0], z0[1], "o", color=col, ms=4, zorder=5)

    # 평형점
    for (px, py, kind) in fixed_pts:
        mk = "o" if kind == "stable" else ("s" if kind == "saddle" else "^")
        fc = "black" if kind == "stable" else "white"
        ax.plot(px, py, mk, mfc=fc, mec="black", ms=9, mew=1.4, zorder=6)

    if allee_line is not None:
        ax.axvline(allee_line, color="#e58a1a", lw=1.5, ls=":", zorder=2)
        ax.text(allee_line, YMAX * 0.96, " Allee threshold A", color="#e58a1a", fontsize=9,
                va="top")

    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX)
    ax.set_xlabel("Prey x")
    ax.set_ylabel("Predator y")
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper right")


def main():
    xs = np.linspace(1e-3, XMAX, 400)

    # ---- 고전 LV ----
    # 널클라인: 피식자 dx=0 → y=a/b ; 포식자 dy=0 → x=d/c
    null_classic = [
        (xs, np.full_like(xs, A_LV / B_LV), "#2e8b57", "Prey nullcline y=a/b"),
        (np.full_like(xs, D_LV / C_LV), np.linspace(0, YMAX, 400), "#8e44ad", "Predator nullcline x=d/c"),
    ]
    fp_classic = [(0, 0, "saddle"), (D_LV / C_LV, A_LV / B_LV, "center")]
    inits_classic = [(0.7, 0.4), (0.7, 0.6), (0.7, 0.8), (0.7, 1.25)]

    # ---- 알리 모델 ----
    # 피식자 널클라인: y = (r/b)(1-x/K)(x/A-1)  (x>0 구간에서 양수인 부분만 의미)
    y_prey = (R / B) * (1 - xs / K) * (xs / ALLEE - 1)
    xstar = M / C
    ystar = (R / B) * (1 - xstar / K) * (xstar / ALLEE - 1)
    null_allee = [
        (xs, y_prey, "#2e8b57", "Prey nullcline"),
        (np.full_like(xs, xstar), np.linspace(0, YMAX, 400), "#8e44ad", "Predator nullcline x=m/c"),
    ]
    fp_allee = [(0, 0, "stable"), (ALLEE, 0, "saddle"), (K, 0, "saddle"),
                (xstar, ystar, "spiral")]
    # 다양한 초기값(멸종·공존 분지 모두 보이게)
    inits_allee = [(0.7, 0.7), (0.85, 1.1), (0.6, 1.3), (0.9, 0.6),   # 공존 분지(안쪽으로 나선)
                   (0.10, 0.2), (0.25, 1.0), (0.3, 1.35), (0.5, 1.45),  # 멸종 분지
                   (0.95, 0.25), (0.2, 0.5)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
    draw_phase(ax1, f_classic, "Classic Lotka-Volterra (neutral closed orbits)", null_classic,
               fp_classic, inits_classic)
    draw_phase(ax2, f_allee, "Lotka-Volterra + Allee effect", null_allee, fp_allee,
               inits_allee, allee_line=ALLEE)

    # 범례 보조(평형점·궤도 색 의미)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#2f6fd0", lw=2, label="Coexistence / cyclic orbit"),
        Line2D([0], [0], color="#d1352c", lw=2, label="Extinction orbit"),
        Line2D([0], [0], marker="o", color="w", mfc="black", mec="black", ms=9, label="Stable equilibrium"),
        Line2D([0], [0], marker="^", color="w", mfc="w", mec="black", ms=9, label="Spiral / center"),
        Line2D([0], [0], marker="s", color="w", mfc="w", mec="black", ms=9, label="Saddle point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Predator-prey phase portrait: adding the Allee effect creates an 'extinction basin'", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out = os.path.join(RESULT_DIR, "lotka_volterra_allee_phase.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print("고전 LV 평형점(중심):", (D_LV / C_LV, A_LV / B_LV))
    print(f"알리 모델 공존 평형점: ({xstar:.3f}, {ystar:.3f})  "
          f"[알리 임계 A={ALLEE}, 수용력 K={K}]")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
