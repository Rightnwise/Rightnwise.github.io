"""
AP Calculus BC: 지수 성장(Exponential Growth)과 로지스틱 성장(Logistic Growth)

지수 성장:
    dy/dt = k*y          (해:  y = y0 * e^(k*t))

로지스틱 성장:
    dy/dt = k*y*(1 - y/M) (해:  y = M / (1 + A*e^(-k*t)),  A = (M - y0)/y0)
    M: 수용력(carrying capacity)
"""

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    # 파라미터
    k = 0.8        # 성장률
    y0 = 10.0      # 초기값
    M = 100.0      # 수용력 (로지스틱)

    t = np.linspace(0, 10, 500)

    # 해석적 해 (AP BC 공식)
    exponential = y0 * np.exp(k * t)
    A = (M - y0) / y0
    logistic = M / (1 + A * np.exp(-k * t))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ── (1) 지수 성장 ──────────────────────────────────
    ax1.plot(t, exponential, "b", lw=2, label="$y = y_0 e^{kt}$")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Population y")
    ax1.set_title(f"Exponential Growth (k={k}, y0={y0:g})")
    ax1.set_ylim(0, M * 1.6)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── (2) 로지스틱 성장 ──────────────────────────────
    ax2.plot(t, logistic, "r", lw=2, label="$y = \\dfrac{M}{1 + Ae^{-kt}}$")
    ax2.axhline(M, color="gray", ls="--", lw=1, label=f"Carrying capacity M = {M:g}")

    # 변곡점: 로지스틱에서 y = M/2 일 때 (성장이 가장 빠른 지점)
    t_inflect = np.log(A) / k
    ax2.plot(t_inflect, M / 2, "ko", ms=8)
    ax2.annotate(f"Inflection (y = M/2)\n t $\\approx$ {t_inflect:.2f}",
                 xy=(t_inflect, M / 2), xytext=(t_inflect + 1.2, M / 2 - 18),
                 arrowprops=dict(arrowstyle="->"))

    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Population y")
    ax2.set_title(f"Logistic Growth (k={k}, y0={y0:g}, M={M:g})")
    ax2.set_ylim(0, M * 1.6)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("AP Calculus BC: Exponential vs Logistic Growth", fontsize=14)
    fig.tight_layout()
    fig.savefig("growth_models.png", dpi=150)
    print("그래프를 growth_models.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
