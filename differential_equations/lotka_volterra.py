"""
로트카-볼테라 방정식 (Lotka-Volterra, 포식자-피식자 모델)

dx/dt =  alpha * x - beta  * x * y     (피식자: prey)
dy/dt = -gamma * y + delta * x * y     (포식자: predator)

x: 피식자 개체수 (예: 토끼)
y: 포식자 개체수 (예: 여우)
"""

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def lv_derivs(state, alpha, beta, gamma, delta):
    """로트카-볼테라 미분방정식 정의"""
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = -gamma * y + delta * x * y
    return np.array([dxdt, dydt])


def integrate(state0, t, alpha, beta, gamma, delta):
    """4차 룽게-쿠타(RK4)로 적분"""
    out = np.zeros((len(t), 2))
    out[0] = state0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        cur = out[i - 1]
        k1 = lv_derivs(cur, alpha, beta, gamma, delta)
        k2 = lv_derivs(cur + 0.5 * dt * k1, alpha, beta, gamma, delta)
        k3 = lv_derivs(cur + 0.5 * dt * k2, alpha, beta, gamma, delta)
        k4 = lv_derivs(cur + dt * k3, alpha, beta, gamma, delta)
        out[i] = cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def main():
    # 모델 파라미터
    alpha = 1.0   # 피식자 자연 증가율
    beta = 0.1    # 포식에 의한 피식자 감소율
    gamma = 1.5   # 포식자 자연 사망률
    delta = 0.075  # 포식자 증가율 (포식 효율)

    # 초기 개체수
    x0, y0 = 10.0, 5.0

    # 시간
    t = np.linspace(0, 50, 2000)

    # 적분
    result = integrate((x0, y0), t, alpha, beta, gamma, delta)
    prey, predator = result.T

    # 그래프: (1) 시간에 따른 개체수, (2) 위상 평면
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (1) 시계열
    ax1.plot(t, prey, "b", lw=2, label="피식자 (토끼)")
    ax1.plot(t, predator, "r", lw=2, label="포식자 (여우)")
    ax1.set_xlabel("시간")
    ax1.set_ylabel("개체수")
    ax1.set_title("로트카-볼테라: 시간에 따른 개체수 변화")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # (2) 위상 평면 (피식자 vs 포식자)
    ax2.plot(prey, predator, "purple", lw=1.5)
    ax2.set_xlabel("피식자 개체수")
    ax2.set_ylabel("포식자 개체수")
    ax2.set_title("위상 평면 (Phase Portrait)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"로트카-볼테라 방정식 "
        f"(alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("lotka_volterra.png", dpi=150)
    print("그래프를 lotka_volterra.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
