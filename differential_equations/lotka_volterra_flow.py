"""
로트카-볼테라 방정식의 벡터장 (Vector Flow / Phase Portrait)

dx/dt =  alpha * x - beta  * x * y     (피식자: prey)
dy/dt = -gamma * y + delta * x * y     (포식자: predator)

위상 평면 위에서 흐름 방향(벡터장)과 궤적, 평형점을 함께 표시한다.

평형점:
  (0, 0)                         - 둘 다 멸종
  (gamma/delta, alpha/beta)      - 공존 (중심, center)
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
    """4차 룽게-쿠타(RK4)로 궤적 적분"""
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
    alpha = 1.0
    beta = 0.1
    gamma = 1.5
    delta = 0.075

    # 평형점 (공존)
    x_eq = gamma / delta   # = 20
    y_eq = alpha / beta    # = 10

    # 위상 평면 격자
    x = np.linspace(0, 40, 25)
    y = np.linspace(0, 25, 25)
    X, Y = np.meshgrid(x, y)

    # 각 격자점에서의 벡터(흐름)
    U = alpha * X - beta * X * Y
    V = -gamma * Y + delta * X * Y
    speed = np.sqrt(U ** 2 + V ** 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # ── (1) quiver: 화살표 벡터장 ──────────────────────────
    # 화살표 길이를 정규화해 방향이 잘 보이도록
    Un = U / (speed + 1e-9)
    Vn = V / (speed + 1e-9)
    q = ax1.quiver(X, Y, Un, Vn, speed, cmap="viridis", pivot="mid")
    fig.colorbar(q, ax=ax1, label="흐름 속도 |(dx/dt, dy/dt)|")

    # 닫힌 궤적 몇 개 겹쳐 그리기
    t = np.linspace(0, 20, 4000)
    for x0 in [12, 14, 16, 18]:
        sol = integrate((x0, 5.0), t, alpha, beta, gamma, delta)
        ax1.plot(sol[:, 0], sol[:, 1], lw=1.2)

    ax1.plot(x_eq, y_eq, "r*", ms=16, label=f"평형점 ({x_eq:g}, {y_eq:g})")
    ax1.plot(0, 0, "ko", ms=7, label="멸종점 (0, 0)")
    ax1.set_xlabel("피식자 개체수 (x)")
    ax1.set_ylabel("포식자 개체수 (y)")
    ax1.set_title("벡터장 (quiver) + 궤적")
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 25)

    # ── (2) streamplot: 유선(streamline) ──────────────────
    strm = ax2.streamplot(
        X, Y, U, V, color=speed, cmap="plasma", density=1.4, linewidth=1
    )
    fig.colorbar(strm.lines, ax=ax2, label="흐름 속도")
    ax2.plot(x_eq, y_eq, "r*", ms=16, label=f"평형점 ({x_eq:g}, {y_eq:g})")
    ax2.plot(0, 0, "ko", ms=7, label="멸종점 (0, 0)")
    ax2.set_xlabel("피식자 개체수 (x)")
    ax2.set_ylabel("포식자 개체수 (y)")
    ax2.set_title("유선 (streamplot)")
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, 40)
    ax2.set_ylim(0, 25)

    fig.suptitle(
        f"로트카-볼테라 벡터장 "
        f"(alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("lotka_volterra_flow.png", dpi=150)
    print("그래프를 lotka_volterra_flow.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
