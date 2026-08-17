"""
로렌츠 끌개 (Lorenz Attractor)

dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z

표준 카오스 파라미터: sigma=10, rho=28, beta=8/3
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 투영 등록용)

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def lorenz_derivs(state, sigma, rho, beta):
    """로렌츠 미분방정식 정의"""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])


def integrate(state0, t, sigma, rho, beta):
    """4차 룽게-쿠타(RK4)로 적분"""
    out = np.zeros((len(t), 3))
    out[0] = state0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        cur = out[i - 1]
        k1 = lorenz_derivs(cur, sigma, rho, beta)
        k2 = lorenz_derivs(cur + 0.5 * dt * k1, sigma, rho, beta)
        k3 = lorenz_derivs(cur + 0.5 * dt * k2, sigma, rho, beta)
        k4 = lorenz_derivs(cur + dt * k3, sigma, rho, beta)
        out[i] = cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def main():
    # 표준 카오스 파라미터
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # 초기 조건
    state0 = (1.0, 1.0, 1.0)

    # 시간
    t = np.linspace(0, 50, 10000)

    # 적분
    sol = integrate(state0, t, sigma, rho, beta)
    x, y, z = sol.T

    fig = plt.figure(figsize=(14, 6))

    # (1) 시계열: x, y, z 가 시간에 따라 변하는 모습
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t, x, lw=0.8, label="x")
    ax1.plot(t, y, lw=0.8, label="y")
    ax1.plot(t, z, lw=0.8, label="z")
    ax1.set_xlabel("시간")
    ax1.set_ylabel("값")
    ax1.set_title("로렌츠 끌개: 시계열")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # (2) 3D 위상 공간: 나비 모양 끌개
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(x, y, z, lw=0.4, color="purple")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("위상 공간 (3D Phase Space)")

    fig.suptitle(
        f"로렌츠 끌개 (sigma={sigma}, rho={rho}, beta=8/3)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("lorenz_attractor.png", dpi=150)
    print("그래프를 lorenz_attractor.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
