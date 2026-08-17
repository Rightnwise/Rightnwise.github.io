"""
로렌츠 끌개의 나비효과 (초기조건 민감성, Sensitive Dependence on Initial Conditions)

두 궤적을 거의 동일한 초기조건(차이 1e-8)에서 출발시켜
시간이 지남에 따라 완전히 달라지는 모습을 비교한다.
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

    # 거의 동일한 두 초기조건 (x 좌표만 1e-8 차이)
    epsilon = 1e-8
    state_a = (1.0, 1.0, 1.0)
    state_b = (1.0 + epsilon, 1.0, 1.0)

    # 시간
    t = np.linspace(0, 50, 10000)

    # 두 궤적 적분
    a = integrate(state_a, t, sigma, rho, beta)
    b = integrate(state_b, t, sigma, rho, beta)

    # 두 궤적 사이의 유클리드 거리
    dist = np.sqrt(np.sum((a - b) ** 2, axis=1))

    fig = plt.figure(figsize=(15, 9))

    # (1) x(t) 비교 시계열
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(t, a[:, 0], "b", lw=0.8, label=f"궤적 A  (x0=1.0)")
    ax1.plot(t, b[:, 0], "r", lw=0.8, label=f"궤적 B  (x0=1.0+{epsilon:g})")
    ax1.set_xlabel("시간")
    ax1.set_ylabel("x")
    ax1.set_title("x(t) 비교: 처음엔 겹치다가 어느 순간 갈라짐")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (2) 두 궤적 사이 거리 (로그 스케일) — 지수적 발산
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.semilogy(t, dist, "purple", lw=1.0)
    ax2.set_xlabel("시간")
    ax2.set_ylabel("두 궤적 사이 거리 (log)")
    ax2.set_title("거리의 지수적 증가 (양의 리아푸노프 지수)")
    ax2.grid(True, alpha=0.3, which="both")

    # (3) 3D 위상 공간에서 두 궤적
    ax3 = fig.add_subplot(2, 2, (3, 4), projection="3d")
    ax3.plot(a[:, 0], a[:, 1], a[:, 2], "b", lw=0.4, alpha=0.7, label="궤적 A")
    ax3.plot(b[:, 0], b[:, 1], b[:, 2], "r", lw=0.4, alpha=0.7, label="궤적 B")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_title("3D 위상 공간: 같은 끌개 위지만 서로 다른 경로")
    ax3.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"로렌츠 끌개의 나비효과 — 초기조건 차이 {epsilon:g}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig("lorenz_butterfly.png", dpi=150)
    print("그래프를 lorenz_butterfly.png 로 저장했습니다.")

    # 두 궤적이 눈에 띄게 갈라지기 시작하는 대략적 시점
    threshold = 1.0
    idx = np.argmax(dist > threshold)
    if dist[idx] > threshold:
        print(f"두 궤적의 거리가 {threshold} 를 넘는 시점: t ≈ {t[idx]:.2f}")

    plt.show()


if __name__ == "__main__":
    main()
