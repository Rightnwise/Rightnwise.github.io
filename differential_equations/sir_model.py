"""
SIR 모델 (Susceptible-Infected-Recovered) 시뮬레이션 및 그래프

dS/dt = -beta * S * I / N
dI/dt =  beta * S * I / N - gamma * I
dR/dt =  gamma * I
"""

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def sir_derivs(y, N, beta, gamma):
    """SIR 미분방정식 정의"""
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])


def integrate(y0, t, N, beta, gamma):
    """4차 룽게-쿠타(RK4)로 SIR 방정식 적분"""
    y = np.zeros((len(t), 3))
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        cur = y[i - 1]
        k1 = sir_derivs(cur, N, beta, gamma)
        k2 = sir_derivs(cur + 0.5 * dt * k1, N, beta, gamma)
        k3 = sir_derivs(cur + 0.5 * dt * k2, N, beta, gamma)
        k4 = sir_derivs(cur + dt * k3, N, beta, gamma)
        y[i] = cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def main():
    # 전체 인구
    N = 1000

    # 초기 조건: 감염자 1명, 회복자 0명, 나머지는 감수성자
    I0, R0 = 1, 0
    S0 = N - I0 - R0

    # 모델 파라미터
    beta = 0.3    # 감염률 (접촉당 전파 확률)
    gamma = 0.01   # 회복률 (1/감염 기간)

    # 기초감염재생산수 R0
    basic_reproduction = beta / gamma

    # 시간 (일 단위, 0 ~ 160일)
    t = np.linspace(0, 160, 160)

    # 미분방정식 적분
    y0 = (S0, I0, R0)
    result = integrate(y0, t, N, beta, gamma)
    S, I, R = result.T

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, S, "b", lw=2, label="감수성자 (S)")
    ax.plot(t, I, "r", lw=2, label="감염자 (I)")
    ax.plot(t, R, "g", lw=2, label="회복자 (R)")

    ax.set_xlabel("시간 (일)")
    ax.set_ylabel("인구 수")
    ax.set_title(
        f"SIR 모델 (beta={beta}, gamma={gamma}, R0={basic_reproduction:.1f})"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("sir_model.png", dpi=150)
    print("그래프를 sir_model.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
