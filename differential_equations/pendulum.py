"""
진자의 운동방정식 (Nonlinear Pendulum)

2계 미분방정식:
    d²θ/dt² = -(g/L) * sin(θ) - b * dθ/dt

1계 연립으로 변환 (θ: 각도, ω: 각속도):
    dθ/dt = ω
    dω/dt = -(g/L) * sin(θ) - b * ω

b = 0 이면 감쇠 없는 이상 진자, b > 0 이면 감쇠 진자.
"""

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

G = 9.81   # 중력가속도 (m/s^2)
L = 1.0    # 진자 길이 (m)


def pendulum_derivs(state, b):
    """진자 미분방정식: state = (theta, omega)"""
    theta, omega = state
    dtheta = omega
    domega = -(G / L) * np.sin(theta) - b * omega
    return np.array([dtheta, domega])


def integrate(state0, t, b):
    """4차 룽게-쿠타(RK4)로 적분"""
    out = np.zeros((len(t), 2))
    out[0] = state0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        cur = out[i - 1]
        k1 = pendulum_derivs(cur, b)
        k2 = pendulum_derivs(cur + 0.5 * dt * k1, b)
        k3 = pendulum_derivs(cur + 0.5 * dt * k2, b)
        k4 = pendulum_derivs(cur + dt * k3, b)
        out[i] = cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def main():
    t = np.linspace(0, 20, 4000)

    # ── 시계열: 감쇠 없는 진자 vs 감쇠 진자 ──────────────
    theta0 = np.radians(150)   # 초기 각도 (큰 진폭: 비선형 효과)
    omega0 = 0.0

    sol_ideal = integrate((theta0, omega0), t, b=0.0)   # 감쇠 없음
    sol_damped = integrate((theta0, omega0), t, b=0.5)  # 감쇠 있음

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # (1) 시계열 (각도)
    ax1.plot(t, np.degrees(sol_ideal[:, 0]), "b", lw=1.2,
             label="감쇠 없음 (b=0)")
    ax1.plot(t, np.degrees(sol_damped[:, 0]), "r", lw=1.2,
             label="감쇠 있음 (b=0.5)")
    ax1.axhline(0, color="gray", lw=0.6)
    ax1.set_xlabel("시간 (s)")
    ax1.set_ylabel("각도 θ (도)")
    ax1.set_title(f"진자 시계열 (초기각 {np.degrees(theta0):.0f}°)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── (2) 위상 공간 벡터장 + 여러 궤적 ────────────────
    th = np.linspace(-2 * np.pi, 2 * np.pi, 30)
    om = np.linspace(-10, 10, 30)
    TH, OM = np.meshgrid(th, om)
    dTH = OM
    dOM = -(G / L) * np.sin(TH)            # 감쇠 없는 벡터장
    speed = np.sqrt(dTH ** 2 + dOM ** 2)

    ax2.streamplot(TH, OM, dTH, dOM, color=speed, cmap="viridis",
                   density=1.3, linewidth=0.8)

    # 여러 에너지 준위의 궤적 (감쇠 없음)
    t2 = np.linspace(0, 10, 3000)
    for th0 in np.radians([30, 90, 150, 179]):
        s = integrate((th0, 0.0), t2, b=0.0)
        ax2.plot(s[:, 0], s[:, 1], lw=1.3)
    # 회전(over-the-top) 궤적: 큰 초기 각속도
    for w0 in [6.5, 7.5]:
        s = integrate((-2 * np.pi, w0), t2, b=0.0)
        ax2.plot(s[:, 0], s[:, 1], "k--", lw=1.0, alpha=0.7)

    # 평형점: 아래(안정 중심) / 위(불안정 안장)
    ax2.plot([0], [0], "go", ms=9, label="안정 평형 (아래, 중심)")
    ax2.plot([-np.pi, np.pi], [0, 0], "rx", ms=11, mew=2,
             label="불안정 평형 (위, 안장점)")

    ax2.set_xlabel("각도 θ (rad)")
    ax2.set_ylabel("각속도 ω (rad/s)")
    ax2.set_title("위상 공간 벡터장 (감쇠 없음)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(-2 * np.pi, 2 * np.pi)
    ax2.set_ylim(-10, 10)

    fig.suptitle(
        f"진자의 운동방정식  d²θ/dt² = -(g/L)·sinθ - b·ω   (g={G}, L={L})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig("pendulum.png", dpi=150)
    print("그래프를 pendulum.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
