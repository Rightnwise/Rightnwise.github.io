"""
직렬 RLC 회로 (series RLC circuit) — 강제 진동

회로 방정식 (q: 충전 전하, i = q' : 전류):
    L*q'' + R*q' + (1/C)*q = V(t),    V(t) = V0*cos(omega*t)

1계 연립으로 변환:
    q' = i
    i' = ( V(t) - R*i - q/C ) / L

원래 식  y'' + 3y' + 2y = cos(t)  는
    L=1, R=3, C=0.5, V0=1, omega=1  에 해당한다.

R, L, C, V0, omega, 초기조건은 모두 solve() 의 인자(parameter)로 바꿀 수 있다.
"""

import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (macOS) — 그림 안 텍스트는 영어만 사용
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def derivs(state, t, R, L, C, V0, omega):
    """state = (q, i),  i = q' ;  구동 전압 V(t)=V0*cos(omega*t)"""
    q, i = state
    Vt = V0 * np.cos(omega * t)
    dq = i
    di = (Vt - R * i - q / C) / L
    return np.array([dq, di])


def integrate(state0, t, R, L, C, V0, omega):
    """4차 룽게-쿠타(RK4)로 적분 (t 의존성 반영)"""
    out = np.zeros((len(t), 2))
    out[0] = state0
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        ti = t[k - 1]
        cur = out[k - 1]
        k1 = derivs(cur, ti, R, L, C, V0, omega)
        k2 = derivs(cur + 0.5 * dt * k1, ti + 0.5 * dt, R, L, C, V0, omega)
        k3 = derivs(cur + 0.5 * dt * k2, ti + 0.5 * dt, R, L, C, V0, omega)
        k4 = derivs(cur + dt * k3, ti + dt, R, L, C, V0, omega)
        out[k] = cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def auto_timescale(R, L, C, omega, periods=12, ppp=80):
    """RLC 값으로부터 안정적인 (t_max, n) 을 자동 계산.

    explicit RK4 는 스텝 dt 가 고유 주기보다 크면 발산하므로,
    가장 빠른 진동(고유 또는 구동)을 ppp 점으로 분해하고
    감쇠가 충분히 보이도록 시간 창을 잡는다.
    """
    omega0 = 1.0 / np.sqrt(L * C)          # 고유 각주파수
    w_fast = max(omega0, omega)            # 분해해야 할 가장 빠른 진동
    T_fast = 2 * np.pi / w_fast
    tau = (2.0 * L / R) if R > 0 else np.inf  # 포락선 감쇠 시정수

    # 시간 창: 감쇠가 충분히 진행되고(8*tau) 진동도 여러 번 보이게
    t_decay = 8 * tau if np.isfinite(tau) else periods * T_fast
    t_max = max(t_decay, periods * T_fast)

    dt = T_fast / ppp                      # 안정/정확성 위한 작은 스텝
    n = int(t_max / dt) + 1
    n = min(max(n, 500), 2_000_000)        # 안전 범위로 제한
    return t_max, n


def solve(R=3.0, L=1.0, C=0.5, V0=1.0, omega=1.0,
          q0=0.0, i0=0.0, t_max=None, n=None):
    """RLC 수치와 초기조건을 parameter 로 받아 해를 반환.

    R     : 저항 (Ohm)
    L     : 인덕턴스 (Henry)
    C     : 정전용량 (Farad)
    V0    : 구동 전압 진폭
    omega : 구동 각주파수
    q0    : q(0)  (초기 전하)
    i0    : q'(0) (초기 전류)
    t_max : 시뮬레이션 시간 (None 이면 RLC 값에 맞춰 자동)
    n     : 적분 스텝 수 (None 이면 자동)
    """
    if t_max is None or n is None:
        auto_tmax, auto_n = auto_timescale(R, L, C, omega)
        t_max = auto_tmax if t_max is None else t_max
        n = auto_n if n is None else n
    t = np.linspace(0, t_max, n)
    sol = integrate((q0, i0), t, R, L, C, V0, omega)
    return t, sol[:, 0], sol[:, 1]


def main():
    # ── RLC 수치 (여기만 바꾸면 됨) ──────────────────────
    R = 6000     # 저항
    L = 2      # 인덕턴스
    C = 100 * 10**(-5)      # 정전용량
    V0 = 1.0     # 구동 전압 진폭
    omega = 500  # 구동 각주파수

    # ── 초기조건 ────────────────────────────────────────
    q0 = 0.0     # q(0)
    i0 = 0.0     # q'(0)

    t, q, _ = solve(R=R, L=L, C=C, V0=V0, omega=omega, q0=q0, i0=i0)

    omega0 = 1.0 / np.sqrt(L * C)   # 고유 각주파수

    # 전하 q(~1e-7 C) 와 전압 V(~1 V) 는 크기가 매우 다르므로 이중 y축 사용
    fig, ax = plt.subplots(figsize=(10, 6))
    line_q, = ax.plot(t, q, "b", lw=2, label="$q(t)$  (charge)")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Charge q (C)", color="b")
    ax.tick_params(axis="y", labelcolor="b")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    line_v, = ax2.plot(t, V0 * np.cos(omega * t), "g", lw=1.2, alpha=0.6,
                       label="$V(t)=V_0\\cos(\\omega t)$")
    ax2.set_ylabel("Drive V (V)", color="g")
    ax2.tick_params(axis="y", labelcolor="g")

    ax.legend(handles=[line_q, line_v], loc="upper right")
    ax.set_title(f"Series RLC:  $Lq'' + Rq' + q/C = V_0\\cos(\\omega t)$\n"
                 f"(R={R:g}, L={L:g}, C={C:g}, "
                 f"$\\omega_0$={omega0:.0f} rad/s, $\\omega$={omega:g})")

    fig.tight_layout()
    fig.savefig("forced_oscillator.png", dpi=150)
    print("그래프를 forced_oscillator.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
