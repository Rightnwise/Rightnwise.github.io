"""
Tuned Mass Damper (TMD) — 2자유도 연립 미분방정식 풀이/시각화

  m1 x1'' + c1 x1' + k1 x1 + k2(x1-x2) + c2(x1'-x2') = F(t)
  m2 x2'' + c2(x2'-x1') + k2(x2-x1) = 0

왼쪽 : 주파수 응답 |X1(w)|  — 흡진기 유무 비교 (단일 피크 -> 골짜기+두 피크)
오른쪽: 시간 응답 x1(t) — 원래 공진주파수로 가진했을 때 진폭 억제
scipy 없이 numpy(복소 선형대수) + 직접 RK4 사용.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── 파라미터 (정규화: 주 공진 w1 = 1) ──
m1, k1 = 1.0, 1.0
zeta1 = 0.01
c1 = 2 * zeta1 * np.sqrt(k1 * m1)        # 주 구조물의 약한 감쇠

mu = 0.10                                # 질량비 m2/m1
m2 = mu * m1
w2 = 1.0                                 # 흡진기를 주 공진(w1=1)에 튜닝
k2 = m2 * w2 ** 2
zeta2 = 0.05
c2 = 2 * zeta2 * np.sqrt(k2 * m2)        # 흡진기 감쇠

w1 = np.sqrt(k1 / m1)

# ── 주파수 응답: phasor (F = e^{jwt}) ──
w = np.linspace(0.4, 1.6, 2000)
X1_no = 1.0 / (-w**2 * m1 + 1j*w*c1 + k1)            # 흡진기 없음 (단일 DOF)

X1_tmd = np.empty_like(w, dtype=complex)
for i, wi in enumerate(w):
    A = np.array([
        [-wi**2*m1 + 1j*wi*(c1+c2) + (k1+k2), -(k2 + 1j*wi*c2)],
        [-(k2 + 1j*wi*c2),                      -wi**2*m2 + 1j*wi*c2 + k2],
    ], dtype=complex)
    X = np.linalg.solve(A, np.array([1.0, 0.0], dtype=complex))
    X1_tmd[i] = X[0]

# ── 시간 응답: RK4 로 가진 w1 에서 비교 ──
def deriv(t, s, with_tmd, wF):
    x1, v1, x2, v2 = s
    F = np.cos(wF * t)
    if with_tmd:
        a1 = (F - c1*v1 - k1*x1 - k2*(x1-x2) - c2*(v1-v2)) / m1
        a2 = (-c2*(v2-v1) - k2*(x2-x1)) / m2
    else:
        a1 = (F - c1*v1 - k1*x1) / m1
        a2 = 0.0
    return np.array([v1, a1, v2, a2])

def rk4(with_tmd, wF, T=400.0, dt=0.01):
    n = int(T/dt)
    s = np.zeros(4)
    ts = np.empty(n); xs = np.empty(n)
    t = 0.0
    for i in range(n):
        k1_ = deriv(t, s, with_tmd, wF)
        k2_ = deriv(t+dt/2, s+dt/2*k1_, with_tmd, wF)
        k3_ = deriv(t+dt/2, s+dt/2*k2_, with_tmd, wF)
        k4_ = deriv(t+dt, s+dt*k3_, with_tmd, wF)
        s = s + dt/6*(k1_ + 2*k2_ + 2*k3_ + k4_)
        ts[i], xs[i] = t, s[0]
        t += dt
    return ts, xs

t_no,  x1_no_t  = rk4(False, w1)
t_tmd, x1_tmd_t = rk4(True,  w1)

print(f"질량비 mu = {mu}, 흡진기 튜닝 w2 = {w2} (= w1)")
print(f"흡진기 없음: |X1| 최대 = {np.abs(X1_no).max():.2f} @ w={w[np.abs(X1_no).argmax()]:.3f}")
print(f"흡진기 있음: |X1| 최대 = {np.abs(X1_tmd).max():.2f}")
print(f"w=1 에서   : 없음 {np.abs(X1_no)[np.abs(w-1).argmin()]:.2f} "
      f"-> 있음 {np.abs(X1_tmd)[np.abs(w-1).argmin()]:.2f} (anti-resonance)")

# ── 그림 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(w, np.abs(X1_no),  "r-",  lw=2, label="without TMD (single peak)")
ax1.plot(w, np.abs(X1_tmd), "b-",  lw=2, label="with TMD (notch + two peaks)")
ax1.axvline(w2, color="0.6", ls="--", lw=1, label="absorber tuning $w_2$")
ax1.set_xlabel("forcing frequency  w / w1")
ax1.set_ylabel("main-mass amplitude  |X1|")
ax1.set_title(f"Frequency response  (mass ratio mu={mu})")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(t_no,  x1_no_t,  "r-", lw=1, alpha=0.8, label="without TMD")
ax2.plot(t_tmd, x1_tmd_t, "b-", lw=1, label="with TMD")
ax2.set_xlabel("time")
ax2.set_ylabel("main-mass displacement  x1(t)")
ax2.set_title("Time response, driven at resonance (w = w1)")
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.suptitle("Tuned Mass Damper: absorber suppresses the main resonance", fontsize=14)
fig.tight_layout()
fig.savefig("tmd_response.png", dpi=120)
print("그래프를 tmd_response.png 로 저장했습니다.")
plt.show()
