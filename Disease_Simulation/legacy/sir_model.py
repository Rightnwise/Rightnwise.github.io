"""
SIR 전염병 모델 시뮬레이션
================================================================
인구를 세 집단으로 나눈다:
  S (Susceptible) : 감염될 수 있는 사람
  I (Infected)    : 현재 감염되어 남을 전염시키는 사람
  R (Recovered)   : 회복(또는 사망)해서 더 이상 관여 안 하는 사람

미분방정식:
  dS/dt = -β·S·I/N
  dI/dt =  β·S·I/N - γ·I
  dR/dt =  γ·I

  β = 감염 전파율(한 명이 단위시간당 감염시키는 기대치 관련)
  γ = 회복율 (1/γ = 평균 감염 기간)
  기초감염재생산수  R0 = β/γ   (한 감염자가 평균 몇 명 감염시키나)
  집단면역 임계치   = 1 - 1/R0  (이만큼 면역되면 유행이 꺾임)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 모든 결과 png 는 result/ 폴더에 저장한다
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------- 파라미터 (여기서 조절) ----------------
N = 1000.0        # 전체 인구
I0 = 1.0          # 초기 감염자
R0_INIT = 0.0     # 초기 회복자
BETA = 0.30       # 감염 전파율 β
GAMMA = 0.10      # 회복율 γ   (평균 감염기간 = 1/γ = 10일)
DAYS = 160        # 시뮬레이션 기간(일)
DT = 0.1          # 적분 시간 간격


# ---------------- 모델 ----------------
def deriv(y):
    S, I, R = y
    dS = -BETA * S * I / N
    dI = BETA * S * I / N - GAMMA * I
    dR = GAMMA * I
    return np.array([dS, dI, dR])


def rk4_step(y, dt):
    k1 = deriv(y)
    k2 = deriv(y + dt / 2 * k1)
    k3 = deriv(y + dt / 2 * k2)
    k4 = deriv(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate():
    steps = int(DAYS / DT)
    t = np.linspace(0, DAYS, steps + 1)
    Y = np.zeros((steps + 1, 3))
    Y[0] = [N - I0 - R0_INIT, I0, R0_INIT]
    for i in range(steps):
        Y[i + 1] = rk4_step(Y[i], DT)
    return t, Y[:, 0], Y[:, 1], Y[:, 2]


t, S, I, R = simulate()

# ---------------- 요약 수치 ----------------
R0 = BETA / GAMMA
herd = 1 - 1 / R0
peak_idx = np.argmax(I)
peak_day, peak_I = t[peak_idx], I[peak_idx]
final_infected = R[-1]            # 최종 감염 경험자(=회복자) 수

print(f"기초감염재생산수 R0 = β/γ = {R0:.2f}")
print(f"집단면역 임계치 = 1 - 1/R0 = {herd*100:.1f}%")
print(f"감염 정점: {peak_day:.1f}일차, 동시감염자 {peak_I:.0f}명 ({peak_I/N*100:.1f}%)")
print(f"최종 누적 감염 비율 = {final_infected/N*100:.1f}%")

# ---------------- 그리기 ----------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t, S, color="#2f6fd0", lw=2.2, label="S Susceptible")
ax.plot(t, I, color="#d1352c", lw=2.2, label="I Infected")
ax.plot(t, R, color="#2e8b57", lw=2.2, label="R Recovered")
ax.fill_between(t, 0, I, color="#d1352c", alpha=0.12)

# 감염 정점 표시
ax.plot(peak_day, peak_I, "o", color="#d1352c", ms=7, zorder=5)
ax.annotate(f"Peak day {peak_day:.0f}\n{peak_I:.0f} people",
            xy=(peak_day, peak_I), xytext=(peak_day + 12, peak_I + 60),
            fontsize=10, color="#d1352c",
            arrowprops=dict(arrowstyle="->", color="#d1352c"))

# 집단면역 임계선 (감염가능 인구가 N/R0 아래로 내려가면 유행 꺾임)
ax.axhline(N / R0, color="0.5", ls="--", lw=1,
           label=f"Herd immunity S=N/R0={N/R0:.0f}")

ax.set_title(f"SIR Model  (R0={R0:.1f},  β={BETA},  γ={GAMMA},  Population {N:.0f})", fontsize=13)
ax.set_xlabel("Time (days)")
ax.set_ylabel("Number of people")
ax.set_xlim(0, DAYS)
ax.set_ylim(0, N * 1.02)
ax.legend(fontsize=10, loc="center right")
ax.grid(True, alpha=0.25)

plt.tight_layout()
_out = os.path.join(RESULT_DIR, "sir_model.png")
plt.savefig(_out, dpi=140)
plt.close(fig)
print(f"저장: {_out}")
