"""
점광원 N개 규칙을 '이상적인 먼 거리' 공식으로 깨끗하게 보기

먼 거리(Fraunhofer) 극한에서 N개 슬릿(간격 d)의 세기는 정확히:

    I(φ) = [ sin(N φ / 2) / sin(φ / 2) ]^2  / N^2       (0~1 로 정규화)

    여기서 φ = 2π d sinθ / λ  (이웃 슬릿 사이 위상차)

- φ = 0, 2π, 4π ... (즉 이웃 위상차가 파장의 정수배) 일 때 주극대 (밝기 1)
- 그 사이에 완전히 어두운 점(0)이 (N-1)개, 약한 부극대가 (N-2)개.

가로축을 φ/2π (= 몇 번째 주극대까지 왔나) 로 두면 주극대가 0,1,2... 정수에 딱 온다.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# 가로축: m = φ/2π  (주극대가 정수 0,1,2 에 위치)
m = np.linspace(-0.2, 2.2, 40000)
phi = 2 * np.pi * m


def intensity(N):
    num = np.sin(N * phi / 2)
    den = np.sin(phi / 2)
    # den=0 인 지점(주극대)은 극한값 N -> 세기 N^2
    I = np.where(np.abs(den) < 1e-9, N**2, (num / den) ** 2)
    return I / N**2


Ns = [2, 3, 4, 5]

fig, axes = plt.subplots(len(Ns), 1, figsize=(11, 9), sharex=True)

for ax, N in zip(axes, Ns):
    I = intensity(N)
    ax.plot(m, I, color="navy", lw=1.6)
    ax.fill_between(m, 0, I, color="navy", alpha=0.15)

    # 주극대 위치 (m = 0,1,2) 세로선
    for mm in [0, 1, 2]:
        ax.axvline(mm, color="crimson", ls=":", lw=1.2, alpha=0.7)

    # 부극대 개수 표시
    n_sub = N - 2
    n_zero = N - 1
    ax.set_title(
        f"점광원 {N}개  →  큰 봉우리 사이:  어두운 점 {n_zero}개, 작은 봉우리 {n_sub}개",
        fontsize=11, loc="left",
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("밝기")
    ax.grid(True, alpha=0.25)

axes[0].annotate("주극대\n(가장 밝음)", xy=(1, 1.0), xytext=(1.15, 0.55),
                 fontsize=9, color="crimson",
                 arrowprops=dict(arrowstyle="->", color="crimson"))
axes[-1].set_xlabel("주극대 번호 m  (빨간 점선 = 주극대,  d·sinθ = m·λ)")
plt.suptitle("이상적 회절격자: 주극대 사이 부극대 = (N−2)개, 어두운 점 = (N−1)개",
             fontsize=13)
plt.tight_layout()
plt.savefig("grating_rule.png", dpi=140)
plt.show()
print("저장 완료: grating_rule.png")
