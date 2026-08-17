"""
폭 있는 광원이 여러 개일 때 (실제 회절격자)

먼 거리(Fraunhofer) 극한에서 '폭 w 인 슬릿 N개(간격 d)'의 세기는 정확히:

    I = [ sin(N·φ/2) / (N·sin(φ/2)) ]^2   x   [ sin(β/2) / (β/2) ]^2
         └──── 간섭항 (슬릿 N개) ────┘        └── 회절 포락선 (폭 w) ──┘

    φ = 2π d sinθ / λ   (이웃 슬릿 중심 사이 위상차)
    β = 2π w sinθ / λ   (한 슬릿 양 끝 사이 위상차)

가로축을 m = φ/2π (주극대 번호) 로 두면:
    - 간섭 주극대는 m = 0,1,2,... 정수에 위치
    - 회절 포락선은 sinc(m·w/d)^2  → m = d/w 에서 첫 0
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

d = 5.0
w = 1.5                 # 슬릿 폭 (w/d = 0.3)
ratio = w / d

m = np.linspace(-3.5, 3.5, 60000)
phi = 2 * np.pi * m


def envelope():
    # 회절 포락선 sinc^2 (폭 w) — np.sinc(x)=sin(pi x)/(pi x)
    return np.sinc(m * ratio) ** 2


def intensity(N):
    num = np.sin(N * phi / 2)
    den = np.sin(phi / 2)
    grating = np.where(np.abs(den) < 1e-9, N**2, (num / den) ** 2) / N**2
    return grating * envelope()


Ns = [2, 3, 4]

fig, axes = plt.subplots(len(Ns), 1, figsize=(11, 8), sharex=True)

for ax, N in zip(axes, Ns):
    I = intensity(N)
    env = envelope()
    ax.plot(m, I, color="navy", lw=1.3)
    ax.fill_between(m, 0, I, color="navy", alpha=0.15)
    ax.plot(m, env, "--", color="crimson", lw=1.4, alpha=0.8,
            label="회절 포락선 sinc² (폭 w)")

    ax.set_title(f"폭 있는 광원 {N}개  (간격 d={d}, 폭 w={w})", fontsize=11, loc="left")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("밝기")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")

axes[-1].set_xlabel("주극대 번호 m   (d·sinθ = m·λ)")
plt.suptitle(
    "폭 있는 광원 여러 개 = (슬릿 간섭무늬) × (폭에 의한 sinc² 포락선)\n"
    f"포락선 첫 0 은 m = d/w = {d/w:.2f}  → 그 근처 무늬는 눌려 사라짐",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("width_multi.png", dpi=140)
plt.show()
print("저장 완료: width_multi.png")
