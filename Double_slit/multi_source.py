"""
광원 개수(1개 / 3개)와 폭(점 / 유한폭)에 따른 무늬 비교

- 광원이 N개, 간격 d, 각 광원의 폭 w (w=0 이면 점광원).
- 각 광원을 여러 점으로 쪼개 같은 위상(coherent)으로 출발시키고,
  스크린 위 각 점에서 모든 점광원의 파동을 복소수로 더해 |합|^2 = 세기.

관찰 포인트
  * 점광원 1개  : 간섭 상대가 없어 밝기 거의 균일 (무늬 없음)
  * 점광원 3개  : 다중슬릿 간섭 -> 강한 주극대 + 사이의 약한 부극대
  * 폭 1개      : 단일슬릿 회절 -> sinc^2 모양 (가운데 밝고 양옆 작은 봉우리)
  * 폭 3개      : (3개 간섭무늬) x (폭에 의한 sinc^2 포락선)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 파라미터 ----------------
wavelength = 0.5
d = 5.0                # 광원 사이 간격
R = 100.0              # 광원면 -> 스크린 거리
screen_height = 120.0
N_screen = 1500
M_src = 151            # 폭이 있는 광원 하나를 쪼갤 점 개수

k = 2 * np.pi / wavelength
ys = np.linspace(-screen_height / 2, screen_height / 2, N_screen)


def source_positions(N, w):
    """N개 광원(간격 d, 폭 w)을 이루는 모든 점광원의 y좌표 배열."""
    centers = d * (np.arange(N) - (N - 1) / 2)      # 0 중심으로 대칭 배치
    if w == 0:
        return centers
    half = np.linspace(-w / 2, w / 2, M_src)
    return np.concatenate([c + half for c in centers])


def simulate(N, w):
    src = source_positions(N, w)
    dist = np.sqrt(R**2 + (ys[:, None] - src[None, :]) ** 2)
    field = np.exp(1j * k * dist).sum(axis=1)       # coherent 합
    I = np.abs(field) ** 2
    return I / I.max()


# ---------------- 4가지 경우 ----------------
cases = [
    (1, 0.0, "점광원 1개",            "gray"),
    (3, 0.0, "점광원 3개 (간격 d=5)", "seagreen"),
    (1, 2.5, "폭 있는 광원 1개 (w=2.5)",       "crimson"),
    (3, 2.5, "폭 있는 광원 3개 (d=5, w=2.5)",  "navy"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

for ax, (N, w, title, color) in zip(axes.flat, cases):
    I = simulate(N, w)
    ax.plot(ys, I, color=color, lw=1.5)
    ax.fill_between(ys, 0, I, color=color, alpha=0.2)

    # 폭이 있으면 회절 포락선(sinc^2)도 점선으로 표시
    if w > 0:
        theta = np.arctan2(ys, R)
        env = np.sinc(w * np.sin(theta) / wavelength) ** 2
        ax.plot(ys, env, "--", color="black", lw=1, alpha=0.6, label="회절 포락선 sinc²")
        ax.legend(fontsize=8, loc="upper right")

    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

axes[0, 0].set_ylabel("상대 밝기")
axes[1, 0].set_ylabel("상대 밝기")
axes[1, 0].set_xlabel("스크린 위치 y")
axes[1, 1].set_xlabel("스크린 위치 y")

plt.suptitle(
    f"광원 개수 · 폭에 따른 무늬 비교  (d={d}, λ={wavelength}, R={R})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("multi_source.png", dpi=130)
plt.show()
print("저장 완료: multi_source.png")
