"""
왜 (N-1)개 어두운 점, (N-2)개 부극대인가? — 화살표(phasor) 그림으로 설명

광원 N개의 파동을 화살표 N개로 본다.
이웃한 광원끼리는 위상차 φ = 2π d sinθ / λ 만큼 어긋나므로,
화살표들이 매번 같은 각도 φ 만큼 회전하면서 머리-꼬리로 이어진다.

- 합(밝기) = 첫 화살표 꼬리 -> 마지막 화살표 머리 까지의 '직선 거리'
- φ=0 : 전부 한 방향 -> 일직선, 최대 길이 N (주극대)
- 화살표들이 이어져 '닫힌 다각형'을 이루면 시작=끝 -> 합 0 (어두운 점)

N=4 로, 각도 θ 를 조금씩 키우며(=φ 를 키우며) 화살표 사슬이
어떻게 감기는지 보여준다.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

N = 4  # 광원 개수

# 보여줄 위상차 φ 들 (도 단위) 과 설명
cases = [
    (0,   "φ=0  : 일직선\n주극대 (최대 밝기)"),
    (90,  "φ=90°: 정사각형 닫힘\n합=0  → 어두운 점 ①"),
    (135, "φ=135°: 살짝 열린 다각형\n작은 봉우리 (부극대)"),
    (180, "φ=180°: 다시 닫힘\n합=0  → 어두운 점 ②"),
    (225, "φ=225°: 살짝 열림\n작은 봉우리 (부극대)"),
    (270, "φ=270°: 또 닫힘\n합=0  → 어두운 점 ③"),
]

fig, axes = plt.subplots(1, len(cases), figsize=(16, 3.6))

for ax, (deg, label) in zip(axes, cases):
    phi = np.deg2rad(deg)
    # 화살표 N개: 각도 0, φ, 2φ, 3φ ...  누적으로 이어붙임
    angles = phi * np.arange(N)
    steps = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    pts = np.vstack([[0, 0], np.cumsum(steps, axis=0)])  # 꼭짓점들

    # 화살표 사슬
    for i in range(N):
        ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                    arrowprops=dict(arrowstyle="->", color="navy", lw=1.8))

    # 합(시작->끝) = 밝기
    resultant = np.hypot(*pts[-1])
    ax.annotate("", xy=pts[-1], xytext=pts[0],
                arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))

    ax.plot(*pts[0], "ko", ms=5)
    ax.set_title(label, fontsize=9)
    ax.text(0.5, -0.16, f"합(빨강) 길이 = {resultant:.2f}",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="crimson", weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(
    f"광원 {N}개의 화살표(phasor) 사슬 — 파랑=각 파동, 빨강=합(=밝기)\n"
    "각도가 커질수록 사슬이 감긴다: 닫히면(다각형) 밝기 0, 살짝 열리면 작은 봉우리",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("phasor_why.png", dpi=140)
plt.show()
print("저장 완료: phasor_why.png")
