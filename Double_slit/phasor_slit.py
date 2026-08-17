"""
폭 있는 광원(단일 슬릿)을 화살표 그림으로 설명

폭 있는 슬릿 = 점광원 '무한 개'가 촘촘히 붙은 것.
-> 화살표 무한 개가 아주 조금씩 회전하며 이어짐 -> 다각형이 아니라 '매끄러운 호(arc)'.

슬릿의 위쪽 끝 조각과 아래쪽 끝 조각의 위상차를 Φ = 2π w sinθ / λ 라 하면,
화살표 사슬(호)이 전체적으로 각도 Φ 만큼 감긴다.

- 호의 '길이'는 항상 일정 (조각 개수/세기 고정)
- 밝기 = 시작점 -> 끝점 '직선 거리(활시위, chord)' = 빨간 화살표
- Φ=0   : 일직선, 최대 밝기 (정면)
- Φ=2π  : 호가 완전한 원 -> 시작=끝 -> 밝기 0 (첫 어두운 점)
- Φ=3π  : 1.5바퀴 -> 작은 활시위 (약한 부극대)
- Φ=4π  : 2바퀴 -> 다시 밝기 0
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

M = 400  # 슬릿을 쪼갠 조각 수 (많을수록 매끄러운 호)

cases = [
    (0.0,      "Φ=0 : 일직선\n최대 밝기 (정면 θ=0)"),
    (np.pi,    "Φ=π : 반원\n아직 밝음 (봉우리 중턱)"),
    (2*np.pi,  "Φ=2π : 완전한 원\n밝기 0 → 첫 어두운 점"),
    (3*np.pi,  "Φ=3π : 1.5바퀴\n작은 활시위 → 부극대"),
    (4*np.pi,  "Φ=4π : 2바퀴\n밝기 0 → 둘째 어두운 점"),
]

fig, axes = plt.subplots(1, len(cases), figsize=(15, 3.6))

for ax, (Phi, label) in zip(axes, cases):
    # 조각 M개, 각 조각의 위상은 0 ~ Φ 로 고르게 증가
    ang = np.linspace(0, Phi, M)
    steps = np.stack([np.cos(ang), np.sin(ang)], axis=1) / M  # 전체 길이 1로 정규화
    pts = np.vstack([[0, 0], np.cumsum(steps, axis=0)])

    # 매끄러운 호 (파란 곡선)
    ax.plot(pts[:, 0], pts[:, 1], color="navy", lw=2.2)

    # 밝기 = 시작 -> 끝 직선 거리 (빨간 활시위)
    chord = np.hypot(*(pts[-1] - pts[0]))
    ax.annotate("", xy=pts[-1], xytext=pts[0],
                arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5))
    ax.plot(*pts[0], "ko", ms=5)
    ax.plot(*pts[-1], "o", color="navy", ms=5)

    ax.set_title(label, fontsize=9)
    ax.text(0.5, -0.16, f"밝기(빨강 길이) = {chord:.2f}",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="crimson", weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-0.45, 0.75)
    ax.set_ylim(-0.55, 0.55)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle(
    "폭 있는 광원 = 화살표 무한개 → 매끄러운 '호'.  파랑=호(모든 조각), 빨강=밝기(활시위)\n"
    "각도 θ가 커지면 호가 더 감긴다: 완전한 원이 되면 밝기 0. (활시위/호 = sin/각도 = sinc)",
    fontsize=11,
)
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig("phasor_slit.png", dpi=140)
plt.show()
print("저장 완료: phasor_slit.png")
