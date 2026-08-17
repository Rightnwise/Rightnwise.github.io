"""
두 점광원(point source) 간섭 시뮬레이션 (Young의 이중슬릿)

- 두 광원은 y축 위에 거리 d 만큼 떨어져 있고, 각각 cos(wt) 파동을 낸다.
- 두 광원의 중점 (0,0) 에서 거리 R 만큼 떨어진 곳에 스크린이 있다.
- 스크린의 한 점을 바라보는 각도를 theta 라 하면,
  두 광원에서 그 점까지의 경로차 = d * sin(theta)
  위상차 delta = 2*pi*d*sin(theta) / lambda
- 두 파동의 합성 진폭은
  cos(wt) + cos(wt + delta) = 2*cos(delta/2)*cos(wt + delta/2)
  이므로 진폭 ~ 2*cos(delta/2), 밝기(세기) I ~ (2*cos(delta/2))^2
"""

import numpy as np
import matplotlib.pyplot as plt

# macOS 한글 폰트 (깨짐 방지)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 파라미터 ----------------
wavelength = 0.5      # 파장 lambda (임의 단위)
d = 5       # 두 광원 사이 거리
R = 100.0             # 중점에서 스크린까지 거리
screen_height = 120.0 # 스크린 세로 길이 (위 ~ 아래)
N = 2000              # 스크린 위 샘플 점 개수

k = 2 * np.pi / wavelength

# ---------------- 스크린 위 좌표 ----------------
# 스크린은 x = R 위치에 세로로 세워져 있다. y 는 위(+) 에서 아래(-) 까지.
y = np.linspace(screen_height / 2, -screen_height / 2, N)

# 각 점을 바라보는 각도 theta (x축과 R 사이 각)
theta = np.arctan2(y, R)

# ---------------- 위상차와 세기 ----------------
path_diff = d * np.sin(theta)          # 경로차 = d sin(theta)
delta = k * path_diff                   # 위상차 = 2*pi*d*sin(theta)/lambda

# 합성 진폭 ~ 2 cos(delta/2), 밝기 I ~ amplitude^2 (최대값 1 로 정규화)
intensity = np.cos(delta / 2) ** 2

# ---------------- 그리기 ----------------
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(9, 6), sharex=True,
    gridspec_kw={"height_ratios": [1, 3]}
)

# (1) 스크린에 맺히는 밝기 무늬 (가로 띠 이미지)
strip = np.tile(intensity[None, :], (40, 1))  # 세로로 살짝 늘려서 띠로 표시
ax1.imshow(
    strip, cmap="inferno", aspect="auto",
    extent=[y.min(), y.max(), 0, 1],
    vmin=0, vmax=1,
)
ax1.set_title("스크린 밝기 무늬")
ax1.set_yticks([])

# (2) 위치에 따른 밝기(세기) 곡선  — x축: 스크린 위치, y축: 밝기
ax2.plot(y, intensity, color="crimson")
ax2.fill_between(y, 0, intensity, color="crimson", alpha=0.3)
ax2.set_title("위치에 따른 세기  I ~ cos²(δ/2)")
ax2.set_xlabel("스크린 위치 y")
ax2.set_ylabel("상대 밝기")
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    f"두 점광원 간섭  (d={d}, λ={wavelength}, R={R})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("double_slit.png", dpi=130)
plt.show()
print("저장 완료: double_slit.png")
