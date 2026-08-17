"""
이중슬릿 간섭 : 슬릿이 '점(point)'이 아니라 '유한한 폭 w'를 가질 때

이전 코드는 두 슬릿을 각각 점광원으로 봤다.
여기서는 각 슬릿이 세로로 길이(폭) w 를 가진다고 보고,
그 폭 안을 여러 개의 점광원으로 잘게 쪼개서 (coherent, 같은 위상으로 출발)
스크린 위 각 점에서 모든 점광원의 파동을 복소수로 더한 뒤 |합|^2 = 세기 를 구한다.

결과:
    전체 무늬 = (두 슬릿 사이 간섭무늬)  x  (슬릿 폭에 의한 단일슬릿 회절 포락선)

이론값과 비교:
    간섭 항   :  cos^2( pi d sinθ / λ )
    회절 포락선:  sinc^2( pi w sinθ / λ ),   sinc(x)=sin(x)/x
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 파라미터 ----------------
wavelength = 0.5      # 파장 lambda
d = 5.0               # 두 슬릿 중심 사이 거리
R = 100.0             # 슬릿면 -> 스크린 거리
screen_height = 120.0 # 스크린 세로 길이
N_screen = 1500       # 스크린 샘플 점 개수
M_src = 151           # 슬릿 하나를 쪼갤 점광원 개수

k = 2 * np.pi / wavelength

# 스크린 위 좌표 (x = R 평면)
ys = np.linspace(-screen_height / 2, screen_height / 2, N_screen)


def simulate(w):
    """슬릿 폭 w 일 때 스크린 세기를 수치적분으로 계산."""
    # 두 슬릿: 중심 +d/2, -d/2, 각각 폭 w 를 M_src 개 점으로 채운다
    if w == 0:
        src = np.array([-d / 2, d / 2])          # 점광원 2개
    else:
        half = np.linspace(-w / 2, w / 2, M_src)
        src = np.concatenate([-d / 2 + half, d / 2 + half])

    # 각 스크린점(row) 에서 각 광원(col) 까지의 거리  -> 브로드캐스팅
    # 광원은 x=0, y=src ;  스크린점은 x=R, y=ys
    dist = np.sqrt(R**2 + (ys[:, None] - src[None, :]) ** 2)

    # 같은 위상으로 출발한 파동을 복소수로 더함 (coherent 합)
    field = np.exp(1j * k * dist).sum(axis=1)
    intensity = np.abs(field) ** 2
    return intensity / intensity.max()           # 최대 1 로 정규화


def theory(w):
    """이론식: 간섭항 x 회절포락선 (비교용)."""
    theta = np.arctan2(ys, R)
    s = np.sin(theta)
    interf = np.cos(np.pi * d * s / wavelength) ** 2
    if w == 0:
        env = np.ones_like(s)
    else:
        arg = np.pi * w * s / wavelength
        env = np.sinc(arg / np.pi) ** 2          # np.sinc(x)=sin(pi x)/(pi x)
    I = interf * env
    return I / I.max()


# ---------------- 여러 슬릿 폭 비교 ----------------
widths = [0.0, 1.0, 2.5]     # 0 = 점광원, 그 외 = 유한 폭
colors = ["gray", "crimson", "navy"]

fig, axes = plt.subplots(len(widths), 1, figsize=(10, 8), sharex=True)

for ax, w, c in zip(axes, widths, colors):
    I = simulate(w)
    ax.plot(ys, I, color=c, lw=1.5, label=f"시뮬레이션 (수치적분)")
    ax.fill_between(ys, 0, I, color=c, alpha=0.2)

    # 회절 포락선(이론)도 점선으로 표시
    if w > 0:
        theta = np.arctan2(ys, R)
        env = np.sinc(w * np.sin(theta) / wavelength) ** 2
        ax.plot(ys, env, "--", color="black", lw=1, alpha=0.7, label="회절 포락선 sinc²")

    title = "슬릿 폭 w = 0  (점광원)" if w == 0 else f"슬릿 폭 w = {w}"
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("상대 밝기")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

axes[-1].set_xlabel("스크린 위치 y")
plt.suptitle(
    f"슬릿 폭에 따른 이중슬릿 무늬  (d={d}, λ={wavelength}, R={R})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("finite_source.png", dpi=130)
plt.show()
print("저장 완료: finite_source.png")
