"""
열방정식(heat equation)을 유도하는 데 쓰이는 '열의 성질' 그림.

수식 대신, 유도에 들어가는 물리적 성질 다섯 가지를 네 개의 그림으로 보여준다.

  (A) 열은 뜨거운 쪽 → 차가운 쪽으로 흐른다.
  (B) 온도 기울기가 가파를수록 열이 세게 흐른다 (푸리에 법칙).
  (C) 들어온 열 − 나간 열 = 토막에 쌓이는 열 (에너지 보존).
  (D) 쌓인 열만큼 온도가 오른다 (열용량) / 온도가 고르면 흐름이 없다.

사용법:
    python heat_properties.py                       # 화면에 표시
    python heat_properties.py --save heat_props.png # PNG 로 저장
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 한글 표기를 위한 폰트 (macOS 기본 제공)
rcParams["font.family"] = "AppleGothic"
rcParams["axes.unicode_minus"] = False


def rod_gradient(ax, temp_left=1.0, temp_right=0.0, cmap="inferno"):
    """축 전체를 온도 그라데이션 막대(rod)로 칠한다. 왼쪽이 뜨겁다."""
    grad = np.linspace(temp_left, temp_right, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 10, 0, 1],
              vmin=0, vmax=1)


def panel_A(ax):
    """(A) 열은 뜨거운 쪽에서 차가운 쪽으로 흐른다."""
    rod_gradient(ax)
    # 흐름 화살표: 모두 뜨거운(왼) → 차가운(오)
    for xc in [2, 4, 6, 8]:
        ax.annotate("", xy=(xc + 0.9, 0.5), xytext=(xc - 0.9, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="white"))
    ax.text(0.4, 1.18, "뜨겁다", color="firebrick", fontsize=11, ha="left",
            fontweight="bold")
    ax.text(9.6, 1.18, "차갑다", color="steelblue", fontsize=11, ha="right",
            fontweight="bold")
    ax.set_title("(A) 열은 뜨거운 쪽 → 차가운 쪽으로만 흐른다", fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])


def panel_B(ax):
    """(B) 온도 기울기가 가파를수록 열이 세게 흐른다 (푸리에 법칙)."""
    x = np.linspace(0, 10, 400)
    # 왼쪽은 가파르게, 오른쪽은 완만하게 떨어지는 온도 곡선
    T = 1.0 / (1 + np.exp((x - 3.2) * 1.6))          # 가파른 계단 → 완만
    ax.plot(x, T, color="0.2", lw=2.6)
    ax.fill_between(x, T, color="orange", alpha=0.15)

    # 몇 지점에서 국소 기울기에 비례한 흐름 화살표(아래로 = 흐름 세기)
    for xc in [1.5, 3.2, 5.0, 7.5]:
        i = np.argmin(np.abs(x - xc))
        slope = -(T[i + 1] - T[i - 1]) / (x[i + 1] - x[i - 1])   # −dT/dx > 0
        L = 0.15 + 2.8 * slope                        # 화살표 길이 ∝ 기울기
        ax.annotate("", xy=(xc, T[i] - L), xytext=(xc, T[i]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color="crimson"))
    ax.text(3.2, 1.02, "가파른 곳\n→ 굵은 흐름", color="crimson", fontsize=10,
            ha="center", va="bottom")
    ax.text(7.5, 0.28, "완만한 곳\n→ 약한 흐름", color="crimson", fontsize=10,
            ha="center", va="top")
    ax.set_title("(B) 온도 기울기가 급할수록 열이 세게 흐른다", fontsize=12)
    ax.set_xlabel("위치 x")
    ax.set_ylabel("온도 T")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.55, 1.35)
    ax.grid(True, alpha=0.15)


def panel_C(ax):
    """(C) 들어온 열 − 나간 열 = 토막에 쌓이는 열 (에너지 보존)."""
    rod_gradient(ax, cmap="inferno")
    # 관심 토막 [4, 6] 강조
    x0, x1 = 4, 6
    ax.axvspan(x0, x1, color="white", alpha=0.0)
    for xv in (x0, x1):
        ax.plot([xv, xv], [0, 1], color="white", lw=1.5, ls="--")
    ax.add_patch(plt.Rectangle((x0, 0), x1 - x0, 1, fill=False,
                               edgecolor="white", lw=2.2))

    # 왼쪽 면: 많이 들어옴(굵은 화살표) / 오른쪽 면: 적게 나감(가는 화살표)
    ax.annotate("", xy=(x0 + 0.7, 0.5), xytext=(x0 - 1.3, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=4.0, color="white"))
    ax.annotate("", xy=(x1 + 1.0, 0.5), xytext=(x1 - 0.3, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="white"))
    ax.text(x0 - 1.3, 1.16, "많이 들어옴", color="firebrick", fontsize=10,
            ha="left", fontweight="bold")
    ax.text(x1 + 1.0, 1.16, "적게 나감", color="steelblue", fontsize=10,
            ha="right", fontweight="bold")
    ax.text(5, -0.22, "차이만큼 이 토막에 열이 쌓인다", color="0.15",
            fontsize=11, ha="center")
    ax.set_title("(C) 들어온 열 - 나간 열 = 쌓이는 열 (보존)", fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 1.35)
    ax.set_yticks([])
    ax.set_xticks([])


def panel_D(ax):
    """(D) 쌓인 열만큼 온도가 오른다(열용량) / 고르면 흐름이 없다."""
    # 위: 쌓인 열 ↑ → 온도 눈금 ↑ (막대 그래프로 비례 표현)
    heat = np.array([1, 2, 3, 4])
    temp = 0.6 * heat                                 # 온도 상승 ∝ 쌓인 열
    xs = np.arange(4)
    ax.bar(xs - 0.18, heat, width=0.32, color="orange", label="쌓인 열")
    ax.bar(xs + 0.18, temp, width=0.32, color="firebrick",
           label="온도 상승")
    ax.plot(xs, heat, "--", color="0.5", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["토막1", "토막2", "토막3", "토막4"])
    ax.legend(loc="upper left", fontsize=9)
    ax.text(1.5, -0.9, "고른 온도(기울기 0)면 흐름도 0 → 평형 유지",
            color="0.15", fontsize=10, ha="center")
    ax.set_title("(D) 쌓인 열 ∝ 온도 상승 (열용량)", fontsize=12)
    ax.set_ylabel("크기 (임의 단위)")
    ax.set_ylim(-1.2, 4.6)
    ax.grid(True, axis="y", alpha=0.15)


def main():
    parser = argparse.ArgumentParser(
        description="열방정식 유도에 쓰이는 열의 성질 그림")
    parser.add_argument("--save", metavar="OUT.png", default=None,
                        help="화면 대신 PNG 로 저장")
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), facecolor="white")
    fig.suptitle("열방정식을 유도하는 열의 성질", fontsize=16, fontweight="bold",
                 y=0.98)

    panel_A(axes[0][0])
    panel_B(axes[0][1])
    panel_C(axes[1][0])
    panel_D(axes[1][1])

    fig.subplots_adjust(top=0.9, bottom=0.07, left=0.06, right=0.97,
                        hspace=0.32, wspace=0.18)

    if args.save:
        fig.savefig(args.save, dpi=130, facecolor="white")
        print(f"이미지 저장 완료: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
