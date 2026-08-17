"""
실제 plucked string 사진 위에 삼각형/치수/유도과정을 겹쳐 그린다.
가로축 전체 = L,  왼쪽 근처(d)에서 높이 h 로 뜯음.

주어진 두 기울기:  왼쪽 5/3,  오른쪽 5/27
    (5/3) d = (5/27)(L - d)  ->  9d = L - d  ->  d = L/10
    h = (5/3)(L/10) = L/6
"""

import os
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
# 파일명에 특수 공백이 있을 수 있어 패턴으로 찾는다
_matches = glob.glob(os.path.join(HERE, "Screenshot*.png"))
IMG = _matches[0] if _matches else os.path.join(HERE, "pluck.png")
OUT = os.path.join(HERE, "pluck_geometry.png")


def main():
    img = mpimg.imread(IMG)

    # 사진 속 주요 지점 (원본 픽셀 좌표, 대략)
    xL, yL = 65, 610       # 왼쪽 고정단 (줄이 묶인 곳)
    xP, yP = 295, 465      # 뜯은 지점 (손) = 삼각형 꼭짓점
    xR, yR = 2240, 615     # 오른쪽 고정단

    fig, ax = plt.subplots(figsize=(14, 5.8))
    ax.imshow(img)

    # 이상화한 뜯은 줄(삼각형)
    ax.plot([xL, xP, xR], [yL, yP, yR], color="gold", lw=2.2, zorder=3)
    ax.plot([xL, xP, xR], [yL, yP, yR], "o", color="gold", ms=8, zorder=4)

    # 평형선(안 뜯겼을 때)
    ax.plot([xL, xR], [yL, yR], "--", color="white", lw=1, alpha=0.6, zorder=2)

    # d 치수 (아래쪽 가로 화살표)
    ybar = yL + 130
    ax.annotate("", xy=(xP, ybar), xytext=(xL, ybar),
                arrowprops=dict(arrowstyle="<->", color="gold", lw=1.8))
    ax.text((xL + xP) / 2, ybar + 55, "d ≈ L/10", ha="center",
            color="gold", fontsize=14, fontweight="bold")

    # h 치수 (세로 화살표, 꼭짓점에서)
    ax.annotate("", xy=(xP, yP), xytext=(xP, yL),
                arrowprops=dict(arrowstyle="<->", color="gold", lw=1.8))
    ax.text(xP + 25, (yP + yL) / 2, "h = L/6", color="gold",
            fontsize=14, fontweight="bold", va="center")

    # 기울기 라벨
    ax.text((xL + xP) / 2 - 40, (yL + yP) / 2 - 45, "slope 5/3",
            color="white", fontsize=12)
    ax.text((xP + xR) / 2, (yP + yR) / 2 - 55, "slope 5/27",
            color="white", fontsize=12, ha="center")

    # 축: 전체 = L
    ax.set_xticks([xL, xP, xR])
    ax.set_xticklabels(["0", "d", "L"], fontsize=12)
    ax.set_yticks([])
    ax.set_xlabel("x   (full span = L)", fontsize=12)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("저장: pluck_geometry.png")
    plt.show()


if __name__ == "__main__":
    main()
