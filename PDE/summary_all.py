"""
================================================================
이중슬릿 / 회절격자 종합 정리 : 세 요소는 각각 '독립적'이다
================================================================

먼 거리(Fraunhofer) 극한에서, 폭 w 인 슬릿 N개(간격 d)의 밝기는
정확히 두 인자의 '곱'으로 나뉜다:

    I(θ) = | sin(N·φ/2) / (N·sin(φ/2)) |^2   ×   | sinc(β/2) |^2
            └──── 간섭 인자 : N, d 담당 ────┘        └ 회절 인자 : w 담당 ┘

    φ = 2π d sinθ / λ   (이웃 슬릿 사이 위상차)
    β = 2π w sinθ / λ   (한 슬릿 양 끝 사이 위상차)

이 '곱으로 나뉜다'는 사실이 곧 독립성의 증거다:
    * N (광원 개수)  → 봉우리의 '날카로움' + 부극대 개수(N-2).   위치·포락선은 안 건드림.
    * w (슬릿 폭)    → 회절 '포락선(덮개)'의 너비.               봉우리 위치·개수는 안 건드림.
    * d (광원 간격)  → 봉우리 사이 '간격'(∝ 1/d).                포락선은 안 건드림.

이 스크립트는 이 세 가지를 각각 따로 실험한 그림 + 분해 그림 + 요약표(치트시트)를
PNG 파일로 저장한다.

가로축은 u = sinθ (스크린 위치에 비례). λ 는 파장.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 공통 파라미터 ----------------
LAM = 0.5                     # 파장 λ
BASE = dict(N=4, d=8.0, w=2.0)  # 기준값 (한 요소만 바꿀 때 나머지는 이 값 고정)
U = np.linspace(-0.4, 0.4, 60000)   # u = sinθ

# 테마 색 (요소별)
COL = {"N": "#1f3b73", "w": "#1f7a4d", "d": "#c25a1a"}
ENVCOL = "#d12e5a"            # 회절 포락선 색 (항상 동일)


# ---------------- 물리 계산 ----------------
def envelope(w):
    """회절 포락선 sinc^2 (폭 w 만 관여).  np.sinc(x)=sin(πx)/(πx)."""
    return np.sinc(w * U / LAM) ** 2


def pattern(N, d, w):
    """전체 밝기 = 간섭 인자 × 회절 포락선."""
    phi = 2 * np.pi * d * U / LAM
    den = np.sin(phi / 2)
    grating = np.where(np.abs(den) < 1e-9, N**2, (np.sin(N * phi / 2) / den) ** 2) / N**2
    return grating * envelope(w)


def draw_panel(ax, N, d, w, color):
    """한 칸에 밝기(채운 곡선) + 회절 포락선(점선) 그리기."""
    I = pattern(N, d, w)
    ax.plot(U, I, color=color, lw=1.3)
    ax.fill_between(U, 0, I, color=color, alpha=0.18)
    ax.plot(U, envelope(w), "--", color=ENVCOL, lw=1.3, alpha=0.85)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(U.min(), U.max())
    ax.grid(True, alpha=0.22)


# ================================================================
# 1~3. 요소별 실험 : 한 요소만 바꾸고 나머지는 고정
# ================================================================
def sweep_figure(fname, key, values, takeaway):
    """key 요소만 values 로 바꿔가며 3칸으로 비교."""
    labels = {"N": "광원 개수 N", "w": "슬릿 폭 w", "d": "광원 간격 d"}
    fixed_txt = {
        "N": f"(d={BASE['d']:.0f}, w={BASE['w']:.0f} 고정)",
        "w": f"(N={BASE['N']}, d={BASE['d']:.0f} 고정)",
        "d": f"(N={BASE['N']}, w={BASE['w']:.0f} 고정)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharex=True, sharey=True)
    for ax, val in zip(axes, values):
        p = dict(BASE)
        p[key] = val
        draw_panel(ax, p["N"], p["d"], p["w"], COL[key])
        unit = "" if key == "N" else ""
        ax.set_title(f"{labels[key].split()[-1]} = {val:g}", fontsize=12)
    axes[0].set_ylabel("밝기")
    for ax in axes:
        ax.set_xlabel("u = sinθ  (스크린 위치에 비례)")
    # 포락선 범례
    axes[-1].plot([], [], "--", color=ENVCOL, label="회절 포락선 sinc²")
    axes[-1].legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"[ {labels[key]} 만 바꾼다 ]   {fixed_txt[key]}\n{takeaway}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


sweep_figure(
    "01_sweep_N.png", "N", [2, 4, 8],
    "→ 봉우리가 점점 '날카로워'지고 부극대(N-2개)가 늘 뿐,  "
    "봉우리 위치와 빨간 포락선은 그대로.",
)
sweep_figure(
    "02_sweep_w.png", "w", [1, 2, 4],
    "→ 빨간 '포락선(덮개)'만 좁아질 뿐,  "
    "봉우리 위치·개수·날카로움은 그대로.",
)
sweep_figure(
    "03_sweep_d.png", "d", [4, 8, 16],
    "→ 봉우리 '간격'만 촘촘해질 뿐(∝1/d),  "
    "빨간 포락선은 그대로.",
)


# ================================================================
# 4. 분해 그림 : 전체 = 간섭 인자 × 회절 포락선
# ================================================================
def decompose_figure(fname):
    N, d, w = BASE["N"], BASE["d"], BASE["w"]
    phi = 2 * np.pi * d * U / LAM
    den = np.sin(phi / 2)
    grating = np.where(np.abs(den) < 1e-9, N**2, (np.sin(N * phi / 2) / den) ** 2) / N**2
    env = envelope(w)
    total = grating * env

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    axes[0].plot(U, grating, color=COL["d"], lw=1.3)
    axes[0].fill_between(U, 0, grating, color=COL["d"], alpha=0.18)
    axes[0].set_title("① 간섭 인자  (광원 개수 N · 간격 d 가 결정)  "
                      "— 봉우리 위치=d, 날카로움=N", fontsize=11, loc="left")

    axes[1].plot(U, env, color=ENVCOL, lw=1.5)
    axes[1].fill_between(U, 0, env, color=ENVCOL, alpha=0.15)
    axes[1].set_title("② 회절 포락선  (슬릿 폭 w 가 결정)  "
                      "— 덮개 너비=1/w", fontsize=11, loc="left")

    axes[2].plot(U, total, color=COL["N"], lw=1.3)
    axes[2].fill_between(U, 0, total, color=COL["N"], alpha=0.18)
    axes[2].plot(U, env, "--", color=ENVCOL, lw=1.2, alpha=0.8)
    axes[2].set_title("③ 실제 밝기 = ① × ②  "
                      "(간섭 봉우리에 회절 덮개를 씌운 것)", fontsize=11, loc="left")

    for ax in axes:
        ax.set_ylim(0, 1.08)
        ax.set_xlim(U.min(), U.max())
        ax.set_ylabel("밝기")
        ax.grid(True, alpha=0.22)
    axes[-1].set_xlabel("u = sinθ  (스크린 위치에 비례)")

    fig.suptitle(
        f"밝기는 두 인자의 '곱'으로 나뉜다  (N={N}, d={d:.0f}, w={w:.0f}, λ={LAM})\n"
        "인자가 분리돼 있으니 세 요소는 서로 독립적이다",
        fontsize=13, y=1.0,
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


decompose_figure("04_decompose.png")


# ================================================================
# 5. 요약 치트시트 : 각 요소가 '무엇을 바꾸고 / 무엇을 안 바꾸는가'
# ================================================================
def cheatsheet_figure(fname):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    cards = [
        ("N", "광원 개수 N",
         ["봉우리 '날카로움'", "부극대 개수 = N-2", "많을수록 뾰족해짐"],
         ["봉우리 위치", "회절 포락선"]),
        ("w", "슬릿 폭 w",
         ["회절 '포락선(덮개)' 너비", "덮개 너비 ∝ 1/w", "넓을수록 덮개 좁아짐"],
         ["봉우리 위치", "봉우리 개수·날카로움"]),
        ("d", "광원 간격 d",
         ["봉우리 사이 '간격'", "간격 ∝ 1/d", "멀수록 무늬 촘촘해짐"],
         ["회절 포락선", "봉우리 날카로움"]),
    ]

    for ax, (key, title, controls, keeps) in zip(axes, cards):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # 색 카드 배경
        ax.add_patch(FancyBboxPatch(
            (0.03, 0.03), 0.94, 0.94,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=COL[key], alpha=0.10, edgecolor=COL[key], lw=2))
        # 제목
        ax.text(0.5, 0.9, title, ha="center", va="center",
                fontsize=15, weight="bold", color=COL[key])
        # 조절하는 것
        ax.text(0.08, 0.75, "✔ 바꾸는 것", fontsize=11, weight="bold",
                color=COL[key])
        for i, c in enumerate(controls):
            ax.text(0.12, 0.66 - i * 0.09, f"• {c}", fontsize=10.5, color="#222")
        # 안 바꾸는 것
        ax.text(0.08, 0.32, "�’ 안 바꾸는 것", fontsize=11, weight="bold",
                color="#888")
        for i, kkeep in enumerate(keeps):
            ax.text(0.12, 0.23 - i * 0.09, f"– {kkeep}", fontsize=10.5, color="#777")

    fig.suptitle(
        "세 손잡이는 서로 독립  —  각자 다른 것 하나씩만 조절한다\n"
        "밝기 = |sin(Nφ/2)/(N·sin(φ/2))|²  ×  |sinc(β/2)|²    "
        "(φ=2πd·sinθ/λ,  β=2πw·sinθ/λ)",
        fontsize=13, y=1.04,
    )
    plt.tight_layout()
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {fname}")


cheatsheet_figure("05_cheatsheet.png")

print("\n완료! 생성된 파일:")
print("  01_sweep_N.png     - 광원 개수만 바꿈")
print("  02_sweep_w.png     - 슬릿 폭만 바꿈")
print("  03_sweep_d.png     - 광원 간격만 바꿈")
print("  04_decompose.png   - 밝기 = 간섭 × 회절 분해")
print("  05_cheatsheet.png  - 요약표 (붙여넣기용)")
