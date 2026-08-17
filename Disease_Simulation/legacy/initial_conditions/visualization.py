"""
visualization.py
================================================================
초기 상태(S, I, R, N) 히트맵 시각화.

결과 png 는 프로젝트의 result/ 폴더에 저장한다(전역 규칙).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# initial_conditions/ 의 부모(Disease_Simulation)의 result 폴더
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(os.path.dirname(_PKG_DIR), "result")
os.makedirs(RESULT_DIR, exist_ok=True)


def plot_initial_conditions(S, I, R, N, title="Initial Conditions", filename=None,
                            dx=1.0):
    """S / I / R / N 네 장의 히트맵을 2x2 로 그린다.

    각 칸은 해당 집단의 인구 '밀도'[명/km²]. I 는 값 범위가 작아 별도
    스케일로 그려야 초기 발생 위치가 잘 보인다."""
    ny, nx = N.shape
    extent = [0, nx * dx, 0, ny * dx]
    panels = [
        ("S Susceptible density", S, "Blues"),
        ("I Infected density",   I, "inferno"),
        ("R Recovered/Immune density", R, "Greens"),
        ("N Population density (=S+I+R)", N, "viridis"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (label, field, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(field, cmap=cmap, origin="lower", extent=extent,
                       vmin=0, vmax=(field.max() if field.max() > 0 else 1))
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("people/km²", fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if filename is None:
        filename = "initial_conditions.png"
    out = os.path.join(RESULT_DIR, filename)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"저장: {out}")
    return out
