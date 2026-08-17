"""
케플러 1법칙 — 모든 외행성을 '같은 기간(5년)' 으로 받아 타원 회귀

앞 실험은 각 행성 주기의 ~40% 를 받았지만,
이번엔 2010-2015 (5년) '같은 달력 기간' 을 모두에게 똑같이 적용한다.
먼 행성일수록 같은 5년이 궤도의 더 짧은 호(=작은 각도)만 덮으므로
타원 회귀가 점점 어려워지는 것을 보여준다.

kepler_first_law.py 의 함수 재사용.
"""

import numpy as np
import matplotlib.pyplot as plt

from kepler_first_law import (
    get_xyz, project_to_orbit_plane, fit_ellipse,
    ellipse_geometry, ellipse_points,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SPAN = ("2010-01-01", "2015-01-01", "50d")   # 모두 동일한 5년

# (이름, id, 실제 이심률, 주기(년), 전체궤도 ref start/stop/step)
CONFIG = [
    ("Jupiter", "599", 0.0489, 11.867, ("2010-01-01", "2022-01-01", "120d")),
    ("Saturn", "699", 0.0565, 29.657, ("2010-01-01", "2040-01-01", "300d")),
    ("Uranus", "799", 0.0457, 84.328, ("1990-01-01", "2074-01-01", "900d")),
    ("Neptune", "899", 0.0113, 165.088, ("1980-01-01", "2145-01-01", "2000d")),
]
SPAN_YEARS = 5.0


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    print("=" * 70)
    print(f"모든 행성 동일 기간: {SPAN[0]} ~ {SPAN[1]} ({SPAN_YEARS:.0f}년)")
    print("-" * 70)
    for (name, pid, kecc, period, full), ax in zip(CONFIG, axes.ravel()):
        Pp = get_xyz(pid, *SPAN)
        Pf = get_xyz(pid, *full)
        u, vv, e1, e2 = project_to_orbit_plane(Pp)
        uf, vf = Pf @ e1, Pf @ e2
        frac = 100 * SPAN_YEARS / period          # 궤도 커버 비율

        coef = fit_ellipse(u, vv)
        g = ellipse_geometry(coef)
        focus_sun = float(np.min(np.linalg.norm(g["foci"], axis=1)))
        print(f"{name:<8} 궤도커버={frac:5.1f}%  점{len(u):>3}  "
              f"a={g['a']:8.3f}AU  e(복원)={g['ecc']:.3f}(실제 {kecc})  "
              f"초점~태양={focus_sun:.3f}AU")

        ex, ey = ellipse_points(coef)
        ax.plot(uf, vf, color="0.8", lw=1.2, label="true orbit (ref)")
        ax.plot(ex, ey, "r-", lw=2, label="fitted ellipse")
        ax.plot(u, vv, "o", ms=5, color="steelblue", label="5-yr data")
        ax.plot(0, 0, "*", color="gold", ms=18, mec="orange", label="Sun")
        ax.plot(g["foci"][:, 0], g["foci"][:, 1], "x", color="red",
                mew=2, ms=9, label="foci")
        ax.set_aspect("equal")
        ax.set_title(f"{name}: 5yr = {frac:.0f}% of orbit\n"
                     f"e={g['ecc']:.3f}(real {kecc}), "
                     f"focus-Sun={focus_sun:.2f} AU", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    print("=" * 70)

    fig.suptitle("Kepler's 1st law: SAME 5-year data for all outer planets\n"
                 "(far planets cover a tiny arc -> ellipse fit degrades)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("kepler_first_law_samespan.png", dpi=120)
    print("그래프를 kepler_first_law_samespan.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
