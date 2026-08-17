"""
케플러 제1법칙 — 외행성(목성/토성/천왕성/해왕성) 부분 궤도 타원 회귀

먼 행성은 주기가 길어(목성 ~12yr ... 해왕성 ~165yr) 부분 호도 긴 기간을 받는다.
각 행성마다 전체의 약 40% 만 받아 타원을 회귀하고, 초점이 태양에 오는지 확인.

kepler_first_law.py 의 함수들을 재사용한다.
"""

import numpy as np
import matplotlib.pyplot as plt

from kepler_first_law import (
    get_xyz, project_to_orbit_plane, fit_ellipse,
    ellipse_geometry, ellipse_points,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# (이름, Horizons id, 실제 이심률, 부분호 start/stop/step, 전체궤도 start/stop/step)
CONFIG = [
    ("Jupiter", "599", 0.0489,
     ("2010-01-01", "2014-10-01", "50d"),
     ("2010-01-01", "2022-01-01", "120d")),
    ("Saturn", "699", 0.0565,
     ("2010-01-01", "2021-10-01", "130d"),
     ("2010-01-01", "2040-01-01", "300d")),
    ("Uranus", "799", 0.0457,
     ("1990-01-01", "2023-08-01", "400d"),
     ("1990-01-01", "2074-01-01", "900d")),
    ("Neptune", "899", 0.0113,
     ("1980-01-01", "2045-11-01", "800d"),
     ("1980-01-01", "2145-01-01", "2000d")),
]


def analyze(pid, part, full):
    Pp = get_xyz(pid, *part)
    Pf = get_xyz(pid, *full)
    u, vv, e1, e2 = project_to_orbit_plane(Pp)
    uf, vf = Pf @ e1, Pf @ e2
    coef = fit_ellipse(u, vv)
    g = ellipse_geometry(coef)
    d_focus = np.linalg.norm(g["foci"], axis=1)
    return dict(u=u, v=vv, uf=uf, vf=vf, coef=coef, g=g,
                npts=len(u), focus_sun=float(min(d_focus)))


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    print("=" * 64)
    for (name, pid, kecc, part, full), ax in zip(CONFIG, axes.ravel()):
        r = analyze(pid, part, full)
        g = r["g"]
        print(f"{name:<8} 점 {r['npts']:>3}개  반장축 a={g['a']:8.3f} AU  "
              f"e(복원)={g['ecc']:.4f} (실제 {kecc})  "
              f"초점~태양={r['focus_sun']:.4f} AU")

        ex, ey = ellipse_points(r["coef"])
        ax.plot(r["uf"], r["vf"], color="0.8", lw=1.2,
                label="true orbit (ref)")
        ax.plot(ex, ey, "r-", lw=2, label="fitted ellipse")
        ax.plot(r["u"], r["v"], "o", ms=5, color="steelblue",
                label="partial data")
        ax.plot(0, 0, "*", color="gold", ms=18, mec="orange",
                label="Sun")
        ax.plot(g["foci"][:, 0], g["foci"][:, 1], "x", color="red",
                mew=2, ms=9, label="foci")
        ax.set_aspect("equal")
        ax.set_title(f"{name}: e={g['ecc']:.3f} (real {kecc}), "
                     f"focus-Sun={r['focus_sun']:.3f} AU", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    print("=" * 64)

    fig.suptitle("Kepler's 1st law for outer planets "
                 "(ellipse fit to ~40% partial arc)", fontsize=14)
    fig.tight_layout()
    fig.savefig("kepler_first_law_outer.png", dpi=120)
    print("그래프를 kepler_first_law_outer.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
