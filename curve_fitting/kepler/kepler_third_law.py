"""
케플러 제3법칙 실험 — 실제 관측 데이터(JPL Horizons) + astropy 단위 검산 + log-log 회귀

케플러 3법칙:  T^2 = (4 pi^2 / (G M)) a^3   ~   T ∝ a^(3/2)
=> log-log 평면(log T vs log a)에서 직선, 기울기 = 1.5

단계:
  1) JPL Horizons 에서 8개 행성의 a(긴반지름), P(주기) 실측값 수집
  2) astropy 상수/단위로 T^2 = 4pi^2/(GM) a^3 검산
  3) log10(P) vs log10(a) 선형회귀 → 기울기가 1.5 인지 확인
"""

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import G, M_sun
from astroquery.jplhorizons import Horizons

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

PLANETS = [
    ("Mercury", "199"), ("Venus", "299"), ("Earth", "399"),
    ("Mars", "499"), ("Jupiter", "599"), ("Saturn", "699"),
    ("Uranus", "799"), ("Neptune", "899"),
]


def fetch():
    """각 행성의 a(AU), P(년) 실측값을 Horizons 에서 가져온다."""
    names, a_au, P_yr = [], [], []
    for name, hid in PLANETS:
        el = Horizons(id=hid, location="@sun",
                      epochs=2451545.0, id_type=None).elements()
        names.append(name)
        a_au.append(float(el["a"][0]))               # AU
        P_yr.append(float(el["P"][0]) / 365.25)      # days -> years
        print(f"  {name:<8} a = {a_au[-1]:8.4f} AU   P = {P_yr[-1]:10.4f} yr")
    return names, np.array(a_au), np.array(P_yr)


def main():
    print("=" * 60)
    print("Step 1) JPL Horizons 실측 데이터")
    names, a, P = fetch()

    # ── Step 2) astropy 로 단위 검산 ──
    print("-" * 60)
    print("Step 2) astropy 단위 검산  T = 2π√(a³/GM)")
    a_q = a * u.au
    T_theory = (2 * np.pi * np.sqrt(a_q ** 3 / (G * M_sun))).to(u.year)
    for nm, po, tt in zip(names, P, T_theory.value):
        print(f"  {nm:<8} 관측 P={po:9.3f} yr   이론 T={tt:9.3f} yr   "
              f"오차={100*abs(po-tt)/po:5.2f}%")

    # ── Step 3) log-log 회귀 ──
    print("-" * 60)
    print("Step 3) log-log 선형회귀  (log10 P vs log10 a)")
    x = np.log10(a)
    y = np.log10(P)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    rmse = np.sqrt(np.mean((y - yhat) ** 2))          # log 공간 거리(잔차)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  기울기(slope)   = {slope:.4f}   (이론값 1.5)")
    print(f"  절편(intercept) = {intercept:.4f}   (이론값 0, 단위 AU·yr)")
    print(f"  R^2             = {r2:.6f}")
    print(f"  RMSE(log space) = {rmse:.4e}")
    print("=" * 60)

    # ── 그림: log-log plot ──
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.loglog(a, P, "o", ms=9, color="steelblue", zorder=3, label="planets (observed)")
    for nm, ai, pi in zip(names, a, P):
        ax.annotate(nm, (ai, pi), textcoords="offset points",
                    xytext=(8, -4), fontsize=9)

    xs = np.logspace(np.log10(a.min()) - 0.2, np.log10(a.max()) + 0.2, 100)
    ax.loglog(xs, 10 ** intercept * xs ** slope, "r-", lw=2,
              label=f"fit:  slope = {slope:.3f}")
    ax.loglog(xs, xs ** 1.5, "g--", lw=1.5,
              label="Kepler ideal:  slope = 1.5")

    ax.set_xlabel("semi-major axis  a  [AU]  (log)")
    ax.set_ylabel("orbital period  T  [yr]  (log)")
    ax.set_title("Kepler's 3rd law:  $T^2 \\propto a^3$  (real data, log-log)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("kepler_third_law.png", dpi=130)
    print("그래프를 kepler_third_law.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
