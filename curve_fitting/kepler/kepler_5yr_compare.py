"""
5년치 동일 관측데이터로 타원 피팅: algebraic vs geometric vs ground truth

- 모든 외행성에 2010-2015 (5년) 같은 관측구간 적용 (+ 소량 관측 노이즈)
- 두 목적함수로 타원 적합: 대수적(algebraic) / 기하학적(geometric)
- 실제 궤도(ground truth: Horizons 전체궤도 곡선 + 알려진 이심률)와 비교
- ground truth 에서 태양은 정확히 초점(focus-Sun = 0), e = 알려진 값

kepler_first_law.py / kepler_geometric_fit.py 함수 재사용.
"""

import numpy as np
import matplotlib.pyplot as plt

from kepler_first_law import get_xyz, project_to_orbit_plane, fit_ellipse
from kepler_geometric_fit import (
    fit_ellipse_geometric, params_from_coef, geo_to_focus_ecc,
    ellipse_curve, rmse_geo,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SPAN = ("2010-01-01", "2025-01-01", "150d")    # 모두 동일한 15년 관측
SPAN_YEARS = 15.0
NOISE_FRAC = 0.005                              # 궤도크기 대비 관측 노이즈

# (이름, id, 실제 e, 주기(년), 전체궤도 ref start/stop/step)
BODIES = [
    ("Jupiter", "599", 0.0489, 11.867, ("2010-01-01", "2022-01-01", "120d")),
    ("Saturn", "699", 0.0565, 29.657, ("2010-01-01", "2040-01-01", "300d")),
    ("Uranus", "799", 0.0457, 84.328, ("1990-01-01", "2074-01-01", "900d")),
    ("Neptune", "899", 0.0113, 165.088, ("1980-01-01", "2145-01-01", "2000d")),
]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    print("=" * 80)
    print(f"동일 {SPAN_YEARS:.0f}년 관측 ({SPAN[0]}~{SPAN[1]}) + 노이즈 {NOISE_FRAC*100:.1f}%")
    print(f"{'body':<9}{'cover%':>7}{'method':>11}{'geo-RMSE':>10}"
          f"{'ecc':>8}{'focus-Sun':>11}")
    print("-" * 80)
    for (name, pid, kecc, period, full), ax, seed in zip(
            BODIES, axes.ravel(), [1, 2, 3, 4]):
        # 5년 관측 + 같은 평면 투영
        P = get_xyz(pid, *SPAN)
        u, v, e1, e2 = project_to_orbit_plane(P)
        scale = np.mean(np.hypot(u, v))
        rng = np.random.RandomState(seed)
        sig = NOISE_FRAC * scale
        u = u + rng.normal(0, sig, u.size)
        v = v + rng.normal(0, sig, v.size)
        cover = 100 * SPAN_YEARS / period

        # ground truth 궤도 곡선 (같은 평면에 투영)
        Pf = get_xyz(pid, *full)
        gx, gy = Pf @ e1, Pf @ e2

        # 두 방법 피팅
        p_alg = params_from_coef(fit_ellipse(u, v))
        p_geo = fit_ellipse_geometric(u, v, p_alg)
        fa, ea = geo_to_focus_ecc(p_alg)
        fg, eg = geo_to_focus_ecc(p_geo)
        fs_a = float(np.min(np.linalg.norm(fa, axis=1)))
        fs_g = float(np.min(np.linalg.norm(fg, axis=1)))

        print(f"{name:<9}{cover:>6.1f}%{'algebraic':>11}{rmse_geo(p_alg,u,v):>10.3f}"
              f"{ea:>8.3f}{fs_a:>11.3f}")
        print(f"{'':<9}{'':>7}{'geometric':>11}{rmse_geo(p_geo,u,v):>10.3f}"
              f"{eg:>8.3f}{fs_g:>11.3f}")
        print(f"{'':<9}{'':>7}{'ground truth':>11}{'-':>10}{kecc:>8.3f}"
              f"{0.0:>11.3f}")
        print("-" * 80)

        # 그림
        ax.plot(gx, gy, color="0.6", lw=1.5, label="ground truth orbit")
        ax.plot(*ellipse_curve(p_alg), "b--", lw=2, label=f"algebraic (e={ea:.3f})")
        ax.plot(*ellipse_curve(p_geo), "r-", lw=2, label=f"geometric (e={eg:.3f})")
        ax.plot(u, v, "o", ms=4, color="steelblue", label="data", zorder=5)
        ax.plot(0, 0, "*", color="gold", ms=18, mec="orange", label="Sun (true focus)")
        # 축을 실제 궤도 크기에 고정 (퇴화된 적합은 화면 밖으로 잘림)
        m = 1.3 * max(np.abs(gx).max(), np.abs(gy).max())
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_aspect("equal")
        ax.set_title(f"{name}: {SPAN_YEARS:.0f}yr = {cover:.0f}% of orbit  (true e={kecc})\n"
                     f"focus-Sun  alg={fs_a:.2f}  geo={fs_g:.2f}  truth=0 AU",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    print("=" * 80)

    fig.suptitle(f"{SPAN_YEARS:.0f}-year observation: algebraic vs geometric vs ground truth",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig("kepler_5yr_compare.png", dpi=120)
    print("그래프를 kepler_5yr_compare.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
