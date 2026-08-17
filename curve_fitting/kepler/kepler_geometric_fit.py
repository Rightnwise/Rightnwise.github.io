"""
타원 피팅: 대수적 거리 vs 기하학적 거리 목적함수 비교

- 대수적(algebraic): min sum F(x_i,y_i)^2,  F = a x^2 + b xy + c y^2 + d x + e y + f
  (Fitzgibbon, 닫힌 해. 빠르지만 편향)
- 기하학적(geometric): min sum (점→타원 최단거리)^2
  (비선형. 대수적 해를 초기값으로 LM 반복 최적화)

차이가 드러나도록 화성 부분궤도에 '측정 노이즈' 를 섞어 두 방법을 비교한다.

kepler_first_law.py 의 데이터/대수적-적합 함수 재사용.
"""

import numpy as np
import matplotlib.pyplot as plt

from kepler_first_law import (
    get_xyz, project_to_orbit_plane, fit_ellipse, ellipse_geometry,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ── 점 -> 표준타원(x^2/A^2+y^2/B^2=1) 최단거리 ──
def closest_dist(u, v, A, B):
    """로컬 좌표 (u,v) 에서 반축 A,B 타원까지의 최단거리."""
    ts = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    d2 = (A * np.cos(ts) - u) ** 2 + (B * np.sin(ts) - v) ** 2
    t = ts[np.argmin(d2)]                     # 거친 탐색으로 시작점
    for _ in range(10):                       # Newton 으로 정밀화
        ct, st = np.cos(t), np.sin(t)
        X, Y = A * ct, B * st
        gp = (X - u) * (-A * st) + (Y - v) * (B * ct)
        gpp = (A * st) ** 2 - (X - u) * A * ct + (B * ct) ** 2 - (Y - v) * B * st
        if abs(gpp) < 1e-12:
            break
        t -= gp / gpp
    X, Y = A * np.cos(t), B * np.sin(t)
    return np.hypot(X - u, Y - v)


def geo_residuals(params, x, y):
    """각 점의 기하학적 거리 벡터. params=[cx,cy,A,B,phi]."""
    cx, cy, A, B, phi = params
    A, B = abs(A), abs(B)
    ct, st = np.cos(phi), np.sin(phi)
    dx, dy = x - cx, y - cy
    u = ct * dx + st * dy                     # 타원 로컬 좌표
    v = -st * dx + ct * dy
    return np.array([closest_dist(ui, vi, A, B) for ui, vi in zip(u, v)])


def fit_ellipse_geometric(x, y, p0, max_iter=200, tol=1e-12):
    """기하학적 거리 제곱합을 LM 으로 최소화 (수치 야코비안)."""
    p = np.array(p0, float)
    lam = 1e-3
    r = geo_residuals(p, x, y)
    S = r @ r
    for _ in range(max_iter):
        # 수치 야코비안 (N x 5)
        J = np.zeros((len(x), 5))
        for k in range(5):
            dp = np.zeros(5)
            h = 1e-6 * max(abs(p[k]), 1e-3)
            dp[k] = h
            J[:, k] = (geo_residuals(p + dp, x, y) - r) / h
        JTJ, JTr = J.T @ J, J.T @ r
        improved = False
        for _ in range(30):
            try:
                delta = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), JTr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            pn = p - delta                    # 잔차 r 을 0 으로: p -= (JTJ)^-1 JTr
            rn = geo_residuals(pn, x, y)
            Sn = rn @ rn
            if Sn < S:
                p, r, lam = pn, rn, max(lam * 0.5, 1e-12)
                improved = True
                if S - Sn < tol * max(S, 1e-30):
                    return p
                S = Sn
                break
            lam *= 10
        if not improved:
            break
    return p


def params_from_coef(coef):
    g = ellipse_geometry(coef)
    phi = np.arctan2(g["major_dir"][1], g["major_dir"][0])
    return np.array([g["center"][0], g["center"][1], g["a"], g["b"], phi])


def geo_to_focus_ecc(params):
    cx, cy, A, B, phi = params
    A, B = abs(A), abs(B)
    A, B = max(A, B), min(A, B)
    c = np.sqrt(A ** 2 - B ** 2)
    cen = np.array([cx, cy])
    d = np.array([np.cos(phi), np.sin(phi)])
    foci = np.array([cen + c * d, cen - c * d])
    return foci, c / A


def ellipse_curve(params, n=400):
    cx, cy, A, B, phi = params
    t = np.linspace(0, 2 * np.pi, n)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    pts = (R @ np.array([abs(A) * np.cos(t), abs(B) * np.sin(t)])).T + [cx, cy]
    return pts[:, 0], pts[:, 1]


def rmse_geo(params, x, y):
    r = geo_residuals(params, x, y)
    return np.sqrt(np.mean(r ** 2))


# (이름, Horizons id, 실제 이심률, 부분궤도 start/stop/step)
BODIES = [
    ("Mercury", "199", 0.2056, ("2020-01-01", "2020-02-19", "2d")),
    ("Pluto",   "999", 0.2488, ("1900-01-01", "2024-01-01", "1500d")),
    ("Neptune", "899", 0.0113, ("1980-01-01", "2045-11-01", "800d")),
]
NOISE_FRAC = 0.02          # 궤도 크기 대비 측정 노이즈 비율(천체마다 자동 스케일)


def analyze_body(name, pid, span, seed):
    P = get_xyz(pid, *span)
    u, v, _, _ = project_to_orbit_plane(P)
    scale = np.mean(np.hypot(u, v))           # 궤도 크기(평균 반지름)
    rng = np.random.RandomState(seed)
    sigma = NOISE_FRAC * scale
    u = u + rng.normal(0, sigma, u.size)
    v = v + rng.normal(0, sigma, v.size)

    p_alg = params_from_coef(fit_ellipse(u, v))         # 대수적
    p_geo = fit_ellipse_geometric(u, v, p_alg)          # 기하학적
    _, ecc_a = geo_to_focus_ecc(p_alg)
    foci_g, ecc_g = geo_to_focus_ecc(p_geo)
    fs_a = float(np.min(np.linalg.norm(geo_to_focus_ecc(p_alg)[0], axis=1)))
    fs_g = float(np.min(np.linalg.norm(foci_g, axis=1)))
    return dict(u=u, v=v, sigma=sigma, p_alg=p_alg, p_geo=p_geo,
                ecc_a=ecc_a, ecc_g=ecc_g, fs_a=fs_a, fs_g=fs_g,
                rmse_a=rmse_geo(p_alg, u, v), rmse_g=rmse_geo(p_geo, u, v))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    print("=" * 78)
    print(f"{'body':<9}{'true e':>8}{'method':>11}"
          f"{'geo-RMSE':>11}{'ecc':>8}{'focus-Sun(AU)':>16}")
    print("-" * 78)
    for (name, pid, kecc, span), ax, seed in zip(BODIES, axes, [1, 2, 3]):
        r = analyze_body(name, pid, span, seed)
        for tag, ecc, fs, rm in [("algebraic", r["ecc_a"], r["fs_a"], r["rmse_a"]),
                                 ("geometric", r["ecc_g"], r["fs_g"], r["rmse_g"])]:
            print(f"{name if tag=='algebraic' else '':<9}"
                  f"{kecc if tag=='algebraic' else '':>8}"
                  f"{tag:>11}{rm:>11.4f}{ecc:>8.3f}{fs:>16.3f}")
        print("-" * 78)

        ax.plot(r["u"], r["v"], "o", ms=5, color="steelblue",
                label="noisy data", zorder=4)
        ax.plot(*ellipse_curve(r["p_alg"]), "b--", lw=2,
                label=f"algebraic (e={r['ecc_a']:.3f})")
        ax.plot(*ellipse_curve(r["p_geo"]), "r-", lw=2,
                label=f"geometric (e={r['ecc_g']:.3f})")
        ax.plot(0, 0, "*", color="gold", ms=18, mec="orange", label="Sun")
        ax.set_aspect("equal")
        ax.set_title(f"{name}  (true e={kecc})\n"
                     f"focus-Sun: alg={r['fs_a']:.2f}, geo={r['fs_g']:.2f} AU",
                     fontsize=11)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    print("=" * 78)

    fig.suptitle("Ellipse fit: algebraic vs geometric distance "
                 "(partial arc + 2% noise)", fontsize=14)
    fig.tight_layout()
    fig.savefig("kepler_geometric_fit.png", dpi=120)
    print("그래프를 kepler_geometric_fit.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
