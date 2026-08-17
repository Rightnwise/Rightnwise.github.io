"""
케플러 제1법칙 실험 — '부분 궤도' 데이터로 타원 회귀(ellipse fitting)

케플러 1법칙: 행성은 태양을 한 '초점(focus)' 으로 하는 타원 궤도를 돈다.

아이디어:
  - 한 주기 전체가 아니라 궤도의 '일부(arc)' 위치 데이터만 JPL Horizons 에서 받는다.
  - 그 부분 점들에 일반 원뿔곡선(타원)을 최소제곱으로 맞춘다 (Fitzgibbon 직접 타원 적합).
  - 복원된 타원의 '초점' 이 실제로 태양(원점)에 오는지 확인 → 1법칙 검증.

astropy: 궤도면 투영/이심률 비교 등에 단위·좌표 활용.
"""

import numpy as np
import matplotlib.pyplot as plt
from astroquery.jplhorizons import Horizons

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def get_xyz(planet_id, start, stop, step):
    v = Horizons(id=planet_id, location="@sun",
                 epochs={"start": start, "stop": stop, "step": step}).vectors()
    return np.array([v["x"], v["y"], v["z"]]).T   # (N,3) AU


def project_to_orbit_plane(P):
    """원점(태양)을 지나는 궤도면에 점들을 2D 로 투영.

    점들이 거의 한 평면 위에 있으므로 SVD 로 평면 기저를 잡는다.
    원점은 그대로 (0,0) 으로 간다(태양이 평면 안에 있으므로).
    반환: (u, v) 2D 좌표, 평면 기저 (e1, e2)
    """
    _, _, Vt = np.linalg.svd(P)
    e1, e2 = Vt[0], Vt[1]          # 평면을 span 하는 직교 기저
    return P @ e1, P @ e2, e1, e2


def fit_ellipse(x, y):
    """Fitzgibbon 직접 타원 적합. 반환: 일반원뿔 계수 [a,b,c,d,e,f]
    (a x^2 + b xy + c y^2 + d x + e y + f = 0), 타원 보장."""
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    C1 = np.array([[0, 0, 2.0], [0, -1.0, 0], [2.0, 0, 0]])
    M = np.linalg.solve(C1, M)
    eigval, eigvec = np.linalg.eig(M)
    cond = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2   # 타원 조건 > 0
    a1 = eigvec[:, np.where(cond > 0)[0][0]]
    a2 = T @ a1
    return np.concatenate([a1, a2])                     # [a,b,c,d,e,f]


def ellipse_geometry(coef):
    """일반원뿔 계수 -> 중심, 반장축/반단축, 회전, 초점, 이심률."""
    a, b, c, d, e, f = coef
    M = np.array([[a, b / 2], [b / 2, c]])
    cen = np.linalg.solve(2 * M, [-d, -e])              # 중심
    fp = a*cen[0]**2 + b*cen[0]*cen[1] + c*cen[1]**2 + d*cen[0] + e*cen[1] + f
    eigval, eigvec = np.linalg.eigh(M)
    axes = np.sqrt(-fp / eigval)                        # 반축 길이
    # 장축/단축 정리
    if axes[0] >= axes[1]:
        A, Bx = axes[0], axes[1]; major_dir = eigvec[:, 0]
    else:
        A, Bx = axes[1], axes[0]; major_dir = eigvec[:, 1]
    cdist = np.sqrt(max(A**2 - Bx**2, 0))               # 중심~초점 거리
    foci = np.array([cen + cdist * major_dir, cen - cdist * major_dir])
    ecc = cdist / A
    return dict(center=cen, a=A, b=Bx, foci=foci, ecc=ecc, major_dir=major_dir)


def ellipse_points(coef, n=400):
    """타원 위 점들 (그리기용) — 일반원뿔 계수로부터."""
    g = ellipse_geometry(coef)
    t = np.linspace(0, 2 * np.pi, n)
    ang = np.arctan2(g["major_dir"][1], g["major_dir"][0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = (R @ np.array([g["a"] * np.cos(t), g["b"] * np.sin(t)])).T + g["center"]
    return pts[:, 0], pts[:, 1]


def main():
    pid, name = "499", "Mars"
    known_ecc = 0.0934                                  # 화성 실제 이심률(참고)

    print("=" * 60)
    print(f"행성: {name}  (궤도주기 ≈ 687일)")

    # ── Step 1) '부분' 궤도 데이터 (약 300일, 전체의 ~44%) ──
    Pp = get_xyz(pid, "2020-01-01", "2020-10-27", "15d")
    print(f"Step 1) 부분 궤도 데이터: {len(Pp)} 점  (약 300일, 전체의 ~44%)")

    # 참고용: 전체 궤도(진짜 타원) — 그리기 비교용으로만 사용
    Pf = get_xyz(pid, "2020-01-01", "2021-12-19", "20d")

    # ── Step 2) 궤도면에 2D 투영 ──
    u, vv, e1, e2 = project_to_orbit_plane(Pp)
    uf, vf = Pf @ e1, Pf @ e2                            # 전체궤도도 같은 평면에

    # ── Step 3) 부분 점들에 타원 회귀 ──
    coef = fit_ellipse(u, vv)
    g = ellipse_geometry(coef)
    # 두 초점 중 원점(태양)에 가까운 쪽
    d_focus = np.linalg.norm(g["foci"], axis=1)
    near = g["foci"][np.argmin(d_focus)]
    print("Step 3) 타원 회귀 결과")
    print(f"  반장축 a = {g['a']:.4f} AU,  반단축 b = {g['b']:.4f} AU")
    print(f"  이심률 e = {g['ecc']:.4f}   (실제 {known_ecc})")
    print(f"  가까운 초점 위치 = ({near[0]:+.4f}, {near[1]:+.4f}) AU")
    print(f"  초점~태양(원점) 거리 = {min(d_focus):.4f} AU  <- 0 에 가까울수록 1법칙 성립")
    print("=" * 60)

    # ── 그림 ──
    ex, ey = ellipse_points(coef)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(uf, vf, color="0.8", lw=1.2, zorder=1,
            label="true full orbit (reference)")
    ax.plot(ex, ey, "r-", lw=2, zorder=3,
            label="fitted ellipse (from partial data)")
    ax.plot(u, vv, "o", ms=6, color="steelblue", zorder=4,
            label="partial data used for fit")
    ax.plot(0, 0, "*", color="gold", ms=22, mec="orange", zorder=5,
            label="Sun (origin)")
    ax.plot(g["foci"][:, 0], g["foci"][:, 1], "x", color="red", mew=2, ms=10,
            zorder=5, label="ellipse foci")

    ax.set_aspect("equal")
    ax.set_xlabel("x in orbital plane [AU]")
    ax.set_ylabel("y in orbital plane [AU]")
    ax.set_title(f"Kepler's 1st law: ellipse fit to a PARTIAL arc ({name})\n"
                 f"recovered e={g['ecc']:.3f}, focus-to-Sun={min(d_focus):.3f} AU")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("kepler_first_law.png", dpi=130)
    print("그래프를 kepler_first_law.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
