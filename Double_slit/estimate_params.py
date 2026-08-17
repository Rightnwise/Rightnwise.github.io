"""
임의의 빛 무늬 2개를 만들고, 그래프만 보고 d, w, N 을 역추정하는 과정.

읽는 법 (세 가지 단서):
  d = λ / Δu          (Δu = 이웃 주극대 사이 간격)
  w = λ / u_env       (u_env = 회절 포락선이 0 이 되는 위치 = '빠진 차수')
  N = (부극대 개수) + 2   (주극대 하나 사이에 낀 작은 봉우리 수 + 2)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

LAM = 0.5
U = np.linspace(-0.28, 0.28, 6000)

# ---- 비밀 파라미터 (그래프에서 이걸 맞혀야 함) ----
SECRETS = [
    dict(name="무늬 A", d=6.0, w=2.0, N=4),
    dict(name="무늬 B", d=5.0, w=2.5, N=3),
]


def pattern(N, d, w):
    phi = 2 * np.pi * d * U / LAM
    den = np.sin(phi / 2)
    g = np.where(np.abs(den) < 1e-9, N**2, (np.sin(N * phi / 2) / den) ** 2) / N**2
    env = np.sinc(w * U / LAM) ** 2
    return g * env


def local_maxima(y, thr):
    idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > thr))[0] + 1
    return idx


def estimate(I):
    """무늬 I 에서 d, w, N 추정."""
    peaks = local_maxima(I, 1e-4)
    up = U[peaks]
    hp = I[peaks]

    # 1) 주극대 = 키 큰 봉우리 / 부극대 = 작은 봉우리
    principal = up[hp > 0.12]
    principal = np.sort(principal)

    # 2) 주극대 간격 -> d
    diffs = np.diff(principal)
    s = np.min(diffs)                      # 참 간격 (빠진 차수는 2s 로 나타남)
    d_est = LAM / s

    # 3) 빠진 차수(포락선 0) 찾기 -> w
    w_est = np.nan
    for k in range(1, 12):
        u_expect = k * s
        if u_expect > U.max():
            break
        near = np.min(np.abs(principal - u_expect))
        if near > 0.35 * s:               # 있어야 할 자리에 주극대 없음 = 빠진 차수
            w_est = LAM / u_expect
            u_env = u_expect
            break
    else:
        u_env = np.nan

    # 4) 중앙 주극대(0)와 다음 주극대(s) 사이 부극대 개수 -> N
    n_sub = np.sum((up > 0.02 * s) & (up < s - 0.02 * s) & (hp < 0.12))
    N_est = int(n_sub) + 2

    return dict(d=d_est, w=w_est, N=N_est, principal=principal,
                s=s, u_env=u_env)


# ---------------- 그리기 (깔끔한 라벨 몇 개만) ----------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7.6))

for ax, sec in zip(axes, SECRETS):
    I = pattern(sec["N"], sec["d"], sec["w"])
    est = estimate(I)
    s, d, ue, w, N = est["s"], est["d"], est["u_env"], est["w"], est["N"]
    nsub = N - 2

    ax.plot(U, I, color="navy", lw=1.4)
    ax.fill_between(U, 0, I, color="navy", alpha=0.14)

    # 주극대 두 개 표시 + 간격 Δu 화살표 -> d
    for p in [0.0, s]:
        ax.plot(p, I[np.argmin(np.abs(U - p))], "o", color="crimson", ms=5, zorder=5)
    ax.annotate("", xy=(s, 1.06), xytext=(0.0, 1.06),
                arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax.text(0.5 * (0 + s), 0.99, r"$\Delta u$", ha="center", color="green", fontsize=10)

    # 빠진 차수(포락선 0) 위치 -> w
    if not np.isnan(ue):
        ax.axvline(ue, color="crimson", ls=":", lw=1.1)
        ax.plot(ue, 0.02, marker="x", color="crimson", ms=8, mew=2)
        ax.text(ue, 0.30, r"$u_0$", ha="center", color="crimson", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    # 상단에 세 결론 라벨 (LaTeX, 한 줄 형태로 잘리지 않게)
    ax.text(0.0, 1.16, r"$d=\lambda/\Delta u\approx %.1f$" % d,
            ha="center", color="green", fontsize=11, clip_on=False)
    if not np.isnan(w):
        ax.text(U.max() - 0.005, 1.16, r"$w=\lambda/u_0\approx %.1f$" % w,
                ha="right", color="crimson", fontsize=11, clip_on=False)
    ax.text(U.min() + 0.005, 1.16, r"$N=\#{+}2=%d$" % N,
            ha="left", color="navy", fontsize=11, clip_on=False)

    ax.set_title(sec["name"] + r"   ($\lambda=0.5$)", fontsize=12)
    ax.set_xlabel(r"$u=\sin\theta$")
    ax.set_ylabel("밝기")
    ax.set_ylim(0, 1.26)
    ax.set_xlim(U.min() - 0.012, U.max() + 0.012)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("estimate_params.png", dpi=140, bbox_inches="tight", pad_inches=0.25)
plt.show()

# 콘솔 출력
print(f"λ = {LAM}")
for sec in SECRETS:
    est = estimate(pattern(sec["N"], sec["d"], sec["w"]))
    print(f"\n{sec['name']}")
    print(f"  Δu={est['s']:.4f} → d=λ/Δu={est['d']:.2f}  (실제 {sec['d']})")
    print(f"  u_env={est['u_env']:.4f} → w=λ/u_env={est['w']:.2f}  (실제 {sec['w']})")
    print(f"  부극대 {est['N']-2}개 → N={est['N']}  (실제 {sec['N']})")
