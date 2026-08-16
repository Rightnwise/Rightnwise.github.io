"""
4차 다항식 피팅에서 '정규화(normalization)' 의 중요성

x 값이 크면(예: 픽셀 좌표, 타임스탬프, 큰 정수) 4차 다항식의 설계행렬
[x^4, x^3, x^2, x, 1] 이 극도로 ill-scaled 되어:

  - float64: 정규방정식 X^T X 의 조건수가 폭발(~1e37 이상) → 해가 garbage
  - float32: x^4, x^8 항이 표현 범위를 넘겨 literal overflow(inf/nan)

해결: x 를 정규화(중심화 + 스케일, 보통 [-1,1] 또는 표준화)하면
설계행렬이 안정되어 똑같은 데이터인데도 깔끔하게 피팅된다.

이 스크립트는 실패(정규화 X)와 성공(정규화 O)을 나란히 보여준다.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def vander_cond(x, deg=4):
    """정규방정식 X^T X 의 조건수 (클수록 불안정)."""
    V = np.vander(x, deg + 1)
    return np.linalg.cond(V.T @ V)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    deg = 4

    # ── 데이터: x 가 매우 큰 영역 (1e6 부근) 에 있는 4차 곡선 ──
    rng = np.random.RandomState(0)
    x = np.linspace(1_000_000, 1_000_000 + 100, 80)     # ill-scaled large x
    # 참값은 정규화 좌표 t 로 정의(깨끗한 W 모양)
    t_true = (x - x.mean()) / (x.max() - x.min()) * 2     # ~[-1, 1]
    y = 3 - 5 * t_true ** 2 + 4 * t_true ** 4 + rng.normal(0, 0.05, x.size)

    print("=" * 64)
    print(f"x 범위: [{x.min():.0f}, {x.max():.0f}]   (4차 피팅)")

    # ── (실패) 정규화 없이 raw x 로 피팅 ────────────────
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        coef_raw = np.polyfit(x, y, deg)          # 경고 발생 가능(RankWarning 등)
    yhat_raw = np.polyval(coef_raw, x)
    cond_raw = vander_cond(x, deg)
    print("-" * 64)
    print("[정규화 X]  raw x 사용")
    print(f"  cond(X^T X) = {cond_raw:.2e}   (수치적으로 거의 특이)")
    print(f"  RMSE = {rmse(y, yhat_raw):.4g}")
    print(f"  계수 절댓값 최대 = {np.max(np.abs(coef_raw)):.3e}")
    if wlist:
        print(f"  경고: {wlist[0].category.__name__}")

    # ── float32 로는 literal overflow(inf) 까지 발생 ────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        V32 = np.vander(x.astype(np.float32), deg + 1)
        A32 = V32.astype(np.float32).T @ V32.astype(np.float32)
    x8_f32 = np.float32(x[0]) ** 8                 # X^T X 에 들어가는 항
    print(f"  (float32) x^8 = {x8_f32}  -> overflow(inf)? {not np.isfinite(x8_f32)}")

    # ── (성공) x 를 정규화 후 피팅 ──────────────────────
    mu, sigma = x.mean(), x.std()
    u = (x - mu) / sigma                            # 표준화: 평균0, 표준편차1
    coef_norm = np.polyfit(u, y, deg)
    yhat_norm = np.polyval(coef_norm, u)
    cond_norm = vander_cond(u, deg)
    print("-" * 64)
    print("[정규화 O]  u = (x - mean) / std 사용")
    print(f"  cond(X^T X) = {cond_norm:.2e}   (안정)")
    print(f"  RMSE = {rmse(y, yhat_norm):.4g}")
    print("=" * 64)

    # ── 그림: 실패 vs 성공 ──────────────────────────────
    xs = np.linspace(x.min(), x.max(), 400)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(x, y, s=18, color="black", label="data")
    ax1.plot(xs, np.polyval(coef_raw, xs), "r-", lw=2, label="quartic fit")
    ax1.set_title(f"FAIL: no normalization\ncond={cond_raw:.1e}, "
                  f"RMSE={rmse(y, yhat_raw):.2g}")
    ax1.set_xlabel("x (raw, ~1e6)")
    ax1.set_ylabel("y")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(x, y, s=18, color="black", label="data")
    ax2.plot(xs, np.polyval(coef_norm, (xs - mu) / sigma), "g-", lw=2,
             label="quartic fit")
    ax2.set_title(f"OK: normalized x\ncond={cond_norm:.1e}, "
                  f"RMSE={rmse(y, yhat_norm):.2g}")
    ax2.set_xlabel("x (raw, ~1e6)")
    ax2.set_ylabel("y")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Quartic fit: why normalization matters", fontsize=14)
    fig.tight_layout()
    fig.savefig("quartic_normalization_demo.png", dpi=150)
    print("그래프를 quartic_normalization_demo.png 로 저장했습니다.")
    plt.show()


if __name__ == "__main__":
    main()
