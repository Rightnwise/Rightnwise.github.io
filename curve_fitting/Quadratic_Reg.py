import numpy as np
import matplotlib.pyplot as plt
x_coords = []
y_coords = []


lim = 20
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xticks([t for t in ax.get_xticks() if t != 0])
ax.grid(True, alpha = 0.3)

# 깨끗한 포물선(y ~ 0.04 x^2 - 1) 위의 점들 + 이상점 1개
# 이상점(2, 18) 때문에 L2(제곱)와 L1(절댓값) 결과가 크게 갈린다.
x_coords = np.array([-15, -10, -5,  0,  5, 10, 15,   2])
y_coords = np.array([  8,   3,  0, -1,  0,  3,  8,  18])
outlier_idx = 7   # 마지막 점 (2, 18) 이 이상점
n = len(x_coords)

# mode = input("Choose mode: (1. Click points, 2. Enter points): ")
# ax.set_title("Click points, then press Enter")
# if mode == "1":
#     pts = plt.ginput(n=-1, timeout = 0)
#     x_coords = np.array([p[0] for p in pts])
#     y_coords = np.array([p[1] for p in pts])
#     n = len(x_coords)
# else:
#     n = int(input("Enter number of points: "))
#     x_coords, y_coords = [], []
#     for i in range(n):
#         x_coords.append(float(input(f"x{i+1}:")))
#         y_coords.append(float(input(f"y{i+1}:")))
#     x_coords = np.array(x_coords)
#     y_coords = np.array(y_coords) 

# 설계행렬 M: 각 행이 [x^2, x, 1],  모델 y = a x^2 + b x + c
def design(xv):
    return np.column_stack([xv**2, xv, np.ones_like(xv)])


# ── L2 회귀: sum (residual)^2 최소화 (정규방정식, 닫힌 해) ──
def fit_l2(xv, yv):
    M = design(xv)
    return np.linalg.solve(M.T @ M, M.T @ yv)   # [a, b, c]


# ── L1 회귀: sum |residual| 최소화 (IRLS, 반복 가중 최소제곱) ──
# 가중치 w_i = 1/|r_i| 로 반복하면 가중최소제곱이 L1 해로 수렴한다.
def fit_l1(xv, yv, iters=200, eps=1e-6, tol=1e-9):
    M = design(xv)
    coef = fit_l2(xv, yv)            # L2 해에서 출발
    for _ in range(iters):
        r = yv - M @ coef
        w = 1.0 / np.maximum(np.abs(r), eps)   # 0 division 방지
        A = M.T @ (w[:, None] * M)
        bb = M.T @ (w * yv)
        coef_new = np.linalg.solve(A, bb)
        if np.max(np.abs(coef_new - coef)) < tol:
            coef = coef_new
            break
        coef = coef_new
    return coef


coef_l2 = fit_l2(x_coords, y_coords)
coef_l1 = fit_l1(x_coords, y_coords)


# ── 두 방법 비교: 각 적합의 두 목적함수 값 계산 ──
def metrics(coef):
    r = y_coords - design(x_coords) @ coef
    return np.sum(r ** 2), np.sum(np.abs(r))   # (sum r^2, sum |r|)


sse_l2, sabs_l2 = metrics(coef_l2)
sse_l1, sabs_l1 = metrics(coef_l1)

print("=" * 60)
print("model: y = a x^2 + b x + c")
print(f"L2 fit (min sum r^2 ): a={coef_l2[0]:.4f}, b={coef_l2[1]:.4f}, "
      f"c={coef_l2[2]:.4f}")
print(f"L1 fit (min sum |r| ): a={coef_l1[0]:.4f}, b={coef_l1[1]:.4f}, "
      f"c={coef_l1[2]:.4f}")
print("-" * 60)
print(f"{'':10s}{'sum r^2':>14s}{'sum |r|':>14s}")
print(f"{'L2 fit':10s}{sse_l2:14.3f}{sabs_l2:14.3f}")
print(f"{'L1 fit':10s}{sse_l1:14.3f}{sabs_l1:14.3f}")
print("(L2 적합은 sum r^2 가 최소, L1 적합은 sum |r| 가 최소)")
print("=" * 60)


# ── 그리기: 데이터 점 + 두 곡선 ──
mask = np.ones(n, dtype=bool)
mask[outlier_idx] = False
ax.scatter(x_coords[mask], y_coords[mask], color="blue", label="data")
ax.scatter(x_coords[outlier_idx], y_coords[outlier_idx],
           color="orange", s=90, marker="D", zorder=5, label="outlier")

xs = np.linspace(-lim, lim, 300)
M = design(xs)
ax.plot(xs, M @ coef_l2, color="red", lw=2,
        label="L2 fit  (squared error)")
ax.plot(xs, M @ coef_l1, color="green", lw=2, ls="--",
        label="L1 fit  (absolute error)")

ax.set_title("Quadratic Regression:  L2 (squared) vs L1 (absolute)")
ax.legend(loc="lower center", fontsize=10)

plt.show()




