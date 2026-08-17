"""
이차 스플라인 보간 (Quadratic Spline Interpolation)

주어진 점 (x0,y0), (x1,y1), ... , (xn,yn) 을 구간별 2차 다항식으로 잇는다.
각 구간 [x_i, x_{i+1}] 에서:

    S_i(x) = y_i + z_i (x - x_i) + (z_{i+1} - z_i) / (2 h_i) (x - x_i)^2 ,
             h_i = x_{i+1} - x_i

조건:
    - 보간: 각 구간이 양 끝점을 지난다.
    - C1 연속: 이웃 구간의 1차 도함수(기울기)가 매듭에서 일치한다.
      이때 매듭에서의 기울기 z_i 는 다음 점화식으로 구한다.

          z_{i+1} = -z_i + 2 (y_{i+1} - y_i) / h_i

    - 자유도가 하나 남으므로 시작 기울기 z_0 를 하나 지정해야 한다.
      기본값 z_0 = 0 (첫 구간을 '가장 평평하게' 시작) 이며,
      first_slope 인자로 바꿀 수 있다.

digitizer.py 로 저장한 CSV (index, pixel_x, pixel_y[, real_x, real_y]) 를
바로 읽어 곡선을 그릴 수 있다.

사용법:
    python quadratic_spline.py digitized_001.csv
    python quadratic_spline.py digitized_001.csv --real     # 실좌표 컬럼 사용
    python quadratic_spline.py digitized_001.csv --slope 0.0

또는 모듈로:
    from quadratic_spline import QuadraticSpline
    spl = QuadraticSpline(x, y)
    yy = spl(xx)
"""

import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt


class QuadraticSpline:
    """구간별 2차 다항식으로 점들을 보간하는 C1 연속 스플라인."""

    def __init__(self, x, y, first_slope=0.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("x, y 는 길이가 같은 1차원 배열이어야 합니다.")
        if x.size < 2:
            raise ValueError("점이 최소 2개는 필요합니다.")

        # x 오름차순 정렬 (보간이 성립하려면 x 가 단조여야 함)
        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("x 값에 중복이 있으면 안 됩니다 (단조 증가 필요).")

        self.h = np.diff(self.x)                    # 각 구간 폭 h_i
        n = self.x.size

        # 매듭에서의 기울기 z_i 점화식으로 계산
        self.z = np.empty(n)
        self.z[0] = float(first_slope)
        for i in range(n - 1):
            self.z[i + 1] = (-self.z[i]
                             + 2.0 * (self.y[i + 1] - self.y[i]) / self.h[i])

    def __call__(self, xq):
        """질의점 xq (스칼라/배열)에서의 보간값."""
        xq = np.asarray(xq, dtype=float)
        scalar = xq.ndim == 0
        xq = np.atleast_1d(xq)

        # 각 xq 가 속한 구간 i 찾기 (마지막 구간에 오른쪽 끝 포함)
        i = np.searchsorted(self.x, xq, side="right") - 1
        i = np.clip(i, 0, self.x.size - 2)

        dx = xq - self.x[i]
        out = (self.y[i]
               + self.z[i] * dx
               + (self.z[i + 1] - self.z[i]) / (2.0 * self.h[i]) * dx ** 2)

        return float(out[0]) if scalar else out

    def derivative(self, xq):
        """질의점에서의 1차 도함수(기울기)."""
        xq = np.asarray(xq, dtype=float)
        scalar = xq.ndim == 0
        xq = np.atleast_1d(xq)

        i = np.searchsorted(self.x, xq, side="right") - 1
        i = np.clip(i, 0, self.x.size - 2)

        dx = xq - self.x[i]
        out = self.z[i] + (self.z[i + 1] - self.z[i]) / self.h[i] * dx
        return float(out[0]) if scalar else out

    def sample(self, num=400):
        """전체 구간을 촘촘히 샘플링한 (xx, yy) 반환 (그래프용)."""
        xx = np.linspace(self.x[0], self.x[-1], num)
        return xx, self(xx)


def load_csv(path, use_real=False):
    """digitizer.py 형식 CSV 에서 x, y 열을 읽는다.

    use_real=True 이면 real_x/real_y, 아니면 pixel_x/pixel_y 사용.
    """
    xs, ys = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        xcol = "real_x" if use_real else "pixel_x"
        ycol = "real_y" if use_real else "pixel_y"
        if xcol not in reader.fieldnames or ycol not in reader.fieldnames:
            raise KeyError(
                f"CSV 에 '{xcol}','{ycol}' 열이 없습니다. "
                f"열: {reader.fieldnames}")
        for row in reader:
            xs.append(float(row[xcol]))
            ys.append(float(row[ycol]))
    return np.array(xs), np.array(ys)


def main():
    parser = argparse.ArgumentParser(description="이차 스플라인 보간")
    parser.add_argument("csv", help="digitizer.py 로 저장한 CSV 경로")
    parser.add_argument("--real", action="store_true",
                        help="pixel 대신 real 좌표 열 사용")
    parser.add_argument("--slope", type=float, default=0.0,
                        help="시작 기울기 z_0 (기본 0)")
    parser.add_argument("--num", type=int, default=400,
                        help="곡선 샘플 개수")
    args = parser.parse_args()

    x, y = load_csv(args.csv, use_real=args.real)
    spl = QuadraticSpline(x, y, first_slope=args.slope)
    xx, yy = spl.sample(args.num)

    plt.figure(figsize=(11, 7))
    plt.plot(x, y, "ro", ms=7, label="data points")
    plt.plot(xx, yy, "b-", lw=1.5, label="quadratic spline")
    if not args.real:
        # 이미지 픽셀 좌표는 y축이 아래로 증가하므로 뒤집어 보기 좋게
        plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f"Quadratic Spline  ({len(x)} points)  ->  {args.csv}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
