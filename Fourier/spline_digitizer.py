"""
스플라인 디지타이저 (Interactive Spline Digitizer)

이미지를 띄우고 마우스로 점을 찍으면, 찍은 점들을 지나는
매개변수 이차 스플라인 곡선을 실시간으로 그려준다.

하트처럼 닫힌 곡선/함수가 아닌 곡선도 그릴 수 있도록,
x 에 대한 함수가 아니라 '클릭한 순서' t = 0,1,2,... 를 매개변수로
x(t), y(t) 를 각각 이차 스플라인으로 보간한다.

사용법:
    python spline_digitizer.py hearts.jpg
    python spline_digitizer.py hearts.jpg --closed   # 시작점과 끝점을 이어 닫기

조작:
    좌클릭        : 점 추가
    우클릭 / 'u'  : 마지막 점 취소 (undo)
    'c'           : 전체 지우기
    'f'           : 닫힌 곡선 <-> 열린 곡선 토글
    's'           : 점 좌표를 CSV 로 저장
    창 닫기       : 종료
"""

import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from quadratic_spline import QuadraticSpline


class SplineDigitizer:
    def __init__(self, image_path, out_path="spline_points.csv", closed=False):
        self.out_path = out_path
        self.closed = closed
        self.points = []          # 찍은 점들 [(x, y), ...]

        # matplotlib 기본 단축키(s=그림저장, f=전체화면, c=뒤로가기) 와
        # 우리 단축키가 충돌하지 않도록 해당 기본 키맵을 비운다.
        for key in ("keymap.save", "keymap.fullscreen", "keymap.back"):
            plt.rcParams[key] = []

        self.img = mpimg.imread(image_path)
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.ax.imshow(self.img)
        self.ax.set_title(self._title())

        (self.dots,) = self.ax.plot([], [], "ro", ms=7)          # 찍은 점
        (self.curve,) = self.ax.plot([], [], "c-", lw=2)         # 스플라인 곡선

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _title(self):
        mode = "closed" if self.closed else "open"
        return (f"Spline Digitizer [{mode}]  |  L-click: add   "
                "R-click/'u': undo   'c': clear   'f': open/close   "
                "'s': save   'p': plot x(t),y(t)")

    # ── 이벤트 ───────────────────────────────────────────
    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 3:            # 우클릭 = undo
            self.undo()
            return
        self.points.append((event.xdata, event.ydata))
        self._redraw()

    def on_key(self, event):
        if event.key == "u":
            self.undo()
        elif event.key == "c":
            self.points.clear()
            self._redraw()
        elif event.key == "f":
            self.closed = not self.closed
            self.ax.set_title(self._title())
            self._redraw()
        elif event.key == "s":
            self.save()
        elif event.key == "p":
            self.plot_functions()

    def undo(self):
        if self.points:
            self.points.pop()
            self._redraw()

    # ── 스플라인 계산/그리기 ─────────────────────────────
    def _build_splines(self):
        """찍은 점으로부터 x(t), y(t) 이차 스플라인과 매듭 매개변수 t 를 만든다.

        매개변수는 코드길이(chord length) 를 누적해서 쓴다. 점 간격이
        들쭉날쭉해도 등간격보다 오버슈트가 덜하다.
        점이 2개 미만이면 (None, None, None) 을 반환한다.
        """
        pts = list(self.points)
        if self.closed and len(pts) >= 3:
            pts = pts + [pts[0]]          # 시작점으로 되돌아와 닫기

        if len(pts) < 2:
            return None, None, None

        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])

        # 코드길이 매개변수: t_{i+1} = t_i + |P_{i+1}-P_i|
        d = np.hypot(np.diff(xs), np.diff(ys))
        d[d == 0] = 1e-9                  # 중복점 방지
        t = np.concatenate([[0.0], np.cumsum(d)])

        sx = QuadraticSpline(t, xs)
        sy = QuadraticSpline(t, ys)
        return sx, sy, t

    def _spline_curve(self, num=None):
        """스플라인을 촘촘히 샘플링한 (tt, xx, yy) 반환. 점 부족 시 빈 배열."""
        sx, sy, t = self._build_splines()
        if sx is None:
            return np.array([]), np.array([]), np.array([])
        if num is None:
            num = max(200, len(t) * 40)
        tt = np.linspace(t[0], t[-1], num)
        return tt, sx(tt), sy(tt)

    def _redraw(self):
        if self.points:
            px, py = zip(*self.points)
        else:
            px, py = [], []
        self.dots.set_data(px, py)

        _, xx, yy = self._spline_curve()
        self.curve.set_data(xx, yy)
        self.fig.canvas.draw_idle()

    # ── 저장 ─────────────────────────────────────────────
    def save(self):
        if not self.points:
            print("저장할 점이 없습니다.")
            return
        # 1) 찍은 점(제어점) 저장
        with open(self.out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "pixel_x", "pixel_y"])
            for i, (x, y) in enumerate(self.points):
                writer.writerow([i, f"{x:.2f}", f"{y:.2f}"])
        print(f"저장 완료: {self.out_path} ({len(self.points)} points)")

        # 2) 스플라인 곡선 x(t), y(t) 샘플 저장
        tt, xx, yy = self._spline_curve()
        if len(tt):
            curve_path = self._curve_path()
            with open(curve_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "x", "y"])
                for tv, xv, yv in zip(tt, xx, yy):
                    writer.writerow([f"{tv:.4f}", f"{xv:.3f}", f"{yv:.3f}"])
            print(f"곡선 저장 완료: {curve_path} ({len(tt)} samples)")

    def _curve_path(self):
        """제어점 파일명에서 곡선 샘플 파일명(_curve) 생성."""
        base = self.out_path
        if base.lower().endswith(".csv"):
            base = base[:-4]
        return base + "_curve.csv"

    def plot_functions(self):
        """x(t), y(t) 를 함수 그래프로, 그리고 복원한 (x,y) 곡선을 새 창에 그린다."""
        tt, xx, yy = self._spline_curve()
        if not len(tt):
            print("그릴 점이 부족합니다 (2개 이상 필요).")
            return

        _, _, tk = self._build_splines()      # 매듭 위치 표시용

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.plot(tt, xx, "b-")
        ax1.plot(tk, [self.points[i % len(self.points)][0]
                      for i in range(len(tk))], "ko", ms=4)
        ax1.set_title("x(t)")
        ax1.set_xlabel("t (chord length)")
        ax1.set_ylabel("x")
        ax1.grid(True, alpha=0.3)

        ax2.plot(tt, yy, "r-")
        ax2.plot(tk, [self.points[i % len(self.points)][1]
                      for i in range(len(tk))], "ko", ms=4)
        ax2.set_title("y(t)")
        ax2.set_xlabel("t (chord length)")
        ax2.set_ylabel("y")
        ax2.grid(True, alpha=0.3)

        px, py = zip(*self.points)
        ax3.plot(xx, yy, "c-", lw=2)
        ax3.plot(px, py, "ro", ms=5)
        ax3.set_title("(x(t), y(t)) curve")
        ax3.set_aspect("equal", adjustable="datalim")
        ax3.invert_yaxis()                    # 이미지 픽셀 좌표계
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.show()

    def run(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="인터랙티브 스플라인 디지타이저")
    parser.add_argument("image", nargs="?", default="hearts.jpg",
                        help="배경 이미지 경로")
    parser.add_argument("--out", default="spline_points.csv",
                        help="저장할 CSV 파일명")
    parser.add_argument("--closed", action="store_true",
                        help="시작점과 끝점을 이어 닫힌 곡선으로")
    args = parser.parse_args()

    try:
        dig = SplineDigitizer(args.image, args.out, closed=args.closed)
    except FileNotFoundError:
        print(f"이미지를 찾을 수 없습니다: {args.image}")
        sys.exit(1)

    print("점을 찍으면 스플라인 곡선이 실시간으로 그려집니다. "
          "'f' 닫기/열기, 's' 저장(제어점+곡선), 'p' x(t)·y(t) 그래프.")
    dig.run()


if __name__ == "__main__":
    main()
