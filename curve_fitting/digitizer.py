"""
이미지 디지타이저 (Image Digitizer)

이미지를 띄우고 마우스로 점을 찍어 좌표를 CSV 로 저장한다.
예: 다리 케이블(현수선) 곡선을 따라 점을 찍어 좌표화.

사용법:
    python digitizer.py [이미지경로] [--calibrate] [--out 파일명.csv]

조작:
    좌클릭        : 점 추가
    우클릭 / 'u'  : 마지막 점 취소 (undo)
    'c'           : 전체 지우기 (clear)
    's'           : 지금까지의 점 저장 (save)
    'n'           : 현재 점 저장 후 '새 파일'로 넘어가 계속 디지타이즈 (next)
    창 닫기       : 저장 후 종료

여러 파일 만들기:
    --out 을 주지 않으면 digitized_001.csv, digitized_002.csv ... 처럼
    기존 파일을 덮어쓰지 않고 다음 번호로 자동 저장한다.
    한 세션에서 'n' 을 누르면 현재까지를 저장하고 새 번호 파일로 이어서 찍을 수 있다.

--calibrate 옵션:
    실제 좌표(예: 미터)로 변환하려면 먼저 기준점 4개를 순서대로 클릭한다.
    X축 기준점 2개 -> 각 실제 X값 입력, Y축 기준점 2개 -> 각 실제 Y값 입력.
    이후 찍는 점은 픽셀 + 보정된 실좌표가 함께 저장된다.
"""

import os
import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def next_filename(base="digitized", ext=".csv"):
    """기존 파일을 덮어쓰지 않는 다음 번호 파일명 반환.

    digitized_001.csv, digitized_002.csv ... 중 비어 있는 첫 번호.
    """
    i = 1
    while True:
        candidate = f"{base}_{i:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


class Digitizer:
    def __init__(self, image_path, out_path, calibrate=False):
        self.image_path = image_path
        self.out_path = out_path
        self.calibrate = calibrate

        self.points = []          # 찍은 점들의 픽셀 좌표 [(x, y), ...]
        self.calib_px = []        # 보정 기준점의 픽셀 좌표
        self.calib_real = []      # 보정 기준점의 실제 좌표 (x 또는 y 값)
        self.transform = None     # 픽셀 -> 실좌표 변환 (a, b) per axis

        # 이미지 로드
        self.img = mpimg.imread(image_path)
        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.ax.imshow(self.img)
        self.ax.set_title(self._title())

        # 그림 객체
        (self.line,) = self.ax.plot([], [], "r.-", lw=1, ms=8, picker=False)

        # 보정 단계 관리
        self.calib_stage = 0      # 0~3: x1,x2,y1,y2 기준점 클릭 단계
        if self.calibrate:
            self._prompt_calibration_intro()

        # 이벤트 연결
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    # ── 화면/안내 ─────────────────────────────────────────
    def _title(self):
        return (f"Digitizer -> {self.out_path}  |  L-click: add   "
                "R-click/'u': undo   'c': clear   's': save   "
                "'n': save & new file   close: save & quit")

    def _prompt_calibration_intro(self):
        print("\n[Calibration] 4개의 기준점을 순서대로 클릭하세요:")
        print("  1) X-axis point #1")
        print("  2) X-axis point #2")
        print("  3) Y-axis point #1")
        print("  4) Y-axis point #2")

    # ── 이벤트 핸들러 ────────────────────────────────────
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # 우클릭(3) = undo
        if event.button == 3:
            self.undo()
            return

        x, y = event.xdata, event.ydata

        # 보정 단계 진행 중이면 기준점으로 처리
        if self.calibrate and self.calib_stage < 4:
            self._add_calibration_point(x, y)
            return

        # 일반 점 추가
        self.points.append((x, y))
        self._redraw()
        print(f"point {len(self.points):3d}: pixel=({x:.1f}, {y:.1f})"
              + self._real_str(x, y))

    def on_key(self, event):
        if event.key == "u":
            self.undo()
        elif event.key == "c":
            self.points.clear()
            self._redraw()
            print("cleared all points.")
        elif event.key == "s":
            self.save()
        elif event.key == "n":
            self.save_and_new()

    def save_and_new(self):
        """현재 점을 저장하고 새 번호 파일로 넘어가 계속 디지타이즈."""
        self.save()
        self.points.clear()
        self._redraw()
        self.out_path = next_filename()
        self.ax.set_title(self._title())
        self.fig.canvas.draw_idle()
        print(f"새 파일로 전환: {self.out_path} (계속 점을 찍으세요)")

    def on_close(self, event):
        # 창을 닫으면 자동 저장
        if self.points:
            self.save()

    # ── 보정 ─────────────────────────────────────────────
    def _add_calibration_point(self, x, y):
        labels = ["X-axis #1", "X-axis #2", "Y-axis #1", "Y-axis #2"]
        label = labels[self.calib_stage]
        # 실제 좌표값 입력 (터미널)
        axis = "x" if self.calib_stage < 2 else "y"
        val = float(input(f"  [{label}] pixel=({x:.1f},{y:.1f}) "
                          f"-> 실제 {axis} 값 입력: "))
        self.calib_px.append((x, y))
        self.calib_real.append(val)
        self.ax.plot(x, y, "b+", ms=14, mew=2)
        self.ax.annotate(label, (x, y), color="blue",
                         textcoords="offset points", xytext=(6, 6))
        self.fig.canvas.draw()
        self.calib_stage += 1
        if self.calib_stage == 4:
            self._finalize_calibration()

    def _finalize_calibration(self):
        # X: 픽셀 x -> 실제 x (선형),  Y: 픽셀 y -> 실제 y (선형)
        (px1, _), (px2, _) = self.calib_px[0], self.calib_px[1]
        rx1, rx2 = self.calib_real[0], self.calib_real[1]
        (_, py1), (_, py2) = self.calib_px[2], self.calib_px[3]
        ry1, ry2 = self.calib_real[2], self.calib_real[3]

        ax_ = (rx2 - rx1) / (px2 - px1)
        bx_ = rx1 - ax_ * px1
        ay_ = (ry2 - ry1) / (py2 - py1)
        by_ = ry1 - ay_ * py1
        self.transform = (ax_, bx_, ay_, by_)
        print("[Calibration] 완료. 이제 점을 찍으면 실좌표가 함께 저장됩니다.\n")

    def _pixel_to_real(self, x, y):
        if self.transform is None:
            return None
        ax_, bx_, ay_, by_ = self.transform
        return ax_ * x + bx_, ay_ * y + by_

    def _real_str(self, x, y):
        real = self._pixel_to_real(x, y)
        if real is None:
            return ""
        return f"  real=({real[0]:.3f}, {real[1]:.3f})"

    # ── 보조 ─────────────────────────────────────────────
    def undo(self):
        if self.points:
            removed = self.points.pop()
            self._redraw()
            print(f"undo: removed ({removed[0]:.1f}, {removed[1]:.1f})")

    def _redraw(self):
        if self.points:
            xs, ys = zip(*self.points)
        else:
            xs, ys = [], []
        self.line.set_data(xs, ys)
        self.fig.canvas.draw_idle()

    def save(self):
        if not self.points:
            print("저장할 점이 없습니다.")
            return
        with open(self.out_path, "w", newline="") as f:
            writer = csv.writer(f)
            if self.transform is not None:
                writer.writerow(["index", "pixel_x", "pixel_y",
                                 "real_x", "real_y"])
                for i, (x, y) in enumerate(self.points):
                    rx, ry = self._pixel_to_real(x, y)
                    writer.writerow([i, f"{x:.2f}", f"{y:.2f}",
                                     f"{rx:.4f}", f"{ry:.4f}"])
            else:
                writer.writerow(["index", "pixel_x", "pixel_y"])
                for i, (x, y) in enumerate(self.points):
                    writer.writerow([i, f"{x:.2f}", f"{y:.2f}"])
        print(f"저장 완료: {self.out_path} ({len(self.points)} points)")

    def run(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="이미지 디지타이저")
    parser.add_argument("image", nargs="?", default="bridge.jpg",
                        help="디지타이즈할 이미지 경로")
    parser.add_argument("--out", default=None,
                        help="저장할 CSV 파일명 (생략 시 digitized_NNN.csv 자동 번호)")
    parser.add_argument("--calibrate", action="store_true",
                        help="실좌표 보정 모드 사용")
    args = parser.parse_args()

    out_path = args.out if args.out else next_filename()

    try:
        dig = Digitizer(args.image, out_path, calibrate=args.calibrate)
    except FileNotFoundError:
        print(f"이미지를 찾을 수 없습니다: {args.image}")
        print("사용법: python digitizer.py [이미지경로] [--calibrate] "
              "[--out out.csv]")
        sys.exit(1)

    print("디지타이저 시작. 좌클릭으로 점을 찍으세요. 도움말은 창 제목 참고.")
    dig.run()


if __name__ == "__main__":
    main()
