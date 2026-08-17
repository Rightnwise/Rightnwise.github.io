"""
이차 베지어 곡선 에디터 (Interactive Quadratic Bézier Editor)

이차 베지어는 점 3개로 정의된다:
    시작점 A0, 방향점(control) C, 끝점 A1
    B(u) = (1-u)^2 A0 + 2(1-u)u C + u^2 A1 ,  u ∈ [0, 1]

방향점 C 가 곡선이 '어느 쪽으로 휘는지' 를 정한다.
여러 구간을 이어 붙이면 이차 베지어 스플라인이 된다.

이 툴에서 할 수 있는 것
----------------------
- 빈 곳 좌클릭      : 앵커(on-curve 점) 추가. 새 구간의 방향점은 중점에 자동 생성
- 앵커/방향점 드래그 : 빨간 원(앵커), 초록 사각(방향점) 을 잡아 끌어 곡선 조절
- 우클릭            : 가장 가까운 앵커 삭제
- 'f'              : 닫힌 곡선 <-> 열린 곡선
- 'c'              : 전체 지우기
- 'p'              : 현재 곡선의 x(t), y(t) 그래프 (드래그로 바뀐 게 반영됨)
- 's'              : 앵커/방향점, 그리고 등간격 x(t),y(t) (curve_txy.csv) 저장

방향점을 움직이면 곡선 모양이 바뀌고, 그에 따라 매개변수화 x(t), y(t) 도
바로 다시 계산된다 (푸리에 단계로 그대로 넘길 수 있음).

사용법:
    python bezier_editor.py hearts.jpg
    python bezier_editor.py hearts.jpg --closed
"""

import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


PICK_RADIUS_PX = 12          # 핸들을 잡았다고 볼 화면상 거리(픽셀)


def quad_bezier(a0, c, a1, u):
    """이차 베지어. a0,c,a1: (2,) 배열, u: 스칼라/배열 -> (len(u),2)."""
    u = np.atleast_1d(np.asarray(u, dtype=float))[:, None]
    return (1 - u) ** 2 * a0 + 2 * (1 - u) * u * c + u ** 2 * a1


class BezierEditor:
    def __init__(self, image_path, out_path="spline_points.csv", closed=False):
        self.out_path = out_path
        self.closed = closed
        self.anchors = []        # on-curve 점 [(x,y), ...]
        self.controls = []       # 구간별 방향점 [(x,y), ...]  (구간 수와 동기화)

        self.drag = None         # ('anchor'|'control', index) 드래그 중
        self._press_xy = None    # 클릭-드래그 구분용

        self.img = mpimg.imread(image_path)
        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.ax.imshow(self.img)
        self.ax.set_title(self._title())

        # matplotlib 기본 단축키(s,f,c) 충돌 제거
        for key in ("keymap.save", "keymap.fullscreen", "keymap.back"):
            plt.rcParams[key] = []

        (self.curve,) = self.ax.plot([], [], "-", color="cyan", lw=2, zorder=1)
        (self.handles,) = self.ax.plot([], [], "--", color="0.6", lw=0.8,
                                       zorder=2)
        (self.anchor_pts,) = self.ax.plot([], [], "o", color="red", ms=8,
                                          zorder=4)
        (self.control_pts,) = self.ax.plot([], [], "s", color="lime", ms=7,
                                           mec="green", zorder=4)

        self.func_fig = None     # x(t),y(t) 그래프 창 (지연 생성)

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _title(self):
        mode = "closed" if self.closed else "open"
        return (f"Quadratic Bézier [{mode}]  |  L-click: add anchor   "
                "drag red=anchor / green=control   R-click: delete   "
                "'f' open/close  'c' clear  'p' plot x(t),y(t)  's' save")

    # ── 구간/방향점 관리 ─────────────────────────────────
    def _num_segments(self):
        n = len(self.anchors)
        if n < 2:
            return 0
        return n if self.closed else n - 1

    def _seg_anchor_indices(self, seg):
        """구간 seg 의 (시작앵커 idx, 끝앵커 idx)."""
        n = len(self.anchors)
        return seg, (seg + 1) % n

    def _add_anchor(self, x, y):
        """앵커를 끝에 추가한다. 닫힌 곡선이면 '직전 닫힘 구간' 의 방향점을
        직전 앵커->새 앵커 기준으로 갱신해, 새 점이 엉뚱하게 첫 점 쪽으로
        휘지 않도록 한다."""
        prev = len(self.anchors)
        self.anchors.append((x, y))
        if self.closed and prev >= 2:
            # prev-1 번 구간은 (직전에는 마지막앵커->첫앵커 였다가) 이제
            # 직전앵커->새앵커 로 바뀌므로 방향점을 그 중점으로 재생성
            m = prev                       # 새 앵커의 인덱스
            (ax0, ay0), (ax1, ay1) = self.anchors[m - 1], self.anchors[m]
            if len(self.controls) >= m:
                self.controls[m - 1] = ((ax0 + ax1) / 2, (ay0 + ay1) / 2)
        self._sync_controls()              # 새로 생긴 구간의 방향점 채우기
        self._redraw()

    def _sync_controls(self):
        """구간 수에 맞춰 방향점 개수를 맞춘다(기존 것은 유지, 부족분은 중점)."""
        need = self._num_segments()
        # 부족하면 각 구간 중점으로 추가
        while len(self.controls) < need:
            seg = len(self.controls)
            i0, i1 = self._seg_anchor_indices(seg)
            ax0, ay0 = self.anchors[i0]
            ax1, ay1 = self.anchors[i1]
            self.controls.append(((ax0 + ax1) / 2, (ay0 + ay1) / 2))
        # 남으면 잘라냄
        if len(self.controls) > need:
            self.controls = self.controls[:need]

    # ── 곡선 계산 ────────────────────────────────────────
    def _dense_curve(self, per_seg=60):
        """모든 구간을 촘촘히 샘플링한 (xs, ys) 배열 (앵커 중복 제거)."""
        segs = self._num_segments()
        if segs == 0:
            return np.array([]), np.array([])
        u = np.linspace(0, 1, per_seg)
        xs, ys = [], []
        for seg in range(segs):
            i0, i1 = self._seg_anchor_indices(seg)
            a0 = np.array(self.anchors[i0])
            a1 = np.array(self.anchors[i1])
            c = np.array(self.controls[seg])
            b = quad_bezier(a0, c, a1, u)
            if seg > 0:                    # 구간 이음매 중복 앵커 제거
                b = b[1:]
            xs.append(b[:, 0])
            ys.append(b[:, 1])
        return np.concatenate(xs), np.concatenate(ys)

    def parametrize(self, N=512):
        """현재 곡선 -> 등간격 (t, x, y, z=x+iy). 방향점이 바뀌면 결과도 바뀜."""
        xs, ys = self._dense_curve(per_seg=80)
        if xs.size < 2:
            return None
        # 코드길이 기반 등간격 재샘플
        d = np.hypot(np.diff(xs), np.diff(ys))
        s = np.concatenate([[0.0], np.cumsum(d)])
        total = s[-1]
        if total == 0:
            return None
        if self.closed:
            sq = np.linspace(0, total, N, endpoint=False)
        else:
            sq = np.linspace(0, total, N)
        x = np.interp(sq, s, xs)
        y = np.interp(sq, s, ys)
        t = sq / total * 2 * np.pi
        return t, x, y, x + 1j * y

    # ── 그리기 ──────────────────────────────────────────
    def _redraw(self):
        self._sync_controls()

        # 곡선
        xs, ys = self._dense_curve()
        self.curve.set_data(xs, ys)

        # 앵커
        if self.anchors:
            ax_, ay_ = zip(*self.anchors)
        else:
            ax_, ay_ = [], []
        self.anchor_pts.set_data(ax_, ay_)

        # 방향점 + 방향점-앵커 연결선
        if self.controls:
            cx_, cy_ = zip(*self.controls)
        else:
            cx_, cy_ = [], []
        self.control_pts.set_data(cx_, cy_)

        hx, hy = [], []
        for seg in range(self._num_segments()):
            i0, i1 = self._seg_anchor_indices(seg)
            cx, cy = self.controls[seg]
            for (ax0, ay0) in (self.anchors[i0], self.anchors[i1]):
                hx += [ax0, cx, np.nan]
                hy += [ay0, cy, np.nan]
        self.handles.set_data(hx, hy)

        self.fig.canvas.draw_idle()
        if self.func_fig is not None:
            self._update_func_plot()

    # ── 핸들 찾기(드래그용) ──────────────────────────────
    def _find_handle(self, event):
        """클릭 위치에서 가장 가까운 앵커/방향점 핸들 반환 (없으면 None)."""
        if event.x is None:
            return None
        best = None
        best_d = PICK_RADIUS_PX
        # 화면 픽셀 좌표로 거리 비교
        for kind, pts in (("control", self.controls), ("anchor", self.anchors)):
            for i, (px, py) in enumerate(pts):
                dx, dy = self.ax.transData.transform((px, py))
                dist = np.hypot(dx - event.x, dy - event.y)
                if dist < best_d:
                    best_d = dist
                    best = (kind, i)
        return best

    # ── 이벤트 ──────────────────────────────────────────
    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if event.button == 3:              # 우클릭 = 앵커 삭제
            self._delete_nearest_anchor(event)
            return
        self._press_xy = (event.x, event.y)
        self.drag = self._find_handle(event)   # 핸들 잡았으면 드래그 시작

    def on_motion(self, event):
        if self.drag is None or event.xdata is None or event.inaxes != self.ax:
            return
        kind, i = self.drag
        if kind == "anchor":
            self.anchors[i] = (event.xdata, event.ydata)
        else:
            self.controls[i] = (event.xdata, event.ydata)
        self._redraw()

    def on_release(self, event):
        # 핸들을 안 잡고, 거의 안 움직였으면 -> 새 앵커 추가(클릭)
        if (self.drag is None and self._press_xy is not None
                and event.xdata is not None and event.button == 1):
            moved = np.hypot(event.x - self._press_xy[0],
                             event.y - self._press_xy[1])
            if moved < 4:
                self._add_anchor(event.xdata, event.ydata)
        self.drag = None
        self._press_xy = None

    def on_key(self, event):
        if event.key == "c":
            self.anchors.clear()
            self.controls.clear()
            self._redraw()
        elif event.key == "f":
            self.closed = not self.closed
            self.ax.set_title(self._title())
            self._redraw()
        elif event.key == "p":
            self._open_func_plot()
        elif event.key == "s":
            self.save()

    def _delete_nearest_anchor(self, event):
        if not self.anchors:
            return
        ds = [np.hypot(*(np.subtract(self.ax.transData.transform((px, py)),
                                     (event.x, event.y))))
              for (px, py) in self.anchors]
        i = int(np.argmin(ds))
        if ds[i] < PICK_RADIUS_PX * 2:
            self.anchors.pop(i)
            self.controls.clear()          # 구간이 바뀌므로 방향점 재생성
            self._redraw()

    # ── x(t), y(t) 그래프 ────────────────────────────────
    def _open_func_plot(self):
        if self.func_fig is None:
            self.func_fig, (self.ax_x, self.ax_y) = plt.subplots(
                2, 1, figsize=(7, 6), sharex=True)
            self.func_fig.canvas.mpl_connect(
                "close_event", self._on_func_close)
        self._update_func_plot()
        self.func_fig.show()

    def _on_func_close(self, event):
        self.func_fig = None

    def _update_func_plot(self):
        res = self.parametrize()
        self.ax_x.clear()
        self.ax_y.clear()
        if res is not None:
            t, x, y, _ = res
            self.ax_x.plot(t, x, "b-")
            self.ax_y.plot(t, y, "r-")
        self.ax_x.set_ylabel("x(t)")
        self.ax_x.grid(True, alpha=0.3)
        self.ax_y.set_ylabel("y(t)")
        self.ax_y.set_xlabel("t  [0, 2π)")
        self.ax_y.grid(True, alpha=0.3)
        self.ax_x.set_title("x(t), y(t)  (방향점 드래그하면 실시간 반영)")
        self.func_fig.canvas.draw_idle()

    # ── 저장 ─────────────────────────────────────────────
    def save(self):
        if not self.anchors:
            print("저장할 점이 없습니다.")
            return
        # 앵커 + 방향점 저장
        with open(self.out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["kind", "index", "x", "y"])
            for i, (x, y) in enumerate(self.anchors):
                writer.writerow(["anchor", i, f"{x:.2f}", f"{y:.2f}"])
            for i, (x, y) in enumerate(self.controls):
                writer.writerow(["control", i, f"{x:.2f}", f"{y:.2f}"])
        print(f"저장 완료: {self.out_path} "
              f"(anchors {len(self.anchors)}, controls {len(self.controls)})")

        # 등간격 x(t), y(t) 저장 (푸리에 단계로 넘기기용)
        res = self.parametrize()
        if res is not None:
            t, x, y, _ = res
            with open("curve_txy.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "x", "y"])
                for tv, xv, yv in zip(t, x, y):
                    writer.writerow([f"{tv:.6f}", f"{xv:.4f}", f"{yv:.4f}"])
            print(f"곡선 저장 완료: curve_txy.csv ({len(t)} samples)")

    def run(self):
        self._redraw()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="이차 베지어 곡선 에디터")
    parser.add_argument("image", nargs="?", default="hearts.jpg",
                        help="배경 이미지 경로")
    parser.add_argument("--out", default="spline_points.csv",
                        help="앵커/방향점 저장 파일명")
    parser.add_argument("--closed", action="store_true",
                        help="닫힌 곡선으로 시작")
    args = parser.parse_args()

    try:
        editor = BezierEditor(args.image, args.out, closed=args.closed)
    except FileNotFoundError:
        print(f"이미지를 찾을 수 없습니다: {args.image}")
        sys.exit(1)

    print("좌클릭으로 앵커 추가. 빨간 원=앵커, 초록 사각=방향점을 드래그해 조절. "
          "'p' 로 x(t),y(t) 확인, 's' 로 저장.")
    editor.run()


if __name__ == "__main__":
    main()
