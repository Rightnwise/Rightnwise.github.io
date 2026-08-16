"""
인터랙티브 커브 피팅 (현수선 / 포물선 / 6차 다항식)

fit_compare.py + overlay_fit.py 를 하나로 합친 인터랙티브 버전.

화면 구성:
    (위)  다리 사진 위에 디지타이즈된 점 + 피팅 곡선 오버레이
    (아래) 선택된 모델들의 잔차(오차) 비교

조작 (터미널 옵션 대신 화면에서 직접):
    - 왼쪽 체크박스 : 보고 싶은 모델(catenary/parabola/degree-6) 켜고 끄기
    - 사진에서 박스 드래그 : 그 안의 점만 선택 -> 선택된 점만으로 다시 피팅
    - 'Reset' 버튼 : 전체 점으로 복귀

모델(요청한 3개 함수 유지):
    catenary  f(x) = C cosh((x-a)/C) + d        (Levenberg-Marquardt)
    parabola  f(x) = C (x-a)^2 + d               (다항 최소제곱)
    poly6     f(x) = sum_{i=0}^{6} a_i x^i        (6차 다항 최소제곱)

사용법:
    python curve_fit_interactive.py [--image 경로] [--csv 경로]
"""

import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import CheckButtons, Button, RectangleSelector

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ════════════════════════════════════════════════════════
#  모델 3개 (피팅 함수 유지)
# ════════════════════════════════════════════════════════
def catenary(x, C, a, d):
    return C * np.cosh((x - a) / C) + d


def _catenary_jacobian(x, C, a, d):
    u = (x - a) / C
    return np.column_stack([
        np.cosh(u) - u * np.sinh(u),   # df/dC
        -np.sinh(u),                   # df/da
        np.ones_like(x),               # df/dd
    ])


def _catenary_initial(x, y):
    b2, b1, b0 = np.polyfit(x, y, 2)
    if abs(b2) < 1e-12:
        b2 = 1e-12
    a0 = -b1 / (2 * b2)
    C0 = 1.0 / (2 * b2)
    d0 = (b0 - b1 ** 2 / (4 * b2)) - C0
    return np.array([C0, a0, d0], dtype=float)


def fit_catenary(x, y, max_iter=200, tol=1e-10):
    """현수선 f(x)=C cosh((x-a)/C)+d 를 Levenberg-Marquardt 로 피팅."""
    p = _catenary_initial(x, y)
    lam = 1e-3

    def sse(pp):
        r = y - catenary(x, *pp)
        return r @ r

    S = sse(p)
    for _ in range(max_iter):
        C, a, d = p
        r = y - catenary(x, C, a, d)
        J = _catenary_jacobian(x, C, a, d)
        JTJ = J.T @ J
        JTr = J.T @ r
        improved = False
        for _ in range(30):
            A = JTJ + lam * np.diag(np.diag(JTJ))
            try:
                delta = np.linalg.solve(A, JTr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            p_new = p + delta
            S_new = sse(p_new)
            if S_new < S:
                p = p_new
                lam = max(lam * 0.5, 1e-12)
                improved = True
                if abs(S - S_new) < tol * max(S, 1e-30):
                    return p, S_new
                S = S_new
                break
            lam *= 10
        if not improved:
            break
    return p, S


def parabola(x, C, a, d):
    return C * (x - a) ** 2 + d


def fit_parabola(x, y):
    """포물선 f(x)=C(x-a)^2+d 의 최소제곱 닫힌 해."""
    b2, b1, b0 = np.polyfit(x, y, 2)
    if abs(b2) < 1e-15:
        b2 = 1e-15
    C = b2
    a = -b1 / (2 * b2)
    d = b0 - C * a ** 2
    p = np.array([C, a, d])
    r = y - parabola(x, *p)
    return p, float(r @ r)


def fit_poly6(x, y):
    """6차 다항식 f(x)=sum a_i x^i (도메인 스케일로 안정적 최소제곱)."""
    series = np.polynomial.Polynomial.fit(x, y, 6)
    r = y - series(x)
    return series, float(r @ r)


# ════════════════════════════════════════════════════════
#  CSV 로드
# ════════════════════════════════════════════════════════
def load_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    cols = {name: i for i, name in enumerate(header)}
    if "real_x" in cols and "real_y" in cols:
        xi, yi = cols["real_x"], cols["real_y"]
    elif "pixel_x" in cols and "pixel_y" in cols:
        xi, yi = cols["pixel_x"], cols["pixel_y"]
    else:
        xi, yi = len(header) - 2, len(header) - 1
    x = np.array([float(r[xi]) for r in rows[1:] if r])
    y = np.array([float(r[yi]) for r in rows[1:] if r])
    return x, y


# ════════════════════════════════════════════════════════
#  인터랙티브 앱
# ════════════════════════════════════════════════════════
MODEL_KEYS = ["catenary", "parabola", "poly6"]
MODEL_LABELS = {"catenary": "catenary",
                "parabola": "parabola",
                "poly6": "degree-6 poly"}
MODEL_COLORS = {"catenary": "red", "parabola": "cyan", "poly6": "yellow"}
MODEL_DASH = {"catenary": "-", "parabola": "--", "poly6": "-."}
MIN_POINTS = {"catenary": 3, "parabola": 3, "poly6": 7}


class CurveFitApp:
    def __init__(self, image_path, csv_path):
        self.x, self.y = load_csv(csv_path)
        self.img = mpimg.imread(image_path)
        self.csv_path = csv_path

        # 선택 마스크 (기본: 전체 점 사용)
        self.mask = np.ones(len(self.x), dtype=bool)
        # 활성 모델 (기본: 셋 다)
        self.active = {k: True for k in MODEL_KEYS}
        self.fits = {}   # key -> dict(curve_x, curve_y, resid_x, resid, rmse)

        self._build_layout()
        self.compute_fits()
        self.redraw()

    # ── 레이아웃 / 위젯 ─────────────────────────────────
    def _build_layout(self):
        self.fig = plt.figure(figsize=(15, 10))
        # 범례를 오른쪽 바깥에 둘 공간을 남기려고 폭을 줄임
        self.ax_img = self.fig.add_axes([0.26, 0.40, 0.58, 0.55])
        self.ax_res = self.fig.add_axes([0.26, 0.07, 0.58, 0.25])

        # 모델 체크박스
        ax_check = self.fig.add_axes([0.02, 0.60, 0.22, 0.22])
        ax_check.set_title("models", fontsize=10)
        self.check = CheckButtons(
            ax_check,
            [MODEL_LABELS[k] for k in MODEL_KEYS],
            [self.active[k] for k in MODEL_KEYS],
        )
        self.check.on_clicked(self._on_check)

        # Reset 버튼
        ax_btn = self.fig.add_axes([0.02, 0.50, 0.22, 0.06])
        self.btn_reset = Button(ax_btn, "Reset (use all points)")
        self.btn_reset.on_clicked(self._on_reset)

        # 안내 텍스트
        self.fig.text(
            0.02, 0.40,
            "사용법:\n"
            "• 체크박스로 모델 on/off\n"
            "• 사진에서 박스 드래그 →\n"
            "  그 안의 점만으로 피팅\n"
            "• Reset = 전체 점 복귀",
            fontsize=9, va="top")

        # 박스 드래그 선택기 (사진 위)
        self.selector = RectangleSelector(
            self.ax_img, self._on_box_select,
            useblit=True, button=[1], interactive=True,
            minspanx=5, minspany=5, spancoords="pixels")

    # ── 피팅 계산 (현재 마스크 기준) ────────────────────
    def compute_fits(self):
        self.fits = {}
        xm, ym = self.x[self.mask], self.y[self.mask]
        n = len(xm)
        if n < 3:
            return
        # 피팅은 선택된 점으로 하지만, 곡선은 '전체' x 범위에 그려
        # 일부만 선택해도 내려갔다 올라가는 전체 형태가 보이게 한다.
        full_span = self.x.max() - self.x.min()
        pad = 0.02 * full_span if full_span > 0 else 1.0
        xs = np.linspace(self.x.min() - pad, self.x.max() + pad, 400)

        for key in MODEL_KEYS:
            if n < MIN_POINTS[key]:
                continue
            if key == "catenary":
                p, S = fit_catenary(xm, ym)
                curve_y = catenary(xs, *p)
                resid = ym - catenary(xm, *p)
            elif key == "parabola":
                p, S = fit_parabola(xm, ym)
                curve_y = parabola(xs, *p)
                resid = ym - parabola(xm, *p)
            else:  # poly6
                series, S = fit_poly6(xm, ym)
                curve_y = series(xs)
                resid = ym - series(xm)
            self.fits[key] = dict(
                curve_x=xs, curve_y=curve_y,
                resid_x=xm, resid=resid,
                rmse=np.sqrt(S / n))

    # ── 그리기 ──────────────────────────────────────────
    def redraw(self):
        active_keys = [k for k in MODEL_KEYS
                       if self.active[k] and k in self.fits]
        multi = len(active_keys) > 1

        # (위) 사진 + 점 + 곡선
        self.ax_img.cla()
        self.ax_img.imshow(self.img)
        # 선택/비선택 점 구분
        self.ax_img.plot(self.x[~self.mask], self.y[~self.mask],
                         "o", color="gray", ms=3, alpha=0.5,
                         label="unselected")
        self.ax_img.plot(self.x[self.mask], self.y[self.mask],
                         "o", color="lime", ms=4, label="selected points")
        for key in active_keys:
            f = self.fits[key]
            ls = MODEL_DASH[key] if multi else "-"
            self.ax_img.plot(f["curve_x"], f["curve_y"],
                             color=MODEL_COLORS[key], ls=ls, lw=2,
                             label=f"{MODEL_LABELS[key]} "
                                   f"(RMSE={f['rmse']:.3g} px)")
        # 축을 사진 크기에 고정 → 외삽으로 곡선이 폭발해도 화면 밖으로 잘림
        h, w = self.img.shape[:2]
        self.ax_img.set_xlim(0, w)
        self.ax_img.set_ylim(h, 0)   # 픽셀 y 는 아래로 증가
        nsel = int(self.mask.sum())
        self.ax_img.set_title(
            f"Curve fitting overlay  |  selected {nsel}/{len(self.x)} points")
        self.ax_img.set_xlabel("pixel x")
        self.ax_img.set_ylabel("pixel y")
        # 범례는 그래프를 가리지 않게 오른쪽 바깥으로
        self.ax_img.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                           fontsize=9, framealpha=0.9)

        # (아래) 잔차(오차) 비교
        self.ax_res.cla()
        self.ax_res.axhline(0, color="gray", lw=1)
        for key in active_keys:
            f = self.fits[key]
            ls = MODEL_DASH[key] if multi else "-"
            self.ax_res.plot(f["resid_x"], f["resid"],
                             color=MODEL_COLORS[key], ls=ls,
                             marker=".", ms=5, lw=0.8,
                             label=f"{MODEL_LABELS[key]} residual")
        self.ax_res.set_xlabel("pixel x")
        self.ax_res.set_ylabel("residual\n$y - f(x)$")
        self.ax_res.set_title(
            "Residual (error) comparison  "
            "(smaller & more random = better)")
        if active_keys:
            self.ax_res.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                               fontsize=9)
        self.ax_res.grid(True, alpha=0.3)

        self.fig.canvas.draw_idle()

    # ── 콜백 ────────────────────────────────────────────
    def _on_check(self, label):
        # label -> key 역매핑
        for k, lab in MODEL_LABELS.items():
            if lab == label:
                self.active[k] = not self.active[k]
                break
        self.redraw()

    def _on_box_select(self, eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        mask = ((self.x >= x0) & (self.x <= x1) &
                (self.y >= y0) & (self.y <= y1))
        if mask.sum() < 3:
            print(f"선택된 점이 {int(mask.sum())}개라 너무 적습니다 "
                  "(최소 3개). 선택 무시.")
            return
        self.mask = mask
        print(f"박스 선택: {int(mask.sum())}개 점으로 다시 피팅합니다.")
        self.compute_fits()
        self.redraw()

    def _on_reset(self, event):
        self.mask = np.ones(len(self.x), dtype=bool)
        print("전체 점으로 복귀.")
        self.compute_fits()
        self.redraw()

    def run(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="인터랙티브 커브 피팅")
    parser.add_argument("--image", default="clifton_6388.jpg")
    parser.add_argument("--csv", default="digitized.csv")
    args = parser.parse_args()

    try:
        app = CurveFitApp(args.image, args.csv)
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e.filename}")
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
