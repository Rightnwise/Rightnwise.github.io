"""
복소 푸리에 급수 & 에피사이클 (Fourier Epicycles)

parametrize.py 로 만든 z(t) = x(t) + i·y(t) 를 복소 푸리에 급수로 바꾼다.

    z(t) ≈ Σ_k  c_k · exp(i · k · t)      (t ∈ [0, 2π))

각 항 c_k·exp(i·k·t) 는 반지름 |c_k|, 각속도 k 로 도는 '원(에피사이클)' 이다.
원들을 머리-꼬리로 이으면 맨 끝점이 곡선을 그린다.
항(=원) 을 늘릴수록 그림이 점점 정확해진다.

계수는 FFT 로 한 번에 구한다:  c = fft(z) / N .
진폭 |c_k| 가 큰 항부터 그려야 원이 자연스럽게 큰 것 -> 작은 것 순으로 겹친다.

사용법:
    python fourier_epicycles.py                       # spline_points.csv 사용
    python fourier_epicycles.py --terms 20            # 원 20개만 사용
    python fourier_epicycles.py --from-curve curve_txy.csv   # 저장된 z 사용
    python fourier_epicycles.py --grow                # 항을 1개씩 늘려가며 보여주기
"""

import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from parametrize import (load_curve, parametrize,
                         bezier_dense, resample_uniform)


def load_curve_txy(path):
    """parametrize.py 가 저장한 t, x, y CSV -> z 배열."""
    xs, ys = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
    return np.array(xs) + 1j * np.array(ys)


def fourier_coeffs(z):
    """z (등간격 N개 샘플) -> (freqs k, 계수 c_k), 진폭 큰 순으로 정렬."""
    n = z.size
    c = np.fft.fft(z) / n              # c_k = (1/N) Σ z_j exp(-i 2π k j / N)
    k = np.fft.fftfreq(n, d=1.0 / n)  # 정수 주파수 0, 1, ..., -2, -1
    k = np.rint(k).astype(int)

    order = np.argsort(-np.abs(c))    # 진폭 큰 항부터
    return k[order], c[order]


def reconstruct(k, c, t):
    """주어진 항들로 z(t) 재구성. t 는 스칼라 또는 배열."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    # (항, 시간) 행렬 합
    terms = c[:, None] * np.exp(1j * k[:, None] * t[None, :])
    return terms.sum(axis=0)


def epicycle_chain(k, c, t):
    """시간 t 에서 각 원 중심의 누적 위치(복소수) 시퀀스 반환.

    반환 pts[i] = 앞의 i개 항까지 더한 끝점. pts[0]=0(원점),
    pts[-1] = 완성된 z(t). 원 i 의 중심=pts[i], 끝=pts[i+1], 반지름=|c_i|.
    """
    vecs = c * np.exp(1j * k * t)     # 각 항의 벡터
    pts = np.concatenate([[0.0 + 0.0j], np.cumsum(vecs)])
    return pts


class EpicycleAnimator:
    def __init__(self, z, n_terms=None, frames=400, grow=False):
        self.k, self.c = fourier_coeffs(z)
        self.total_terms = self.k.size
        self.n_terms = n_terms or self.total_terms
        self.frames = frames
        self.grow = grow

        # 전체 곡선(모든 항) — 배경 참고용
        tt = np.linspace(0, 2 * np.pi, 600)
        full = reconstruct(self.k, self.c, tt)

        self.fig, self.ax = plt.subplots(figsize=(8, 8.8), facecolor="white")
        self.ax.set_facecolor("white")
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()            # 이미지 픽셀 좌표계
        self.ax.grid(True, alpha=0.15, lw=0.5)

        # 어떤 급수인지 위에 표시 (복소 푸리에 급수 = 에피사이클)
        self.fig.suptitle(
            r"complex Fourier series:  "
            r"$z(t)=x(t)+i\,y(t)=\sum_{k} c_k\, e^{\,i k t}$",
            fontsize=14, y=0.975)
        # suptitle / 축 제목이 잘리지 않도록 위쪽 여백 확보
        self.fig.subplots_adjust(top=0.9, bottom=0.05, left=0.06, right=0.97)

        pad = 0.08 * (full.real.max() - full.real.min() + 1)
        self.ax.set_xlim(full.real.min() - pad, full.real.max() + pad)
        self.ax.set_ylim(full.imag.max() + pad, full.imag.min() - pad)

        # 선들: antialiased 로 부드럽게
        (self.circles,) = self.ax.plot([], [], "-", color="0.72", lw=0.7,
                                       antialiased=True)
        (self.radii,) = self.ax.plot([], [], "-", color="tab:blue", lw=1.0,
                                     antialiased=True, solid_capstyle="round")
        (self.path,) = self.ax.plot([], [], "-", color="crimson", lw=2.4,
                                    antialiased=True, solid_capstyle="round",
                                    solid_joinstyle="round")

        # 원 하나를 그릴 각도 해상도(촘촘할수록 원이 매끈)
        self._theta = np.linspace(0, 2 * np.pi, 120)

        # 궤적을 프레임 수와 무관하게 매끈히 그리기 위해, 완성 곡선을
        # 고해상도로 미리 계산해두고 프레임 진행에 따라 잘라서 보여준다.
        self._trace_res = 3000
        self._tt_full = np.linspace(0, 2 * np.pi, self._trace_res)
        if not self.grow:
            self._path_full = reconstruct(
                self.k[:self.n_terms], self.c[:self.n_terms], self._tt_full)

    def _current_nterms(self, frame):
        if not self.grow:
            return self.n_terms
        # grow 모드: 프레임 진행에 따라 항 개수를 1 -> n_terms 로 증가
        frac = (frame + 1) / self.frames
        return max(1, int(round(frac * self.n_terms)))

    def update(self, frame):
        t = 2 * np.pi * frame / self.frames
        m = self._current_nterms(frame)
        k = self.k[:m]
        c = self.c[:m]

        pts = epicycle_chain(k, c, t)

        # 원들
        cxs, cys = [], []
        for i in range(m):
            cx, cy = pts[i].real, pts[i].imag
            r = abs(c[i])
            cxs.extend(cx + r * np.cos(self._theta))
            cys.extend(cy + r * np.sin(self._theta))
            cxs.append(np.nan)            # 원 사이 선 끊기
            cys.append(np.nan)
        self.circles.set_data(cxs, cys)

        # 중심들을 잇는 반지름 선
        self.radii.set_data(pts.real, pts.imag)

        # 끝점이 그리는 곡선
        if self.grow:
            # grow 모드: 항 개수가 매 프레임 바뀌므로 그 항들로 완성 곡선을 표시
            rec = reconstruct(k, c, self._tt_full)
            self.path.set_data(rec.real, rec.imag)
        else:
            # 미리 계산한 고해상도 곡선을 진행도만큼 잘라 매끈하게 공개
            idx = int(round((frame + 1) / self.frames * self._trace_res))
            seg = self._path_full[:max(idx, 2)]
            self.path.set_data(seg.real, seg.imag)

        self.ax.set_title(
            f"terms (circles): {m} / {self.total_terms}"
            + ("   [grow]" if self.grow else ""), fontsize=11)
        return self.circles, self.radii, self.path

    def _make_anim(self, interval=20):
        return animation.FuncAnimation(
            self.fig, self.update, frames=self.frames,
            interval=interval, blit=False, repeat=True)

    def run(self):
        self.anim = self._make_anim()
        plt.show()

    def save_gif(self, path, fps=30, dpi=130):
        """애니메이션을 GIF 로 저장 (Pillow 사용, 외부 프로그램 불필요).

        fps 를 높이고 프레임을 늘릴수록 부드럽고, dpi 를 높일수록 선명하다.
        """
        # tight_layout 은 suptitle 을 잘라낼 수 있어 쓰지 않고,
        # __init__ 의 subplots_adjust 여백을 그대로 사용한다.
        self.anim = self._make_anim(interval=1000 / fps)
        writer = animation.PillowWriter(fps=fps)
        self.anim.save(path, writer=writer, dpi=dpi,
                       savefig_kwargs={"facecolor": "white"})
        print(f"GIF 저장 완료: {path}  "
              f"({self.frames} frames @ {fps}fps, dpi={dpi})")


def main():
    parser = argparse.ArgumentParser(description="복소 푸리에 에피사이클")
    parser.add_argument("csv", nargs="?", default="spline_points.csv",
                        help="클릭점 CSV (parametrize 로 z 생성)")
    parser.add_argument("--from-curve", default=None,
                        help="parametrize 가 저장한 t,x,y CSV 에서 z 로드")
    parser.add_argument("--N", type=int, default=512,
                        help="등간격 재샘플 개수 (2^k 권장)")
    parser.add_argument("--terms", type=int, default=None,
                        help="사용할 원(항) 개수 (기본: 전부)")
    parser.add_argument("--open", action="store_true",
                        help="열린 곡선으로 처리 (기본은 닫힌 곡선)")
    parser.add_argument("--frames", type=int, default=300,
                        help="애니메이션 프레임 수 (많을수록 부드러움)")
    parser.add_argument("--grow", action="store_true",
                        help="항을 1개씩 늘려가며 정확해지는 과정 보기")
    parser.add_argument("--gif", metavar="OUT.gif", default=None,
                        help="화면 대신 GIF 파일로 저장")
    parser.add_argument("--fps", type=int, default=30,
                        help="GIF 프레임레이트 (높을수록 부드러움)")
    parser.add_argument("--dpi", type=int, default=130,
                        help="GIF 해상도 (높을수록 선명, 파일 커짐)")
    args = parser.parse_args()

    if args.from_curve:
        z = load_curve_txy(args.from_curve)
        print(f"z 로드: {args.from_curve} (N={z.size})")
    else:
        # spline_points.csv 는 옛 점 형식 / bezier_editor 형식 모두 지원
        kind, data = load_curve(args.csv)
        if kind == "bezier":
            anchors, controls, closed = data
            if args.open:
                closed = False
            xs, ys = bezier_dense(anchors, controls, closed)
            _, _, _, z = resample_uniform(xs, ys, args.N, closed)
            print(f"베지어 {len(anchors)}앵커 -> z (N={z.size})")
        else:
            px, py = data
            _, _, _, z = parametrize(px, py, N=args.N, closed=not args.open)
            print(f"점 {px.size}개 -> z (N={z.size})")

    anim = EpicycleAnimator(z, n_terms=args.terms,
                            frames=args.frames, grow=args.grow)
    if args.gif:
        anim.save_gif(args.gif, fps=args.fps, dpi=args.dpi)
    else:
        anim.run()


if __name__ == "__main__":
    main()
