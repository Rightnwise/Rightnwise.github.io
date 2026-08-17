"""
매개변수 곡선 만들기 (Parametrize):  점 -> x(t), y(t)

목표: 나중에 z(t) = x(t) + i·y(t) 를 복소 푸리에 급수(에피사이클)로 바꾸기
위한 '재료'를 만든다. 푸리에는 아직 신경 쓰지 않고, 여기서는 x(t), y(t) 를
푸리에가 먹기 좋은 형태로 뽑는 데만 집중한다.

왜 이렇게 하나
--------------
1) 매듭 매개변수는 '코드길이(chord length)' 누적으로 잡는다.
   클릭 간격이 불규칙해도 곡선 모양이 안정적으로 나온다.
   닫힌 곡선이므로 마지막에 첫 점으로 되돌려 이어 붙인다.

2) 그 이차 스플라인을 t 에 대해 '등간격'으로 재샘플링한다.
   DFT/FFT 는 매개변수가 균일 간격인 주기 데이터를 전제로 하므로,
   클릭점을 그대로 쓰지 않고 등간격 x_k, y_k 를 새로 뽑는다.
   t 는 [0, 2π) 로 정규화한다 (푸리에 계수 해석이 깔끔).

결과
----
parametrize() 가 (t, x, y, z) 를 돌려준다.
    t : shape (N,), [0, 2π) 등간격
    x : shape (N,), x(t)
    y : shape (N,), y(t)
    z : shape (N,), 복소수 x + i·y   (다음 단계 푸리에용)

사용법:
    python parametrize.py spline_points.csv --N 512 --closed
"""

import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

from quadratic_spline import QuadraticSpline
from bezier_editor import quad_bezier


def load_points(path):
    """옛 digitizer 형식 (index, pixel_x, pixel_y) 에서 점 배열 읽기."""
    xs, ys = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["pixel_x"]))
            ys.append(float(row["pixel_y"]))
    return np.array(xs), np.array(ys)


def load_curve(path):
    """CSV 형식을 자동 판별해 곡선 정보를 반환한다.

    반환 ("points", (px, py))
        옛 digitizer 형식 (index, pixel_x, pixel_y): 클릭점만 있음.
    반환 ("bezier", (anchors, controls, closed))
        bezier_editor 형식 (kind, index, x, y): 앵커 + 방향점.
        방향점 개수로 닫힘 여부를 추론한다
        (controls == anchors -> 닫힘, controls == anchors-1 -> 열림).
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    if "kind" in fields:                      # bezier_editor 형식
        anchors, controls = [], []
        for row in rows:
            p = (float(row["x"]), float(row["y"]))
            if row["kind"] == "anchor":
                anchors.append(p)
            elif row["kind"] == "control":
                controls.append(p)
        closed = len(controls) == len(anchors) and len(anchors) >= 3
        return "bezier", (anchors, controls, closed)

    # 옛 형식
    px = np.array([float(r["pixel_x"]) for r in rows])
    py = np.array([float(r["pixel_y"]) for r in rows])
    return "points", (px, py)


def bezier_dense(anchors, controls, closed, per_seg=80):
    """앵커 + 방향점 -> 이차 베지어 곡선을 촘촘히 샘플링한 (xs, ys)."""
    n = len(anchors)
    if n < 2:
        raise ValueError("앵커가 최소 2개 필요합니다.")
    segs = n if closed else n - 1
    u = np.linspace(0, 1, per_seg)
    xs, ys = [], []
    for seg in range(segs):
        a0 = np.array(anchors[seg])
        a1 = np.array(anchors[(seg + 1) % n])
        c = np.array(controls[seg])
        b = quad_bezier(a0, c, a1, u)
        if seg > 0:                           # 이음매 중복 앵커 제거
            b = b[1:]
        xs.append(b[:, 0])
        ys.append(b[:, 1])
    return np.concatenate(xs), np.concatenate(ys)


def resample_uniform(xs, ys, N, closed):
    """조밀한 곡선점 (xs, ys) 를 코드길이 기준 등간격 N개로 재샘플 -> (t,x,y,z)."""
    d = np.hypot(np.diff(xs), np.diff(ys))
    d[d == 0] = 1e-9
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = s[-1]
    if total == 0:
        raise ValueError("곡선 길이가 0 입니다.")
    sq = np.linspace(0.0, total, N, endpoint=not closed)
    x = np.interp(sq, s, xs)
    y = np.interp(sq, s, ys)
    t = sq / total * (2.0 * np.pi)
    return t, x, y, x + 1j * y


def parametrize(px, py, N=512, closed=True):
    """클릭점 (px, py) -> 등간격 매개변수 곡선 (t, x, y, z).

    px, py : 클릭한 점들의 좌표 (1차원 배열)
    N      : 재샘플링할 점 개수 (푸리에용, 2의 거듭제곱 권장)
    closed : 닫힌 곡선이면 True (시작점으로 되돌려 이어 붙임)
    """
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if px.size < 2:
        raise ValueError("점이 최소 2개 필요합니다.")

    if closed:
        px = np.concatenate([px, px[:1]])
        py = np.concatenate([py, py[:1]])

    # 1) 코드길이 매개변수: s_{i+1} = s_i + |P_{i+1} - P_i|
    d = np.hypot(np.diff(px), np.diff(py))
    d[d == 0] = 1e-9                       # 중복점 방지
    s = np.concatenate([[0.0], np.cumsum(d)])   # 매듭에서의 누적 길이
    total = s[-1]

    # 2) 이차 스플라인 x(s), y(s)
    sx = QuadraticSpline(s, px)
    sy = QuadraticSpline(s, py)

    # 3) 등간격 재샘플. 닫힌 곡선이면 마지막점(=첫점)을 빼서 중복 제거
    #    -> 주기 신호로서 t_k = 2π k / N, k=0..N-1
    if closed:
        s_query = np.linspace(0.0, total, N, endpoint=False)
    else:
        s_query = np.linspace(0.0, total, N)

    x = sx(s_query)
    y = sy(s_query)

    # 매개변수를 [0, 2π) 로 정규화
    t = s_query / total * (2.0 * np.pi)

    z = x + 1j * y
    return t, x, y, z


def save_curve(path, t, x, y):
    """t, x(t), y(t) 를 CSV 로 저장 (다음 단계에서 다시 읽기 좋게)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y"])
        for tv, xv, yv in zip(t, x, y):
            writer.writerow([f"{tv:.6f}", f"{xv:.4f}", f"{yv:.4f}"])
    print(f"저장 완료: {path} ({len(t)} samples)")


def plot(t, x, y, px, py):
    """x(t), y(t) 함수 그래프와 복원한 (x, y) 곡선을 함께 그린다."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(t, x, "b-")
    ax1.set_title("x(t)")
    ax1.set_xlabel("t  [0, 2π)")
    ax1.set_ylabel("x")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, y, "r-")
    ax2.set_title("y(t)")
    ax2.set_xlabel("t  [0, 2π)")
    ax2.set_ylabel("y")
    ax2.grid(True, alpha=0.3)

    # 닫힌 곡선 시각화를 위해 첫 점을 끝에 이어 붙여 그림
    ax3.plot(np.append(x, x[0]), np.append(y, y[0]), "c-", lw=1.5,
             label="resampled x(t),y(t)")
    ax3.plot(px, py, "ro", ms=4, label="clicked points")
    ax3.set_title("(x(t), y(t)) curve")
    ax3.set_aspect("equal", adjustable="datalim")
    ax3.invert_yaxis()                     # 이미지 픽셀 좌표계
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="점 -> x(t), y(t) 매개변수화")
    parser.add_argument("csv", nargs="?", default="spline_points.csv",
                        help="digitizer 로 저장한 점 CSV")
    parser.add_argument("--N", type=int, default=512,
                        help="등간격 재샘플 개수 (푸리에용, 2^k 권장)")
    parser.add_argument("--open", action="store_true",
                        help="열린 곡선으로 처리 (기본은 닫힌 곡선)")
    parser.add_argument("--out", default="curve_txy.csv",
                        help="t, x, y 저장 파일명")
    args = parser.parse_args()

    kind, data = load_curve(args.csv)

    if kind == "bezier":
        anchors, controls, closed = data
        if args.open:
            closed = False                    # 강제로 열린 곡선 처리
        xs, ys = bezier_dense(anchors, controls, closed)
        t, x, y, z = resample_uniform(xs, ys, args.N, closed)
        px = np.array([a[0] for a in anchors])
        py = np.array([a[1] for a in anchors])
        print(f"베지어: 앵커 {len(anchors)}개, 방향점 {len(controls)}개, "
              f"{'closed' if closed else 'open'} -> 등간격 {args.N}개 샘플")
    else:
        px, py = data
        t, x, y, z = parametrize(px, py, N=args.N, closed=not args.open)
        print(f"점 {px.size}개 -> 등간격 {args.N}개 샘플")

    print(f"z(t) = x + i·y  준비 완료  (shape={z.shape}, dtype={z.dtype})")

    save_curve(args.out, t, x, y)
    plot(t, x, y, px, py)


if __name__ == "__main__":
    main()
