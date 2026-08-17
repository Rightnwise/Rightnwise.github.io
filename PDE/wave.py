"""
파동방정식 정상파(normal modes) 애니메이션

1차원 파동방정식  u_tt = c^2 u_xx ,  0 <= x <= L , 양 끝 고정
    u(0,t) = u(L,t) = 0
을 변수분리하면 기본해(노멀 모드)는

    u_n(x,t) = sin(nπx/L) · [ A_n cos(cnπt/L) + B_n sin(cnπt/L) ]

즉 시간 부분이 cos / sin 두 종류다.
    윗줄 :  u = cos(cnπt/L) · sin(nπx/L)      (t=0 에서 최대 변위)
    아랫줄:  u = sin(cnπt/L) · sin(nπx/L)      (t=0 에서 0, 이후 진동)

세 가지 보기 모드(--mode):
    single  : n 하나만 (----n 으로 지정)
    compare : n = 1, 2, 3 ... 을 나란히 비교 (--modes)
    sum     : 여러 모드의 '합'(중첩) 을 한 그림에. 개별 모드는 옅게 함께 표시.
              u(x,t) = Σ_n a_n sin(nπx/L) cos/sin(cnπt/L)
              (초기조건이 없으므로 계수 a_n 은 기본 1, --coeffs 로 지정 가능)

사용법:
    python wave.py                          # 기본: compare, n=1,2,3
    python wave.py --mode single --n 2
    python wave.py --mode compare --modes 1 2 3 4
    python wave.py --mode sum --modes 1 2 3 --coeffs 1 0.5 0.3
    python wave.py --mode sum --save wave_sum.gif
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def family(kind, ns, coeffs, x, L, c, t):
    """kind='cos'|'sin' 시간부. ns/coeffs 합쳐서 u(x) 반환."""
    u = np.zeros_like(x)
    for n, a in zip(ns, coeffs):
        omega = c * n * np.pi / L
        time = np.cos(omega * t) if kind == "cos" else np.sin(omega * t)
        u += a * time * np.sin(n * np.pi * x / L)
    return u


def pluck_coeffs(ns, d, h, L):
    """삼각형(한 점 d 에서 높이 h 로 뜯음) 초기조건의 사인 계수 a_n.

        a_n = 2 h L^2 / (π^2 n^2 d (L-d)) · sin(nπd/L),   b_n = 0
    """
    ns = np.asarray(ns, dtype=float)
    return (2 * h * L ** 2 / (np.pi ** 2 * ns ** 2 * d * (L - d))
            * np.sin(ns * np.pi * d / L))


def run_pluck(args):
    """삼각형 초기조건에서 출발한 줄의 실제 진동 애니메이션."""
    L, c, h, d = args.L, args.c, args.h, args.d
    terms = args.terms
    ns = np.arange(1, terms + 1)
    a = pluck_coeffs(ns, d, h, L)                 # 계수 a_n (b_n=0)

    omega = c * ns * np.pi / L                    # 각 모드 각진동수 ω_n ∝ n
    # 주파수 '제곱'에 비례하는 감쇠: γ_n = damping · ω_n²
    # (고차 모드가 n² 로 훨씬 빨리 죽음 — 실제 줄의 점성/굽힘 감쇠에 가까움)
    gamma = args.damping * omega ** 2

    x = np.linspace(0, L, 500)
    S = np.sin(np.outer(ns, np.pi * x / L))       # (terms, x): sin(nπx/L)

    # 초기 삼각형(참고용)
    tri = np.where(x <= d, h * x / d, h * (L - x) / (L - d))

    T1 = 2 * L / c                                # 기본 주기
    times = np.linspace(0, args.periods * T1, args.frames)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    fig.suptitle("Plucked string:  "
                 "u(x,t) = Σ a_n sin(nπx/L) cos(cnπt/L) · exp(−γ_n t)",
                 fontsize=13, y=0.97)
    ax.set_title(f"pluck d={d:g}, h={h:g}, terms={terms}, "
                 f"damping={args.damping:g}  (γ_n ∝ frequency²)",
                 fontsize=11)
    ax.set_xlim(0, L)
    ax.set_ylim(-1.3 * h, 1.3 * h)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.plot(x, tri, "--", color="0.6", lw=1, label="initial triangle f(x)")
    ax.axvline(d, color="0.85", lw=0.8)
    (line,) = ax.plot([], [], color="crimson", lw=2.4, label="u(x,t)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.15)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11)

    def update(frame):
        t = times[frame]
        # a_n cos(ω_n t) · exp(−γ_n t)   (γ_n ∝ 주파수² → 고차 모드 n²로 빠르게 감쇠)
        temporal = a * np.cos(omega * t) * np.exp(-gamma * t)
        u = temporal @ S                                 # Σ_n ... sin(nπx/L)
        line.set_data(x, u)
        time_text.set_text(f"t = {t:.3f}   (T₁ = {T1:.3f})")
        return line, time_text

    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.08, right=0.97)
    anim = animation.FuncAnimation(fig, update, frames=args.frames,
                                   interval=1000 / args.fps, blit=False)
    if args.save:
        anim.save(args.save, writer=animation.PillowWriter(fps=args.fps),
                  dpi=110, savefig_kwargs={"facecolor": "white"})
        print(f"GIF 저장 완료: {args.save} "
              f"({args.frames} frames @ {args.fps}fps)")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="파동방정식 정상파 모드 애니메이션")
    parser.add_argument("--mode",
                        choices=["single", "compare", "sum", "pluck"],
                        default="compare", help="보기 방식")
    parser.add_argument("--n", type=int, default=1,
                        help="single 모드에서 그릴 n")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3],
                        help="compare/sum 에서 쓸 n 목록")
    parser.add_argument("--coeffs", type=float, nargs="+", default=None,
                        help="sum 에서 각 모드 계수 a_n (기본 모두 1)")
    parser.add_argument("--d", type=float, default=0.5,
                        help="pluck 모드: 뜯는 위치 (기본 L/2=0.5)")
    parser.add_argument("--h", type=float, default=0.2,
                        help="pluck 모드: 뜯는 높이")
    parser.add_argument("--terms", type=int, default=40,
                        help="pluck 모드: 더할 모드 개수")
    parser.add_argument("--damping", type=float, default=0.0,
                        help="pluck 모드: 주파수제곱비례 감쇠 세기 "
                             "(γ_n = damping·ω_n², 0=감쇠없음)")
    parser.add_argument("--L", type=float, default=1.0, help="길이 L")
    parser.add_argument("--c", type=float, default=1.0, help="파동 속도 c")
    parser.add_argument("--frames", type=int, default=200,
                        help="애니메이션 프레임 수")
    parser.add_argument("--periods", type=float, default=1.0,
                        help="기본 모드 기준 몇 주기를 보여줄지")
    parser.add_argument("--save", metavar="OUT.gif", default=None,
                        help="화면 대신 GIF 로 저장")
    parser.add_argument("--fps", type=int, default=30, help="GIF 프레임레이트")
    args = parser.parse_args()

    if args.mode == "pluck":
        run_pluck(args)
        return

    L, c = args.L, args.c
    x = np.linspace(0, L, 400)

    # ── 열(column) 구성: 각 열은 (제목, ns, coeffs, 개별표시여부) ──
    if args.mode == "single":
        columns = [(f"n = {args.n}", [args.n], [1.0], False)]
    elif args.mode == "compare":
        columns = [(f"n = {n}", [n], [1.0], False) for n in args.modes]
    else:  # sum
        ns = args.modes
        coeffs = args.coeffs if args.coeffs else [1.0] * len(ns)
        if len(coeffs) != len(ns):
            parser.error("--coeffs 개수가 --modes 개수와 같아야 합니다.")
        label = "  +  ".join(f"{a:g}·sin({n}πx/L)" for n, a in zip(ns, coeffs))
        columns = [(f"sum:  {label}", ns, coeffs, True)]

    ncols = len(columns)

    # 시간 범위: 가장 낮은 n 의 주기 T = 2L/(c·n) 기준
    n_base = min(min(cols[1]) for cols in columns)
    T0 = 2 * L / (c * n_base)
    times = np.linspace(0, args.periods * T0, args.frames)

    # y 범위: 합이면 계수 절댓값 합, 아니면 1.2
    ymax = max(1.0, max(sum(abs(a) for a in cols[2]) for cols in columns)) * 1.15

    fig_w = max(8.0, 4.2 * ncols + 1)      # 공식이 안 잘리도록 최소 너비 확보
    fig, axes = plt.subplots(2, ncols, figsize=(fig_w, 6.6),
                             sharex=True, sharey=True, facecolor="white",
                             squeeze=False)

    fig.suptitle(
        r"Wave equation normal modes:  "
        r"$u_n(x,t)=\sin\frac{n\pi x}{L}"
        r"\left[\cos\frac{cn\pi t}{L}\ \mathrm{or}\ \sin\frac{cn\pi t}{L}\right]$",
        fontsize=13, y=0.985)

    row_label = [r"$\cos\frac{cn\pi t}{L}\,\sin\frac{n\pi x}{L}$",
                 r"$\sin\frac{cn\pi t}{L}\,\sin\frac{n\pi x}{L}$"]
    kinds = ["cos", "sin"]

    main_lines = {}      # (row, j) -> 합/모드 곡선
    comp_lines = {}      # (row, j) -> [개별 모드 곡선들] (sum 일 때만)

    for j, (title, ns, coeffs, show_comp) in enumerate(columns):
        for row in range(2):
            ax = axes[row][j]
            ax.set_xlim(0, L)
            ax.set_ylim(-ymax, ymax)
            ax.axhline(0, color="0.8", lw=0.8)
            ax.grid(True, alpha=0.15)

            if not show_comp:
                # 단일 모드: 진폭 포락선 ±sin(nπx/L)
                env = coeffs[0] * np.sin(ns[0] * np.pi * x / L)
                ax.plot(x, env, "--", color="0.75", lw=1)
                ax.plot(x, -env, "--", color="0.75", lw=1)
            else:
                # 합: 개별 모드를 옅게 함께 표시
                comp_lines[(row, j)] = [
                    ax.plot([], [], color="steelblue", lw=1,
                            alpha=0.5)[0] for _ in ns]

            (ln,) = ax.plot([], [], color="crimson", lw=2.4)
            main_lines[(row, j)] = ln

            if row == 0:
                ax.set_title(title, fontsize=12)
            if j == 0:
                ax.set_ylabel(row_label[row], fontsize=12)

    time_text = fig.text(0.5, 0.925, "", ha="center", fontsize=11)

    def update(frame):
        t = times[frame]
        arts = []
        for j, (title, ns, coeffs, show_comp) in enumerate(columns):
            for row in range(2):
                u = family(kinds[row], ns, coeffs, x, L, c, t)
                main_lines[(row, j)].set_data(x, u)
                arts.append(main_lines[(row, j)])
                if show_comp:
                    for cl, n, a in zip(comp_lines[(row, j)], ns, coeffs):
                        ui = family(kinds[row], [n], [a], x, L, c, t)
                        cl.set_data(x, ui)
                        arts.append(cl)
        time_text.set_text(f"t = {t:.3f}   (T = {T0:.3f} for n={n_base})")
        return arts

    fig.subplots_adjust(top=0.87, bottom=0.07, hspace=0.15,
                        wspace=0.1, left=0.08, right=0.97)

    anim = animation.FuncAnimation(fig, update, frames=args.frames,
                                   interval=1000 / args.fps, blit=False)

    if args.save:
        writer = animation.PillowWriter(fps=args.fps)
        anim.save(args.save, writer=writer, dpi=110,
                  savefig_kwargs={"facecolor": "white"})
        print(f"GIF 저장 완료: {args.save} "
              f"({args.frames} frames @ {args.fps}fps)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
