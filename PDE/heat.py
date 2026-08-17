"""
열방정식(heat equation) 확산 애니메이션

1차원 열방정식  u_t = α u_xx ,  0 <= x <= L , 양 끝 온도 0 고정
    u(0,t) = u(L,t) = 0 ,   u(x,0) = f(x)
을 변수분리하면 기본해(모드)는

    u_n(x,t) = sin(nπx/L) · exp( −α (nπ/L)^2 t )

파동방정식과 달리 시간부가 진동(cos/sin)이 아니라 지수감쇠 하나뿐이다.
고차 모드일수록 (nπ/L)^2 에 비례해 훨씬 빨리 사그라든다.

초기조건 f(x) 는 삼각형(한 점 d 에서 높이 h):

    f(x) =  h·x/d           (0 ≤ x ≤ d)
            h·(L−x)/(L−d)   (d ≤ x ≤ L)

이 f(x) 의 사인급수 계수는

    b_n = (2/L)∫₀ᴸ f(x) sin(nπx/L) dx
        = 2 h L² / (π² n² d (L−d)) · sin(nπd/L)

따라서 열방정식의 해는

    u(x,t) = Σ_n b_n sin(nπx/L) exp( −α (nπ/L)² t )

세 가지 보기 모드(--mode):
    triangle : 삼각형 초기조건 f(x) 가 확산으로 퍼져 사라지는 실제 해 (기본)
    single   : 모드 n 하나만 (--n 으로 지정)
    compare  : n = 1, 2, 3 ... 을 나란히 비교 (--modes)
    sum      : 여러 모드의 '합'(중첩)을 한 그림에. 개별 모드는 옅게 함께 표시.

사용법:
    python heat.py                              # 기본: 삼각형 초기조건 확산
    python heat.py --mode single --n 2
    python heat.py --mode compare --modes 1 2 3 4
    python heat.py --mode sum --modes 1 2 3 --coeffs 1 0.5 0.3
    python heat.py --mode triangle --save heat_triangle.gif
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def family(ns, coeffs, x, L, alpha, t):
    """모드 ns/coeffs 를 합쳐 시각 t 의 u(x) 반환.

        u(x) = Σ a_n sin(nπx/L) exp(−α (nπ/L)² t)
    """
    u = np.zeros_like(x)
    for n, a in zip(ns, coeffs):
        decay = np.exp(-alpha * (n * np.pi / L) ** 2 * t)
        u += a * decay * np.sin(n * np.pi * x / L)
    return u


def triangle_coeffs(ns, d, h, L):
    """삼각형(한 점 d 에서 높이 h) 초기조건 f(x) 의 사인급수 계수 b_n.

        b_n = 2 h L² / (π² n² d (L−d)) · sin(nπd/L)
    """
    ns = np.asarray(ns, dtype=float)
    return (2 * h * L ** 2 / (np.pi ** 2 * ns ** 2 * d * (L - d))
            * np.sin(ns * np.pi * d / L))


def run_triangle(args):
    """삼각형 초기조건 f(x) 가 열확산으로 퍼져 0 으로 사라지는 애니메이션."""
    L, alpha, h, d = args.L, args.alpha, args.h, args.d
    terms = args.terms
    ns = np.arange(1, terms + 1)
    b = triangle_coeffs(ns, d, h, L)              # 계수 b_n

    x = np.linspace(0, L, 500)
    S = np.sin(np.outer(ns, np.pi * x / L))       # (terms, x): sin(nπx/L)
    rate = alpha * (ns * np.pi / L) ** 2          # 모드별 감쇠율 λ_n ∝ n²

    # 초기 삼각형(참고용)
    tri = np.where(x <= d, h * x / d, h * (L - x) / (L - d))

    tau1 = 1.0 / (alpha * (np.pi / L) ** 2)       # 기본 모드 감쇠시간 τ₁
    times = np.linspace(0, args.taus * tau1, args.frames)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    fig.suptitle("Heat equation:  "
                 "u(x,t) = Σ b_n sin(nπx/L) · exp(−α(nπ/L)² t)",
                 fontsize=13, y=0.97)
    ax.set_title(f"triangle f(x):  d={d:g}, h={h:g}, α={alpha:g}, "
                 f"terms={terms}   (λ_n ∝ n²)",
                 fontsize=11)
    ax.set_xlim(0, L)
    ax.set_ylim(-0.1 * h, 1.25 * h)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.plot(x, tri, "--", color="0.6", lw=1, label="initial triangle f(x)")
    ax.axvline(d, color="0.85", lw=0.8)

    # 잔상(trail): 지나간 시각의 온도 곡선을 옅게 남긴다 (오래될수록 흐려짐)
    n_trail = max(0, args.trails)
    cmap = plt.get_cmap("inferno")
    trail_lines = [ax.plot([], [], lw=1.4)[0] for _ in range(n_trail)]

    (line,) = ax.plot([], [], color="orangered", lw=2.4, label="u(x,t)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.15)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11)

    history = []      # 지금까지 계산한 (t, u) 기록

    def update(frame):
        t = times[frame]
        temporal = b * np.exp(-rate * t)                 # b_n exp(−λ_n t)
        u = temporal @ S                                 # Σ_n ... sin(nπx/L)
        line.set_data(x, u)
        history.append(u)

        # 최근 n_trail 개의 과거 곡선을, 오래될수록 옅게 표시
        past = history[-n_trail - 1:-1]                  # 현재 프레임 제외
        for tl in trail_lines:
            tl.set_data([], [])
        for k, past_u in enumerate(past):
            age = (len(past) - k) / (n_trail + 1)        # 0(최근)~1(과거)
            tl = trail_lines[k]
            tl.set_data(x, past_u)
            tl.set_color(cmap(0.15 + 0.7 * age))
            tl.set_alpha(0.55 * (1 - age) + 0.12)
        time_text.set_text(f"t = {t:.3f}   (τ₁ = {tau1:.3f})")
        return [line, time_text, *trail_lines]

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
    parser = argparse.ArgumentParser(description="열방정식 확산 애니메이션")
    parser.add_argument("--mode",
                        choices=["triangle", "single", "compare", "sum"],
                        default="triangle", help="보기 방식")
    parser.add_argument("--n", type=int, default=1,
                        help="single 모드에서 그릴 n")
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3],
                        help="compare/sum 에서 쓸 n 목록")
    parser.add_argument("--coeffs", type=float, nargs="+", default=None,
                        help="sum 에서 각 모드 계수 a_n (기본 모두 1)")
    parser.add_argument("--d", type=float, default=0.5,
                        help="triangle 모드: 꼭짓점 위치 (기본 L/2=0.5)")
    parser.add_argument("--h", type=float, default=1.0,
                        help="triangle 모드: 꼭짓점 높이")
    parser.add_argument("--terms", type=int, default=40,
                        help="triangle 모드: 더할 모드 개수")
    parser.add_argument("--trails", type=int, default=12,
                        help="triangle 모드: 남길 잔상(과거 곡선) 개수 "
                             "(0=끔)")
    parser.add_argument("--L", type=float, default=1.0, help="길이 L")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="열확산계수 α")
    parser.add_argument("--frames", type=int, default=200,
                        help="애니메이션 프레임 수")
    parser.add_argument("--taus", type=float, default=1.5,
                        help="기본 감쇠시간 τ₁ 기준 몇 배까지 볼지")
    parser.add_argument("--periods", type=float, default=None,
                        help="--taus 의 다른 이름 (호환용)")
    parser.add_argument("--save", metavar="OUT.gif", default=None,
                        help="화면 대신 GIF 로 저장")
    parser.add_argument("--fps", type=int, default=30, help="GIF 프레임레이트")
    args = parser.parse_args()

    if args.periods is not None:      # --periods 를 써도 --taus 처럼 동작
        args.taus = args.periods

    if args.mode == "triangle":
        run_triangle(args)
        return

    L, alpha = args.L, args.alpha
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

    # 시간 범위: 가장 낮은 n 의 감쇠시간 τ = 1/(α(nπ/L)²) 기준
    n_base = min(min(cols[1]) for cols in columns)
    tau0 = 1.0 / (alpha * (n_base * np.pi / L) ** 2)
    times = np.linspace(0, args.taus * tau0, args.frames)

    # y 범위: 합이면 계수 절댓값 합, 아니면 1.2
    ymax = max(1.0, max(sum(abs(a) for a in cols[2]) for cols in columns)) * 1.15

    fig_w = max(8.0, 4.2 * ncols + 1)      # 공식이 안 잘리도록 최소 너비 확보
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.4),
                             sharex=True, sharey=True, facecolor="white",
                             squeeze=False)
    axes = axes[0]      # (1, ncols) → (ncols,)

    fig.suptitle(
        r"Heat equation modes:  "
        r"$u_n(x,t)=\sin\frac{n\pi x}{L}\,"
        r"\exp\!\left[-\alpha\left(\frac{n\pi}{L}\right)^{2}t\right]$",
        fontsize=13, y=0.98)

    main_lines = {}      # j -> 합/모드 곡선
    comp_lines = {}      # j -> [개별 모드 곡선들] (sum 일 때만)

    for j, (title, ns, coeffs, show_comp) in enumerate(columns):
        ax = axes[j]
        ax.set_xlim(0, L)
        ax.set_ylim(-ymax, ymax)
        ax.axhline(0, color="0.8", lw=0.8)
        ax.grid(True, alpha=0.15)

        if not show_comp:
            # 단일 모드: 초기 포락선 ±sin(nπx/L) (t=0 모양)
            env = coeffs[0] * np.sin(ns[0] * np.pi * x / L)
            ax.plot(x, env, "--", color="0.75", lw=1)
            ax.plot(x, -env, "--", color="0.75", lw=1)
        else:
            # 합: 개별 모드를 옅게 함께 표시
            comp_lines[j] = [
                ax.plot([], [], color="steelblue", lw=1,
                        alpha=0.5)[0] for _ in ns]

        (ln,) = ax.plot([], [], color="orangered", lw=2.4)
        main_lines[j] = ln
        ax.set_title(title, fontsize=12)
        if j == 0:
            ax.set_ylabel(r"$u(x,t)$", fontsize=12)
        ax.set_xlabel("x", fontsize=11)

    time_text = fig.text(0.5, 0.9, "", ha="center", fontsize=11)

    def update(frame):
        t = times[frame]
        arts = []
        for j, (title, ns, coeffs, show_comp) in enumerate(columns):
            u = family(ns, coeffs, x, L, alpha, t)
            main_lines[j].set_data(x, u)
            arts.append(main_lines[j])
            if show_comp:
                for cl, n, a in zip(comp_lines[j], ns, coeffs):
                    ui = family([n], [a], x, L, alpha, t)
                    cl.set_data(x, ui)
                    arts.append(cl)
        time_text.set_text(f"t = {t:.3f}   (τ = {tau0:.3f} for n={n_base})")
        return arts

    fig.subplots_adjust(top=0.82, bottom=0.13, wspace=0.1,
                        left=0.08, right=0.97)

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
