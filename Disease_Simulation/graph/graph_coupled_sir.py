"""
graph_coupled_sir.py
================================================================
county graph 위에서 '이웃 감염 압력(graph-coupled infection pressure)'으로
감염이 edge 를 따라 파동처럼 퍼지는 것을 확인하는 단계.

★ 아직 사람 이동(diffusion)은 없다. S/I/R 를 county 사이로 옮기지 않는다.
  이웃 county 의 감염자 비율이 '감염 압력'으로 작용해 현재 county 의 S 를
  추가로 감염시킬 뿐이다. N 은 county 마다 고정된다.

각 county i:
    dS_i/dt = -beta*S_i*I_i/N_i  -  alpha*S_i * Σ_j w_ij * (I_j/N_j)
    dI_i/dt =  beta*S_i*I_i/N_i  +  alpha*S_i * Σ_j w_ij * (I_j/N_j) - gamma*I_i
    dR_i/dt =  gamma*I_i

  beta  : county 내부 감염률
  gamma : 회복률
  alpha : 이웃으로부터 받는 감염 압력 강도
  w_ij  : edge weight. 처음에는 1/degree(i) 로 정규화
          → 이웃이 많은 county 가 과도한 압력을 받지 않게 함(이웃 감염비율의 평균).

이전 단계 재사용:
  county_graph.py  : GeoJSON → graph
  sir_init.py      : node 에 N,S,I,R 초기화
  local_sir.py     : totals / 보존검증 / 지도·애니메이션 helper

실행:
  python graph/graph_coupled_sir.py --state CA --geoid 06075
  python graph/graph_coupled_sir.py --state CA --geoid 06075 --gif
  python graph/graph_coupled_sir.py --state CA --geoid 06075 --show
  python graph/graph_coupled_sir.py --state CA --geoid 06075 --no-sweep   # alpha 비교 생략
"""

import os
import argparse
import matplotlib.pyplot as plt

from county_graph import (
    load_counties, compute_centroids, build_adjacency_graph, OUT_DIR,
)
from sir_init import initialize_sir_compartments, check_sir_consistency
from local_sir import (
    totals, assert_conservation, INFECTED_THRESHOLD,
    _save, plot_final_map, animate_local_sir,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ── 1. 한 timestep: graph-coupled 감염 압력 (동시 업데이트) ──────
def step_graph_coupled_sir(G, beta, gamma, alpha, dt):
    """
    모든 county 를 한 step 전진. 이웃 감염 압력을 포함한다.

    반드시 '먼저 전부 계산 → 나중에 일괄 적용'.
    이웃 값 I_j/N_j 는 모두 '업데이트 전(현재)' 값에서 읽으므로
    업데이트 순서와 무관(simultaneous update).
    edge weight w_ij = 1/degree(i) → Σ_j w_ij*(I_j/N_j) = 이웃 감염비율의 평균.
    """
    new_vals = {}
    for i in G.nodes:
        d = G.nodes[i]
        S, I, R, N = d["S"], d["I"], d["R"], d["N"]
        deg = G.degree(i)
        w = (1.0 / deg) if deg > 0 else 0.0

        # 이웃 감염 압력 (현재 값에서 읽음)
        neighbor_pressure = 0.0
        for j in G.neighbors(i):
            dj = G.nodes[j]
            if dj["N"] > 0:
                neighbor_pressure += w * (dj["I"] / dj["N"])

        # local 과 neighbor 를 분리해서 계산
        local_infection = beta * S * I / N if N > 0 else 0.0
        neighbor_infection = alpha * S * neighbor_pressure

        # 신규 감염자는 현재 S 를, 회복자는 현재 I 를 넘을 수 없다
        total_new_infections = min(S, (local_infection + neighbor_infection) * dt)
        new_recoveries = min(I, gamma * I * dt)

        nS = S - total_new_infections
        nI = I + total_new_infections - new_recoveries
        nR = R + new_recoveries
        new_vals[i] = (nS, nI, nR)

    # 일괄 적용
    for i, (nS, nI, nR) in new_vals.items():
        d = G.nodes[i]
        d["S"], d["I"], d["R"] = nS, nI, nR
    return G


# ── 2. 여러 날 시뮬레이션 ────────────────────────────────────────
def run_graph_coupled_sir(G, beta, gamma, alpha, dt, days,
                          frame_nodes=None, n_frames=150, verbose=True):
    """
    days 일 시뮬레이션. 매 step total S/I/R/N, 감염 county 수, '신규 감염 county 수' 기록.
    frame_nodes 주면 애니메이션용 I/N 스냅샷도 저장.
    """
    n_steps = int(round(days / dt))
    hist = {"time": [], "S": [], "I": [], "R": [], "N": [],
            "infected": [], "newly": [], "frames": []}

    record_frames = frame_nodes is not None
    frame_every = max(1, n_steps // n_frames) if record_frames else 0

    def snapshot(t):
        ratio = [(G.nodes[g]["I"] / G.nodes[g]["N"] if G.nodes[g]["N"] > 0 else 0.0)
                 for g in frame_nodes]
        tS, tI, tR, tN, _ = totals(G)
        hist["frames"].append({"t": t, "ratio": ratio, "I": tI, "R": tR})

    def infected_set():
        return {g for g in G.nodes if G.nodes[g]["I"] > INFECTED_THRESHOLD}

    prev_inf = infected_set()
    tS, tI, tR, tN, inf = totals(G)
    hist["time"].append(0.0)
    hist["S"].append(tS); hist["I"].append(tI); hist["R"].append(tR)
    hist["N"].append(tN); hist["infected"].append(inf); hist["newly"].append(len(prev_inf))
    if record_frames:
        snapshot(0.0)

    if verbose:
        print(f"[run] beta={beta}, gamma={gamma}, alpha={alpha}, dt={dt}, days={days} "
              f"({n_steps} steps)  R0_local={beta/gamma:.2f}")

    worst_err = 0.0
    for k in range(1, n_steps + 1):
        step_graph_coupled_sir(G, beta, gamma, alpha, dt)

        ok, bad = assert_conservation(G)
        if not ok:
            print(f"[run] ❌ step {k}: S+I+R != N at {bad}")

        cur_inf = infected_set()
        newly = len(cur_inf - prev_inf)
        prev_inf = cur_inf

        tS, tI, tR, tN, inf = totals(G)
        worst_err = max(worst_err, abs(tS + tI + tR - tN))

        hist["time"].append(k * dt)
        hist["S"].append(tS); hist["I"].append(tI); hist["R"].append(tR)
        hist["N"].append(tN); hist["infected"].append(inf); hist["newly"].append(newly)

        if record_frames and k % frame_every == 0:
            snapshot(k * dt)

    if verbose:
        print(f"[run] 완료. 전체 보존 최대 오차 |S+I+R-N| = {worst_err:.3e} (tolerance 내 = 정상)")
        if record_frames:
            print(f"[run] 애니메이션 프레임 {len(hist['frames'])}장 기록")
    return hist


# ── 3. 시각화 ────────────────────────────────────────────────────
def plot_timeseries(hist, out_path):
    """전체 S/I/R 시계열 (이제 확산으로 I·R 이 커지므로 한 패널로 충분)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    t = hist["time"]
    ax.plot(t, hist["S"], color="steelblue", lw=2, label="S (susceptible)")
    ax.plot(t, hist["I"], color="crimson",   lw=2, label="I (infected)")
    ax.plot(t, hist["R"], color="seagreen",  lw=2, label="R (recovered)")
    ax.set_xlabel("time (days)"); ax.set_ylabel("People (total across counties)")
    ax.set_title("Graph-coupled SIR — total S/I/R (spread by neighbor infection pressure)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_infected_counties(hist, out_path):
    """감염 county 수(누적적 확산) + 신규 감염 county 수."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist["time"], hist["infected"], color="darkorange", lw=2,
            label=f"Infected counties (I > {INFECTED_THRESHOLD})")
    ax.set_xlabel("time (days)"); ax.set_ylabel("Number of counties")
    ax.set_title("Infected counties — should grow along edges (spread checkpoint)")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(hist["time"], hist["newly"], width=hist["time"][1] if len(hist["time"]) > 1 else 0.1,
            color="steelblue", alpha=0.35, label="Newly infected counties (per step)")
    ax2.set_ylabel("Newly infected counties", color="steelblue")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    _save(fig, out_path)


# ── 4. alpha 민감도 비교 ─────────────────────────────────────────
def run_alpha_sweep(G, seeds, pop, initial_I, beta, gamma, dt, days,
                    alpha_values):
    """
    여러 alpha 로 각각 시뮬레이션해 전파 지표를 비교 출력.
    (매번 compartment 를 재초기화 → 공정 비교. graph 구조는 그대로 재사용.)
    """
    print("\n================ ALPHA 민감도 비교 ================")
    print(f"{'alpha':>6} | {'peak day':>9} | {'peak I':>12} | "
          f"{'final R':>12} | {'max 감염county':>14}")
    print("-" * 66)
    rows = []
    for a in alpha_values:
        initialize_sir_compartments(G, pop, seeds, initial_I)
        hist = run_graph_coupled_sir(G, beta, gamma, a, dt, days, verbose=False)
        I_series = hist["I"]
        pk = max(range(len(I_series)), key=lambda i: I_series[i])
        peak_day = hist["time"][pk]
        peak_I = I_series[pk]
        final_R = hist["R"][-1]
        max_cnt = max(hist["infected"])
        rows.append((a, peak_day, peak_I, final_R, max_cnt))
        print(f"{a:>6.2f} | {peak_day:>9.1f} | {peak_I:>12,.0f} | "
              f"{final_R:>12,.0f} | {max_cnt:>14d}")
    print("=" * 66)

    # alpha 커질수록 전파 빨라지는지(=peak day 앞당겨지는지) 판정
    faster = all(rows[k][1] >= rows[k + 1][1] for k in range(len(rows) - 1))
    if faster:
        print("→ alpha ↑ 일수록 peak day 가 앞당겨짐 = 전파가 빨라짐 ✅\n")
    else:
        print("→ peak day 가 단조롭게 앞당겨지진 않음(파라미터/포화 영향)\n")
    return rows


# ── 5. 결과 해석 ─────────────────────────────────────────────────
def interpret(hist, seeds, G):
    I_series = hist["I"]
    pk = max(range(len(I_series)), key=lambda i: I_series[i])
    peak_day = hist["time"][pk]
    peak_I = I_series[pk]
    final_R = hist["R"][-1]
    max_cnt = max(hist["infected"])

    now_infected = {g for g in G.nodes if G.nodes[g]["I"] > INFECTED_THRESHOLD}
    ever_spread = max_cnt > len(seeds)

    print("\n================ 결과 해석 ================")
    print(f"peak infected day       : {peak_day:.1f} day")
    print(f"peak total infected     : {peak_I:,.0f} 명")
    print(f"final recovered (R)     : {final_R:,.0f} 명")
    print(f"max infected county 수  : {max_cnt} / {G.number_of_nodes()}")
    print(f"seed county             : {sorted(seeds)}  ({len(seeds)}곳)")
    if ever_spread:
        print("확산 여부               : ✅ seed 밖으로 퍼짐 (edge 따라 전파됨 → 정상)")
    else:
        print("확산 여부               : ❌ seed 밖으로 안 퍼짐 (alpha 가 너무 작을 수 있음)")
    print("==========================================\n")


# ── main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="graph-coupled infection pressure SIR")
    p.add_argument("--state", default="CA",
                   help="주(예: CA). 미국 본토 전체는 US / all / conus")
    p.add_argument("--geoid", default="06075", help="seed GEOID(쉼표로 여러 개)")
    p.add_argument("--pop", type=int, default=10000)
    p.add_argument("--infected", type=int, default=10, help="seed 초기 I")
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--gamma", type=float, default=0.10)
    p.add_argument("--alpha", type=float, default=0.08)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--gif", action="store_true", help="확산 애니메이션 GIF 저장")
    p.add_argument("--show", action="store_true", help="저장 없이 창으로 재생")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no-sweep", action="store_true", help="alpha 민감도 비교 생략")
    args = p.parse_args()

    # 1) graph (한 번만 생성, 이후 재초기화로 재사용)
    #    US / all / conus → 미국 본토 전체(state=None)
    state = None if str(args.state).lower() in {"us", "all", "conus"} else args.state
    gdf = compute_centroids(load_counties(state=state))
    G = build_adjacency_graph(gdf)
    seeds = [g.strip() for g in args.geoid.split(",") if g.strip()]

    # 2) 메인 실행 (권장 파라미터)
    initialize_sir_compartments(G, args.pop, seeds, args.infected)
    check_sir_consistency(G)
    want_anim = args.gif or args.show
    hist = run_graph_coupled_sir(
        G, args.beta, args.gamma, args.alpha, args.dt, args.days,
        frame_nodes=(list(gdf.index) if want_anim else None))

    # 3) 시각화
    plot_timeseries(hist, os.path.join(OUT_DIR, "graph_coupled_sir_timeseries.png"))
    plot_infected_counties(hist, os.path.join(OUT_DIR, "graph_coupled_sir_infected_counties.png"))
    plot_final_map(gdf, G, args.days, os.path.join(OUT_DIR, "graph_coupled_sir_final_map.png"))
    if want_anim:
        gif_path = os.path.join(OUT_DIR, "graph_coupled_sir_animation.gif") if args.gif else None
        _anim = animate_local_sir(
            gdf, hist, args.days, out_path=gif_path, fps=args.fps, show=args.show,
            curve_title="Total epidemic curve (spread along edges)")

    # 4) 해석
    interpret(hist, seeds, G)

    # 5) alpha 민감도 비교
    if not args.no_sweep:
        run_alpha_sweep(G, seeds, args.pop, args.infected,
                        args.beta, args.gamma, args.dt, args.days,
                        alpha_values=[0.02, 0.05, 0.08, 0.12])

    print("다음 단계 힌트: 이제 S/I/R 자체를 edge 로 옮기는 graph Laplacian diffusion "
          "(사람 이동)을 넣으면 PDE 기반 metapopulation 으로 확장된다.")
    return G, gdf, hist


if __name__ == "__main__":
    main()
