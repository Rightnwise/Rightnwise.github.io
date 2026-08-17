"""
local_sir.py
================================================================
county graph(G) 의 각 node 안에서만 진행되는 'local SIR dynamics' 시뮬레이션.

이번 단계의 핵심:
  · 각 county 는 서로 독립적인 SIR 미분방정식을 따른다.
  · county 간 diffusion / mobility / edge 기반 전파는 절대 넣지 않는다.
    → 감염은 seed county 밖으로 퍼지지 않는 것이 '정상'이며, 그것을 확인하는 단계.

각 county i 의 방정식:
    dS_i/dt = -beta * S_i * I_i / N_i
    dI_i/dt =  beta * S_i * I_i / N_i - gamma * I_i
    dR_i/dt =  gamma * I_i

수치기법: 명시적 오일러(전진차분), dt 간격.
  · 한 step 은 '모든 node 의 new 값 먼저 계산 → 마지막에 일괄 적용'(동시 업데이트).
  · new_infections <= S, new_recoveries <= I 로 음수 방지.

이전 단계 코드 재사용:
  county_graph.py  : GeoJSON → graph
  sir_init.py      : node 에 N,S,I,R 초기화

실행:
  python graph/local_sir.py --state CA --geoid 06075
  python graph/local_sir.py --state CA --geoid 06075 --beta 0.35 --gamma 0.10 --dt 0.1 --days 160
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from county_graph import (
    load_counties, compute_centroids, build_adjacency_graph, OUT_DIR,
)
from sir_init import initialize_sir_compartments, check_sir_consistency

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# I 가 이 값보다 크면 '감염 중인 county' 로 센다(0.5명 미만이면 사실상 소멸).
INFECTED_THRESHOLD = 0.5


# ── 1. 한 timestep 업데이트 (동시 업데이트) ──────────────────────
def step_local_sir(G, beta, gamma, dt):
    """
    모든 county node 를 한 step(dt) 만큼 전진시킨다.

    반드시 '먼저 전부 계산 → 나중에 일괄 적용' 해야 앞 node 수정이
    같은 step 의 뒤 node 계산에 영향을 주지 않는다(동시 업데이트).
    음수 방지: new_infections <= S, new_recoveries <= I.
    """
    new_vals = {}
    for geoid in G.nodes:
        d = G.nodes[geoid]
        S, I, R, N = d["S"], d["I"], d["R"], d["N"]

        if N > 0:
            new_infections = beta * S * I / N * dt
            new_recoveries = gamma * I * dt
        else:
            new_infections = 0.0
            new_recoveries = 0.0

        # 음수/초과 방지
        new_infections = min(new_infections, S)   # S 보다 많이 감염될 수 없다
        new_recoveries = min(new_recoveries, I)   # I 보다 많이 회복될 수 없다

        nS = S - new_infections
        nI = I + new_infections - new_recoveries
        nR = R + new_recoveries
        new_vals[geoid] = (nS, nI, nR)

    # 일괄 적용
    for geoid, (nS, nI, nR) in new_vals.items():
        d = G.nodes[geoid]
        d["S"], d["I"], d["R"] = nS, nI, nR
    return G


# ── 2. 전체 값 집계 & 보존 검증 ──────────────────────────────────
def totals(G):
    """total S/I/R/N 과 감염 중 county 수를 반환."""
    tS = tI = tR = tN = 0.0
    infected = 0
    for geoid in G.nodes:
        d = G.nodes[geoid]
        tS += d["S"]; tI += d["I"]; tR += d["R"]; tN += d["N"]
        if d["I"] > INFECTED_THRESHOLD:
            infected += 1
    return tS, tI, tR, tN, infected


def assert_conservation(G, tol=1e-6):
    """모든 county 에서 S+I+R == N 인지 tolerance 로 확인(부동소수 오차 허용)."""
    for geoid in G.nodes:
        d = G.nodes[geoid]
        # 상대 tolerance: 값이 커도 안전
        if abs(d["S"] + d["I"] + d["R"] - d["N"]) > tol * d["N"] + 1e-9:
            return False, geoid
    return True, None


# ── 3. 여러 날 시뮬레이션 ────────────────────────────────────────
def run_local_sir_simulation(G, beta, gamma, dt, days,
                             frame_nodes=None, n_frames=150):
    """
    days 일 동안 local SIR 를 돌리고 매 timestep 값을 기록한다.
    반환: history dict (time, S, I, R, N, infected_count 리스트)

    frame_nodes 를 주면(=애니메이션용) 일정 간격으로 각 county 의 I/N 스냅샷을
    hist["frames"] 에 함께 저장한다(그 order 는 frame_nodes 순서와 동일).
    """
    n_steps = int(round(days / dt))
    hist = {"time": [], "S": [], "I": [], "R": [], "N": [], "infected": []}

    # 애니메이션 프레임 기록 설정
    record_frames = frame_nodes is not None
    frame_every = max(1, n_steps // n_frames) if record_frames else 0
    hist["frames"] = []

    def snapshot(t):
        ratio = [(G.nodes[g]["I"] / G.nodes[g]["N"] if G.nodes[g]["N"] > 0 else 0.0)
                 for g in frame_nodes]
        tS, tI, tR, tN, _ = totals(G)
        hist["frames"].append({"t": t, "ratio": ratio, "I": tI, "R": tR})

    # t=0 초기 상태 기록
    tS, tI, tR, tN, inf = totals(G)
    hist["time"].append(0.0)
    hist["S"].append(tS); hist["I"].append(tI); hist["R"].append(tR)
    hist["N"].append(tN); hist["infected"].append(inf)
    if record_frames:
        snapshot(0.0)

    print(f"[run] beta={beta}, gamma={gamma}, dt={dt}, days={days} "
          f"({n_steps} steps) 시작  (R0={beta/gamma:.2f})")

    worst_err = 0.0
    for k in range(1, n_steps + 1):
        step_local_sir(G, beta, gamma, dt)

        ok, bad = assert_conservation(G)
        if not ok:
            print(f"[run] ❌ step {k}: S+I+R != N at {bad}")

        tS, tI, tR, tN, inf = totals(G)
        # 보존 오차(전체 기준) 추적
        worst_err = max(worst_err, abs(tS + tI + tR - tN))

        hist["time"].append(k * dt)
        hist["S"].append(tS); hist["I"].append(tI); hist["R"].append(tR)
        hist["N"].append(tN); hist["infected"].append(inf)

        if record_frames and k % frame_every == 0:
            snapshot(k * dt)

    print(f"[run] 완료. 전체 보존 최대 오차 |S+I+R-N| = {worst_err:.3e} "
          f"(tolerance 내 = 정상)")
    if record_frames:
        print(f"[run] 애니메이션 프레임 {len(hist['frames'])}장 기록")
    return hist


# ── 4. 시각화 ────────────────────────────────────────────────────
def plot_timeseries(hist, out_path):
    """
    total S, I, R 시계열 line plot.
    seed county 만 유행하므로 전체 S 가 매우 커서 I/R 곡선이 묻힌다.
    → 위: 요청대로 전체 S/I/R, 아래: I/R 만 확대해 실제 SIR 곡선을 보이게.
    """
    t = hist["time"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # (위) 전체 S/I/R
    ax1.plot(t, hist["S"], color="steelblue", lw=2, label="S (susceptible)")
    ax1.plot(t, hist["I"], color="crimson",   lw=2, label="I (infected)")
    ax1.plot(t, hist["R"], color="seagreen",  lw=2, label="R (recovered)")
    ax1.set_ylabel("People (total across counties)")
    ax1.set_title("Local SIR — total S/I/R time series (no inter-county spread)")
    ax1.legend(loc="center right"); ax1.grid(alpha=0.3)

    # (아래) I, R 만 확대 (유행이 실제로 어떤 곡선인지 보이게)
    ax2.plot(t, hist["I"], color="crimson",  lw=2, label="I (infected)")
    ax2.plot(t, hist["R"], color="seagreen", lw=2, label="R (recovered)")
    ax2.set_xlabel("time (days)")
    ax2.set_ylabel("People (I·R zoomed)")
    ax2.set_title("Zoom: seed county epidemic curve (I peak -> converges to R)")
    ax2.legend(loc="center right"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, out_path)


def plot_infected_counties(hist, out_path):
    """감염 중 county 수 시계열 line plot."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist["time"], hist["infected"], color="darkorange", lw=2)
    ax.set_xlabel("time (days)")
    ax.set_ylabel(f"Infected counties (I > {INFECTED_THRESHOLD})")
    ax.set_title("Number of infected counties (without diffusion should not exceed seed count)")
    ax.grid(alpha=0.3)
    # 정수축 느낌
    ymax = max(hist["infected"]) if hist["infected"] else 1
    ax.set_ylim(-0.2, ymax + 1)
    fig.tight_layout()
    _save(fig, out_path)


def plot_final_map(gdf, G, days, out_path):
    """마지막 timestep 의 감염 비율 I/N 을 county 지도에 색으로 표시."""
    gdf = gdf.copy()
    gdf["I"] = [G.nodes[g]["I"] for g in gdf.index]
    gdf["N"] = [G.nodes[g]["N"] for g in gdf.index]
    gdf["inf_ratio"] = gdf["I"] / gdf["N"]

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(
        column="inf_ratio", ax=ax, cmap="Reds",
        vmin=0.0, vmax=max(gdf["inf_ratio"].max(), 1e-9),
        edgecolor="0.7", linewidth=0.2,
        legend=True,
        legend_kwds={"label": "Infection ratio I / N", "shrink": 0.5},
    )
    infected = gdf[gdf["I"] > INFECTED_THRESHOLD]
    if len(infected):
        cx = [G.nodes[g]["x"] for g in infected.index]
        cy = [G.nodes[g]["y"] for g in infected.index]
        ax.scatter(cx, cy, s=60, facecolor="none", edgecolor="blue",
                   linewidth=1.5, zorder=5,
                   label=f"Infected counties ({len(infected)})")
        ax.legend(loc="lower left")
    ax.set_title(f"Final infection ratio I/N  (day {days})", fontsize=14)
    ax.set_axis_off(); ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, out_path)


def animate_local_sir(gdf, hist, days, out_path=None, fps=20, show=False,
                      map_label="Infection ratio I / N", curve_title="seed county epidemic curve"):
    """
    시뮬레이션 애니메이션.
    왼쪽 : county 지도의 I/N 이 시간에 따라 변하는 모습
    오른쪽: 같은 시각까지의 SIR(I·R) 곡선 + 현재 시점 표시선
    (diffusion 이 없으므로 seed county 만 붉어졌다가 식는다 → 정상)

    out_path 가 있으면 GIF 로 저장하고, show=True 면 창으로 재생한다(둘 다 가능).
    반환한 anim 객체는 창 재생 동안 살아 있어야 하므로 호출부에서 잡아둔다.
    """
    frames = hist["frames"]
    if not frames:
        print("[anim] 프레임이 없어 애니메이션 생략")
        return None

    # 색 스케일: 전체 프레임 통틀어 최대 I/N
    vmax = max(max(f["ratio"]) for f in frames) or 1e-9

    # frames 의 ratio 는 gdf.index 순서(=frame_nodes). GEOID→ratio 로 다루기 위한 매핑.
    order = list(gdf.index)

    def ratio_by_geoid(frame):
        return {order[i]: frame["ratio"][i] for i in range(len(order))}

    # MultiPolygon(섬 포함 카운티) 때문에 폴리곤 개수 != 행 개수가 될 수 있어,
    # 단일 폴리곤으로 explode 해서 collection 길이와 색 배열을 정확히 맞춘다.
    gdf_e = gdf.reset_index().explode(index_parts=False).reset_index(drop=True)
    geoids_e = gdf_e["GEOID"].tolist()

    fig, (axm, axc) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.3, 1]})

    # ── 왼쪽: 지도(첫 프레임으로 그려두고 매 프레임 색만 갱신) ──
    d0 = ratio_by_geoid(frames[0])
    gdf_e["ratio"] = [d0[g] for g in geoids_e]
    gdf_e.plot(column="ratio", ax=axm, cmap="Reds", vmin=0.0, vmax=vmax,
               edgecolor="0.7", linewidth=0.2, legend=True,
               legend_kwds={"label": map_label, "shrink": 0.5})
    coll = axm.collections[0]           # county 폴리곤 collection
    axm.set_axis_off(); axm.set_aspect("equal")
    map_title = axm.set_title("day 0", fontsize=14)

    # ── 오른쪽: SIR 곡선(전체 미리 그리고, 시간선만 이동) ──
    t = hist["time"]
    axc.plot(t, hist["I"], color="crimson",  lw=2, label="I (infected)")
    axc.plot(t, hist["R"], color="seagreen", lw=2, label="R (recovered)")
    axc.set_xlabel("time (days)"); axc.set_ylabel("People (total)")
    axc.set_title(curve_title)
    axc.legend(loc="center right"); axc.grid(alpha=0.3)
    vline = axc.axvline(0.0, color="0.3", lw=1.2, ls="--")

    def update(i):
        f = frames[i]
        d = ratio_by_geoid(f)
        coll.set_array(np.array([d[g] for g in geoids_e]))
        vline.set_xdata([f["t"], f["t"]])
        map_title.set_text(f"day {f['t']:.0f}   (total I={f['I']:,.0f})")
        return coll, vline, map_title

    interval = 1000 / fps      # 창 재생 속도(ms/frame)
    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval, blit=False)

    # (1) 파일로 저장
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            os.remove(out_path)
        print(f"[anim] GIF 렌더링 중... ({len(frames)} 프레임, fps={fps})")
        anim.save(out_path, writer=PillowWriter(fps=fps))
        print(f"[anim] 저장 완료: {out_path}")

    # (2) 창으로 재생(저장 없이도 가능)
    if show:
        print("[anim] 창으로 재생합니다. 창을 닫으면 종료됩니다.")
        plt.show()
    else:
        plt.close(fig)
    return anim


def _save(fig, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장 완료: {out_path}")


# ── 5. 결과 해석 출력 ────────────────────────────────────────────
def interpret(hist, seed_geoids, G):
    """peak day/count, 최종 R, seed 밖 확산 여부를 출력."""
    I_series = hist["I"]
    peak_idx = max(range(len(I_series)), key=lambda i: I_series[i])
    peak_day = hist["time"][peak_idx]
    peak_I = I_series[peak_idx]
    final_R = hist["R"][-1]

    # seed 밖으로 퍼졌는지: 현재 감염 중 county 가 seed 집합의 부분집합인지
    now_infected = {g for g in G.nodes if G.nodes[g]["I"] > INFECTED_THRESHOLD}
    escaped = now_infected - set(seed_geoids)
    contained = len(escaped) == 0

    print("\n================ 결과 해석 ================")
    print(f"peak infected day    : {peak_day:.1f} day")
    print(f"peak infected count  : {peak_I:,.0f} 명 (전체 합)")
    print(f"final recovered (R)  : {final_R:,.0f} 명")
    print(f"seed county          : {sorted(seed_geoids)}")
    if contained:
        print("확산 여부            : ✅ seed county 밖으로 퍼지지 않음 (diffusion 없음 → 정상)")
    else:
        print(f"확산 여부            : ❌ seed 밖 감염 county 발견: {sorted(escaped)}")
    print("==========================================\n")


# ── main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="county 내부 local SIR dynamics")
    p.add_argument("--state", default="CA", help="주(예: CA). 생략 시 CA")
    p.add_argument("--geoid", default="06075", help="seed GEOID(쉼표로 여러 개)")
    p.add_argument("--pop", type=int, default=10000)
    p.add_argument("--infected", type=int, default=10, help="seed 초기 I")
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--gamma", type=float, default=0.10)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--days", type=int, default=160)
    p.add_argument("--gif", action="store_true", help="감염 확산 애니메이션을 GIF 파일로 저장")
    p.add_argument("--show", action="store_true",
                   help="저장 없이 애니메이션을 창으로 바로 재생")
    p.add_argument("--fps", type=int, default=20, help="애니메이션 프레임률")
    args = p.parse_args()

    # 1) graph + 초기화 (이전 단계 재사용)
    gdf = load_counties(state=args.state)
    gdf = compute_centroids(gdf)
    G = build_adjacency_graph(gdf)

    seeds = [g.strip() for g in args.geoid.split(",") if g.strip()]
    initialize_sir_compartments(
        G, default_population=args.pop,
        initial_infected_geoids=seeds, initial_I=args.infected)
    check_sir_consistency(G)

    # 2) 시뮬레이션 (애니메이션을 원하면 프레임도 기록)
    want_anim = args.gif or args.show
    hist = run_local_sir_simulation(
        G, args.beta, args.gamma, args.dt, args.days,
        frame_nodes=(list(gdf.index) if want_anim else None))

    # 3) 시각화 3종
    plot_timeseries(hist, os.path.join(OUT_DIR, "local_sir_timeseries.png"))
    plot_infected_counties(hist, os.path.join(OUT_DIR, "local_sir_infected_counties.png"))
    plot_final_map(gdf, G, args.days, os.path.join(OUT_DIR, "local_sir_final_map.png"))

    # 3-b) 애니메이션 (--gif: 저장 / --show: 창 재생 / 둘 다 가능)
    if want_anim:
        gif_path = os.path.join(OUT_DIR, "local_sir_animation.gif") if args.gif else None
        _anim = animate_local_sir(gdf, hist, args.days,
                                  out_path=gif_path, fps=args.fps, show=args.show)

    # 4) 해석
    interpret(hist, seeds, G)
    print("다음 단계 힌트: step_local_sir 뒤에 edge 를 따라 I 를 이웃으로 흘리는 "
          "diffusion 항을 더하면 metapopulation 확산이 된다.")
    return G, gdf, hist


if __name__ == "__main__":
    main()
