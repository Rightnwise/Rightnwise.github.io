"""
reaction_diffusion_sir.py
================================================================
county graph 위에서 PDE 의 Laplacian diffusion 에 해당하는 항을 실제로 구현한
graph-based reaction-diffusion SIR 모델.

★ 앞 단계(infection pressure)와 다른 점:
  이제 S, I, R '사람 자체'가 edge 를 따라 이웃 county 로 이동한다(diffusion).
  → county 별 N 은 고정되지 않아도 되고(사람이 오가므로), 전체 population 만 보존된다.

각 county i (edge weight w_ij = 1):
  dS_i/dt = -beta*S_i*I_i/N_i            + D_S * Σ_j (S_j - S_i)
  dI_i/dt =  beta*S_i*I_i/N_i - gamma*I_i + D_I * Σ_j (I_j - I_i)
  dR_i/dt =  gamma*I_i                    + D_R * Σ_j (R_j - R_i)

구현 규칙:
  · 한 step = ① local SIR reaction  → ② graph Laplacian diffusion (연산자 분리)
  · diffusion 은 node 합산이 아니라 반드시 edge 별 'pairwise flux' 로:
        edge (a,b): flux = D*(val_a - val_b)*dt
        delta[a] -= flux ;  delta[b] += flux
    → 한쪽에서 빠진 양이 정확히 다른 쪽으로 들어가 전체 보존.
  · reaction·diffusion 모두 simultaneous update (전부 계산 후 일괄 적용).
  · diffusion 후 각 county N = S+I+R 로 재계산.
  · 음수(너무 큰 D/dt 로 인한 불안정)가 생기면 경고하고 0 으로 클램프.

이전 단계 재사용:
  county_graph / sir_init / local_sir (helper: _save, INFECTED_THRESHOLD,
  plot_final_map, animate_local_sir)

실행:
  python graph/reaction_diffusion_sir.py --state CA --geoid 06075
  python graph/reaction_diffusion_sir.py --state CA --geoid 06075 --gif
  python graph/reaction_diffusion_sir.py --state US --geoid 17031 --no-sweep
  # 이웃보다 S/I/R 이동을 다르게 주어 인구 재분포를 보고 싶으면:
  python graph/reaction_diffusion_sir.py --state CA --geoid 06075 --DI 0.002
"""

import os
import argparse
from time import perf_counter
import matplotlib.pyplot as plt

from county_graph import (
    load_counties, compute_centroids, build_adjacency_graph, OUT_DIR,
)
from sir_init import initialize_sir_compartments, check_sir_consistency
from population import load_county_population
from numpy_backend import (
    run_reaction_diffusion_simulation_numpy, numpy_state_to_graph,
    compare_networkx_vs_numpy_backend,
)
from local_sir import (
    INFECTED_THRESHOLD, _save, plot_final_map, animate_local_sir,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ── 1. 한 timestep: reaction + Laplacian diffusion (pairwise flux) ─
def step_reaction_diffusion_sir(G, beta, gamma, D_S, D_I, D_R, dt):
    """
    ① local SIR reaction (simultaneous) → ② diffusion (pairwise flux, simultaneous).
    반환: 이번 step 에서 음수가 발생했는지(bool).
    """
    # ── ① reaction : 각 node 독립, 먼저 전부 계산 후 일괄 적용 ──
    reacted = {}
    for i in G.nodes:
        d = G.nodes[i]
        S, I, R, N = d["S"], d["I"], d["R"], d["N"]
        new_infections = min(S, beta * S * I / N * dt) if N > 0 else 0.0
        new_recoveries = min(I, gamma * I * dt)
        reacted[i] = (S - new_infections,
                      I + new_infections - new_recoveries,
                      R + new_recoveries)
    for i, (S, I, R) in reacted.items():
        d = G.nodes[i]
        d["S"], d["I"], d["R"] = S, I, R

    # ── ② diffusion : edge 별 pairwise flux 를 delta 에 누적 후 일괄 적용 ──
    delta = {i: [0.0, 0.0, 0.0] for i in G.nodes}   # [S, I, R]
    for a, b in G.edges():
        da, db = G.nodes[a], G.nodes[b]
        flux_S = D_S * (da["S"] - db["S"]) * dt
        flux_I = D_I * (da["I"] - db["I"]) * dt
        flux_R = D_R * (da["R"] - db["R"]) * dt
        delta[a][0] -= flux_S; delta[b][0] += flux_S
        delta[a][1] -= flux_I; delta[b][1] += flux_I
        delta[a][2] -= flux_R; delta[b][2] += flux_R

    neg = False
    for i in G.nodes:
        d = G.nodes[i]
        d["S"] += delta[i][0]
        d["I"] += delta[i][1]
        d["R"] += delta[i][2]
        # 음수 방지(불안정 신호) — 발생 시 경고 후 0 으로 클램프
        if d["S"] < -1e-9 or d["I"] < -1e-9 or d["R"] < -1e-9:
            neg = True
            d["S"] = max(d["S"], 0.0)
            d["I"] = max(d["I"], 0.0)
            d["R"] = max(d["R"], 0.0)
        # diffusion 후 N 재계산 (사람이 오갔으므로)
        d["N"] = d["S"] + d["I"] + d["R"]
    return neg


# ── 1-b. SIRS reaction-diffusion step (NetworkX) ─────────────────
def step_reaction_diffusion_sirs(G, beta, gamma, omega, D_S, D_I, D_R, dt):
    """
    SIR step 과 동일하되 local reaction 에 면역소실 ω·R(회복→감수성) 추가.
    diffusion(pairwise flux)·음수처리·N 재계산은 기존과 완전히 동일.
    omega=0 이면 step_reaction_diffusion_sir 과 같은 결과.
    """
    # ── ① reaction (+ waning immunity) ──
    reacted = {}
    for i in G.nodes:
        d = G.nodes[i]
        S, I, R, N = d["S"], d["I"], d["R"], d["N"]
        new_infections = min(S, beta * S * I / N * dt) if N > 0 else 0.0
        new_recoveries = min(I, gamma * I * dt)
        waning_immunity = min(R, omega * R * dt)
        reacted[i] = (S - new_infections + waning_immunity,
                      I + new_infections - new_recoveries,
                      R + new_recoveries - waning_immunity)
    for i, (S, I, R) in reacted.items():
        d = G.nodes[i]
        d["S"], d["I"], d["R"] = S, I, R

    # ── ② diffusion (기존 SIR 과 동일한 pairwise flux) ──
    delta = {i: [0.0, 0.0, 0.0] for i in G.nodes}
    for a, b in G.edges():
        da, db = G.nodes[a], G.nodes[b]
        flux_S = D_S * (da["S"] - db["S"]) * dt
        flux_I = D_I * (da["I"] - db["I"]) * dt
        flux_R = D_R * (da["R"] - db["R"]) * dt
        delta[a][0] -= flux_S; delta[b][0] += flux_S
        delta[a][1] -= flux_I; delta[b][1] += flux_I
        delta[a][2] -= flux_R; delta[b][2] += flux_R

    neg = False
    for i in G.nodes:
        d = G.nodes[i]
        d["S"] += delta[i][0]; d["I"] += delta[i][1]; d["R"] += delta[i][2]
        if d["S"] < -1e-9 or d["I"] < -1e-9 or d["R"] < -1e-9:
            neg = True
            d["S"] = max(d["S"], 0.0); d["I"] = max(d["I"], 0.0); d["R"] = max(d["R"], 0.0)
        d["N"] = d["S"] + d["I"] + d["R"]
    return neg


# ── 2. 통계 스냅샷 ───────────────────────────────────────────────
def snapshot_stats(G):
    """전체 S/I/R/N, 감염 county 수, 최대 감염비율, 최소·최대 인구, 최소 compartment."""
    tS = tI = tR = tN = 0.0
    infected = 0
    max_ratio = 0.0
    min_pop = float("inf"); max_pop = 0.0
    min_comp = float("inf")
    for i in G.nodes:
        d = G.nodes[i]
        S, I, R, N = d["S"], d["I"], d["R"], d["N"]
        tS += S; tI += I; tR += R; tN += N
        if I > INFECTED_THRESHOLD:
            infected += 1
        if N > 0:
            max_ratio = max(max_ratio, I / N)
        min_pop = min(min_pop, N); max_pop = max(max_pop, N)
        min_comp = min(min_comp, S, I, R)
    return {"S": tS, "I": tI, "R": tR, "N": tN, "infected": infected,
            "max_ratio": max_ratio, "min_pop": min_pop, "max_pop": max_pop,
            "min_comp": min_comp}


# ── 3. 여러 날 시뮬레이션 ────────────────────────────────────────
def run_reaction_diffusion_sir(G, beta, gamma, D_S, D_I, D_R, dt, days,
                               frame_nodes=None, n_frames=150, verbose=True,
                               step_fn=None):
    # step_fn 을 주면(예: weighted step) 그 step 함수로 진행. 기본은 unweighted.
    if step_fn is None:
        step_fn = step_reaction_diffusion_sir
    n_steps = int(round(days / dt))
    initial_total_N = snapshot_stats(G)["N"]
    hist = {k: [] for k in ("time", "S", "I", "R", "N", "infected",
                            "max_ratio", "min_pop", "max_pop")}
    hist["frames"] = []
    hist["global_min_comp"] = float("inf")
    hist["max_cons_err"] = 0.0

    record_frames = frame_nodes is not None
    frame_every = max(1, n_steps // n_frames) if record_frames else 0

    def snapshot_frame(t, tI, tR):
        ratio = [(G.nodes[g]["I"] / G.nodes[g]["N"] if G.nodes[g]["N"] > 0 else 0.0)
                 for g in frame_nodes]
        hist["frames"].append({"t": t, "ratio": ratio, "I": tI, "R": tR})

    def record(t, take_frame):
        s = snapshot_stats(G)
        hist["time"].append(t)
        for k in ("S", "I", "R", "N", "infected", "max_ratio", "min_pop", "max_pop"):
            hist[k].append(s[k])
        hist["global_min_comp"] = min(hist["global_min_comp"], s["min_comp"])
        hist["max_cons_err"] = max(hist["max_cons_err"],
                                   abs(s["S"] + s["I"] + s["R"] - initial_total_N),
                                   abs(s["N"] - initial_total_N))
        if record_frames and take_frame:
            snapshot_frame(t, s["I"], s["R"])
        return s

    record(0.0, take_frame=True)
    if verbose:
        print(f"[run] beta={beta}, gamma={gamma}, D_S={D_S}, D_I={D_I}, D_R={D_R}, "
              f"dt={dt}, days={days} ({n_steps} steps)")

    neg_warned = False
    for k in range(1, n_steps + 1):
        neg = step_fn(G, beta, gamma, D_S, D_I, D_R, dt)
        if neg and not neg_warned:
            neg_warned = True
            print(f"[run] ⚠️  step {k}: 음수 S/I/R 발생 → D 또는 dt 를 줄이세요 "
                  f"(현재 dt={dt}). 0 으로 클램프하고 계속합니다.")
        # 시계열은 매 step, 애니메이션 프레임은 간격(frame_every)마다
        record(k * dt, take_frame=(record_frames and (k % frame_every == 0 or k == n_steps)))

    if verbose:
        print(f"[run] 완료. population 보존 최대 오차 = {hist['max_cons_err']:.3e} "
              f"(tolerance 내 = 정상)")
        print(f"[run] 전 기간 최소 compartment 값 = {hist['global_min_comp']:.3e} "
              f"(음수면 불안정)")
        if record_frames:
            print(f"[run] 애니메이션 프레임 {len(hist['frames'])}장 기록")
    return hist


# ── 4. 시각화 ────────────────────────────────────────────────────
def plot_timeseries(hist, out_path):
    fig, ax = plt.subplots(figsize=(11, 6))
    t = hist["time"]
    ax.plot(t, hist["S"], color="steelblue", lw=2, label="S (susceptible)")
    ax.plot(t, hist["I"], color="crimson",   lw=2, label="I (infected)")
    ax.plot(t, hist["R"], color="seagreen",  lw=2, label="R (recovered)")
    ax.set_xlabel("time (days)"); ax.set_ylabel("People (total across counties)")
    ax.set_title("Reaction-Diffusion SIR — total S/I/R (S/I/R move along edges)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_infected_counties(hist, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist["time"], hist["infected"], color="darkorange", lw=2)
    ax.set_xlabel("time (days)"); ax.set_ylabel(f"Infected counties (I > {INFECTED_THRESHOLD})")
    ax.set_title("Infected counties — should grow along edges via diffusion")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_population_map(gdf, G, days, out_path):
    """diffusion 결과 county 별 인구 N 지도(사람이 오가면 N 이 변한다)."""
    gdf = gdf.copy()
    gdf["N"] = [G.nodes[g]["N"] for g in gdf.index]
    nmin, nmax = gdf["N"].min(), gdf["N"].max()
    mean = gdf["N"].mean()

    # 균등 D + 균등 초기인구면 N 변화가 부동소수 노이즈(~1e-9)뿐이다.
    # 그 노이즈를 컬러바가 확대하지 않도록, 변화가 미미하면 넉넉한 밴드로 고정.
    if (nmax - nmin) < 1e-3 * max(mean, 1.0):
        vmin, vmax = mean * 0.9, mean * 1.1
    else:
        vmin, vmax = nmin, nmax

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(column="N", ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
             edgecolor="0.7", linewidth=0.2, legend=True,
             legend_kwds={"label": "County population N", "shrink": 0.5})
    ax.set_title(f"Population N per county  (day {days})   "
                 f"[min={nmin:,.0f}, max={nmax:,.0f}]", fontsize=13)
    ax.set_axis_off(); ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, out_path)


# ── 5. D 민감도 비교 ─────────────────────────────────────────────
def run_D_sweep(G, seeds, pop, initial_I, beta, gamma, dt, days, D_values,
                populations=None):
    print("\n================ D (diffusion) 민감도 비교 ================")
    print(f"{'D':>6} | {'peak day':>8} | {'peak I':>11} | {'final R':>12} | "
          f"{'max감염cty':>10} | {'보존오차':>10} | {'min S/I/R':>11}")
    print("-" * 86)
    rows = []
    for D in D_values:
        initialize_sir_compartments(G, pop, seeds, initial_I, populations=populations)
        hist = run_reaction_diffusion_sir(G, beta, gamma, D, D, D, dt, days,
                                          verbose=False)
        I_series = hist["I"]
        pk = max(range(len(I_series)), key=lambda i: I_series[i])
        rows.append((D, hist["time"][pk], I_series[pk], hist["R"][-1],
                     max(hist["infected"]), hist["max_cons_err"],
                     hist["global_min_comp"]))
        print(f"{D:>6.3f} | {rows[-1][1]:>8.1f} | {rows[-1][2]:>11,.0f} | "
              f"{rows[-1][3]:>12,.0f} | {rows[-1][4]:>10d} | "
              f"{rows[-1][5]:>10.1e} | {rows[-1][6]:>11.1e}")
    print("=" * 86)
    # 확산 속도 지표: D 가 커질수록 peak day 가 앞당겨지는가(=더 빨리 확산)
    faster = all(rows[k][1] >= rows[k + 1][1] for k in range(len(rows) - 1))
    if faster:
        print("→ D ↑ 일수록 peak day 가 앞당겨짐 = 공간 확산(diffusion)이 빨라짐 ✅\n")
    else:
        print("→ peak day 가 단조롭게 앞당겨지진 않음(포화/파라미터 영향)\n")
    return rows


# ── 6. 결과 해석 ─────────────────────────────────────────────────
def interpret(hist, seeds, G):
    I_series = hist["I"]
    pk = max(range(len(I_series)), key=lambda i: I_series[i])
    max_cnt = max(hist["infected"])
    pop_changed = (hist["max_pop"][-1] - hist["min_pop"][-1]) > 1e-6

    print("\n================ 결과 해석 ================")
    print(f"peak infected day        : {hist['time'][pk]:.1f} day")
    print(f"peak total infected      : {I_series[pk]:,.0f} 명")
    print(f"final recovered (R)      : {hist['R'][-1]:,.0f} 명")
    print(f"max infected county 수   : {max_cnt} / {G.number_of_nodes()}")
    print(f"population 보존 오차      : {hist['max_cons_err']:.3e}  (≈0 이면 보존)")
    print(f"전 기간 최소 S/I/R 값    : {hist['global_min_comp']:.3e}  (음수 없어야 정상)")
    print(f"county 인구 N 범위(최종) : {hist['min_pop'][-1]:,.1f} ~ {hist['max_pop'][-1]:,.1f}")
    if pop_changed:
        print("county별 population      : ✅ 변함 (S/I/R 이 edge 로 이동함)")
    else:
        print("county별 population      : 거의 불변 (D_S=D_I=D_R & 균등 초기인구 → N 보존).")
        print("                           → 인구 재분포를 보려면 D 를 다르게(예: --DI 0.002) 주세요.")
    print(f"확산 여부                : "
          f"{'✅ seed 밖으로 퍼짐 (edge 따라 전파)' if max_cnt > len(seeds) else '❌ 안 퍼짐'}")
    print("==========================================\n")


# ── main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="graph-based reaction-diffusion SIR")
    p.add_argument("--state", default="CA", help="주(예: CA). 미국 본토 전체는 US")
    p.add_argument("--geoid", default="06075", help="seed GEOID(쉼표로 여러 개)")
    p.add_argument("--pop", type=int, default=10000,
                   help="임시 인구(--real-pop 없을 때). 미매칭 county 대체값으로도 쓰임")
    p.add_argument("--real-pop", action="store_true",
                   help="data/co-est2025-pop.xlsx 의 실제 county 인구로 초기화")
    p.add_argument("--pop-year", type=int, default=2025,
                   help="실제 인구 연도(2020~2025, 기본 2025)")
    p.add_argument("--infected", type=int, default=10)
    # ── 모델 선택 (SIR 유지 / SIRS 추가) ──
    p.add_argument("--model", choices=["sir", "sirs"], default="sir",
                   help="sir(기본) 또는 sirs(면역 소실 → 재감수성)")
    p.add_argument("--recovery-days", type=float, default=30.0,
                   help="SIRS: gamma=1/recovery_days (--gamma 없을 때). 기본 30일")
    p.add_argument("--immunity-days", type=float, default=90.0,
                   help="SIRS: omega=1/immunity_days (--omega 없을 때). 기본 90일(≈3개월)")
    p.add_argument("--omega", type=float, default=None,
                   help="면역 소실률(직접 지정 시 immunity-days 무시). SIR 은 0")
    # beta/gamma/D/days 는 model 별 기본값을 쓰기 위해 default=None
    p.add_argument("--beta", type=float, default=None,
                   help="감염률(기본 0.35, sir·sirs 공통)")
    p.add_argument("--gamma", type=float, default=None,
                   help="회복률(직접 지정. 없으면 sir=0.10, sirs=1/recovery_days)")
    p.add_argument("--D", type=float, default=None,
                   help="공통 diffusion 계수(기본 0.01, sir·sirs 공통)")
    p.add_argument("--DS", type=float, default=None, help="S diffusion(생략 시 --D)")
    p.add_argument("--DI", type=float, default=None, help="I diffusion(생략 시 --D)")
    p.add_argument("--DR", type=float, default=None, help="R diffusion(생략 시 --D)")
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--days", type=int, default=None,
                   help="시뮬레이션 일수(기본 sir=200, sirs=365)")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no-sweep", action="store_true")
    # ── 성능 개선용 백엔드 옵션 ──
    p.add_argument("--backend", choices=["networkx", "numpy"], default="networkx",
                   help="시뮬레이션 백엔드(기본 networkx, numpy 가 빠름)")
    p.add_argument("--compare-backends", action="store_true",
                   help="짧은 기간으로 두 백엔드 결과 일치 검증 + speedup 출력")
    p.add_argument("--frame-interval-days", type=float, default=1.0,
                   help="numpy 백엔드 애니메이션 프레임 간격(일)")
    p.add_argument("--backend-suffix", action="store_true",
                   help="결과 파일명에 _<backend> 접미사 붙이기(기존 파일명 유지가 기본)")
    args = p.parse_args()

    # ── model 별 파라미터 해석 (SIR 기본값은 기존과 동일 → 하위호환) ──
    SMALLPOX_R0 = 5.0   # 천연두 기본 R0(문헌 3.5~7 중앙값). SIRS 기본에 사용
    model = args.model
    D = args.D if args.D is not None else 0.01
    days = args.days if args.days is not None else (365 if model == "sirs" else 200)

    # gamma 를 먼저 정해야 beta=R0*gamma 로 R0 를 고정할 수 있다
    if args.gamma is not None:
        gamma = args.gamma
        recovery_days = 1.0 / gamma if gamma > 0 else float("inf")
    elif model == "sirs":
        recovery_days = args.recovery_days
        gamma = 1.0 / recovery_days
    else:  # 기존 SIR 기본
        gamma = 0.10
        recovery_days = 1.0 / gamma

    # beta: SIRS 는 smallpox R0(=5) 기준 → beta=R0*gamma. SIR 은 기존 0.35(하위호환)
    if args.beta is not None:
        beta = args.beta
    elif model == "sirs":
        beta = SMALLPOX_R0 * gamma
    else:
        beta = 0.35

    if args.omega is not None:
        omega = args.omega
        immunity_days = (1.0 / omega) if omega > 0 else float("inf")
    elif model == "sirs":
        immunity_days = args.immunity_days
        omega = 1.0 / immunity_days
    else:  # SIR: 면역 소실 없음
        omega = 0.0
        immunity_days = float("inf")

    D_S = args.DS if args.DS is not None else D
    D_I = args.DI if args.DI is not None else D
    D_R = args.DR if args.DR is not None else D

    t_total0 = perf_counter()

    # ── graph 로딩 (시각화·초기화용, loop 밖) ──
    t0 = perf_counter()
    state = None if str(args.state).lower() in {"us", "all", "conus"} else args.state
    gdf = compute_centroids(load_counties(state=state))
    G = build_adjacency_graph(gdf)
    seeds = [g.strip() for g in args.geoid.split(",") if g.strip()]
    t_graph = perf_counter() - t0

    # 실제 인구로 초기조건을 바꿀지
    populations = None
    if args.real_pop:
        populations, _ = load_county_population(gdf, year=args.pop_year)

    def fresh_init():
        initialize_sir_compartments(G, args.pop, seeds, args.infected,
                                    populations=populations)

    # SIRS 일 때 NetworkX step 은 omega 를 캡처한 래퍼로 넘긴다(기존 run 시그니처 유지)
    nx_step = None
    if model == "sirs":
        def nx_step(Gg, b, g, ds, di, dr, dtt):
            return step_reaction_diffusion_sirs(Gg, b, g, omega, ds, di, dr, dtt)

    # ── (선택) 백엔드 일치 검증 ──
    if args.compare_backends:
        fresh_init()
        cmp_days = min(20, days)
        print(f"[compare] model={model}, days={cmp_days} 로 두 백엔드 비교…")
        compare_networkx_vs_numpy_backend(
            G, beta, gamma, D_S, D_I, D_R, args.dt, cmp_days,
            networkx_step_fn=nx_step, model=model, omega=omega)

    # ── 메인 실행 (선택한 백엔드) ──
    fresh_init()
    check_sir_consistency(G)
    want_anim = args.gif or args.show
    frame_nodes = list(gdf.index) if want_anim else None

    t0 = perf_counter()
    if args.backend == "numpy":
        hist, geoids, S, I, R = run_reaction_diffusion_simulation_numpy(
            G, beta, gamma, D_S, D_I, D_R, args.dt, days,
            frame_nodes=frame_nodes, frame_interval_days=args.frame_interval_days,
            model=model, omega=omega)
        numpy_state_to_graph(G, geoids, S, I, R)   # 지도 시각화용으로 G 반영
    else:
        hist = run_reaction_diffusion_sir(
            G, beta, gamma, D_S, D_I, D_R, args.dt, days,
            frame_nodes=frame_nodes, step_fn=nx_step)
    t_sim = perf_counter() - t0

    # ── 결과 파일명: SIRS 는 prefix 에 sirs 포함, 기존 SIR 파일명은 그대로 ──
    sfx = f"_{args.backend}" if args.backend_suffix else ""
    prefix = "sirs_reaction_diffusion" if model == "sirs" else "reaction_diffusion_sir"

    def out(name, ext):
        return os.path.join(OUT_DIR, f"{prefix}_{name}{sfx}.{ext}")

    t0 = perf_counter()
    plot_timeseries(hist, out("timeseries", "png"))
    plot_infected_counties(hist, out("infected_counties", "png"))
    plot_final_map(gdf, G, days, out("final_map", "png"))
    plot_population_map(gdf, G, days, out("population_map", "png"))
    if want_anim:
        gif_path = out("animation", "gif") if args.gif else None
        animate_local_sir(gdf, hist, days, out_path=gif_path,
                          fps=args.fps, show=args.show,
                          curve_title=f"Total epidemic curve ({model} reaction-diffusion)")
    t_plot = perf_counter() - t0

    interpret(hist, seeds, G)

    # D-sweep 은 SIR 모델에서만(sweep 은 SIR step 기반)
    if not args.no_sweep and model == "sir":
        run_D_sweep(G, seeds, args.pop, args.infected, beta, gamma,
                    args.dt, days, D_values=[0.002, 0.005, 0.01, 0.02],
                    populations=populations)

    # ── 최종 요약 (SIRS 파라미터 포함) ──
    t_total = perf_counter() - t_total0
    I_series = hist["I"]
    pk = max(range(len(I_series)), key=lambda i: I_series[i])
    print("================ RUN SUMMARY ================")
    print(f"model type              : {model}")
    print(f"beta                    : {beta:.4f}")
    print(f"gamma                   : {gamma:.5f}  (recovery_days={recovery_days:.1f})")
    print(f"R0 (=beta/gamma)        : {beta / gamma:.2f}")
    if model == "sirs":
        print(f"omega                   : {omega:.6f}  (immunity_days={immunity_days:.1f})")
    else:
        print(f"omega                   : 0  (SIR = 영구면역)")
    print(f"D_S / D_I / D_R         : {D_S} / {D_I} / {D_R}")
    print(f"backend                 : {args.backend}")
    print(f"node / edge count       : {G.number_of_nodes()} / {G.number_of_edges()}")
    print(f"timestep count          : {int(round(days / args.dt))}")
    print(f"peak infected day       : {hist['time'][pk]:.1f} day")
    print(f"peak total infected     : {I_series[pk]:,.0f}")
    print(f"final S / I / R         : {hist['S'][-1]:,.0f} / {hist['I'][-1]:,.0f} / {hist['R'][-1]:,.0f}")
    print(f"graph / sim / plot time : {t_graph:.3f} / {t_sim:.3f} / {t_plot:.3f} s")
    print(f"total runtime           : {t_total:.3f} s")
    print(f"population 보존 오차     : {hist['max_cons_err']:.3e}")
    print(f"최소 compartment 값     : {hist['global_min_comp']:.3e}")
    print(f"clamp 발생              : {hist.get('clamp_count', 0)}회"
          f"{' (음수 → D/dt 축소 권장)' if hist.get('clamp_count', 0) else ' (없음 ✅)'}")
    print(f"결과 폴더               : {OUT_DIR}")
    print("=============================================")
    return G, gdf, hist


if __name__ == "__main__":
    main()
