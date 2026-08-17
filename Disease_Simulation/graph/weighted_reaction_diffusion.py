"""
weighted_reaction_diffusion.py
================================================================
unweighted reaction-diffusion SIR(reaction_diffusion_sir.py)를 유지한 채,
GeoJSON geometry 로 계산한 edge weight 를 넣은 weighted 버전.

edge weight (PDE finite-volume 느낌):
    raw_w_ij = shared_border_length_ij / centroid_distance_ij
    (경계를 길게 공유하고 중심이 가까울수록 diffusion ↑)
그리고 raw_w>0 인 edge 들의 평균이 1 이 되도록 정규화:
    w_ij = raw_w_ij / mean(raw_w)         → 기존 D 값과 스케일 비교 쉬움

weighted pairwise flux:
    flux_X = D_X * w_ij * (X_a - X_b) * dt   (X ∈ {S, I, R})
    delta[a] -= flux ; delta[b] += flux      → 전체 S/I/R/N 보존

geometry 는 projected CRS(EPSG:5070, load_counties 가 이미 재투영)에서 계산한다.
위경도에서 직접 길이/거리를 재지 않는다.

이전 단계 재사용:
  county_graph / sir_init / population / local_sir / reaction_diffusion_sir

실행:
  python graph/weighted_reaction_diffusion.py --state CA --geoid 06075 --real-pop
  python graph/weighted_reaction_diffusion.py --state CA --geoid 06075 --gif
  python graph/weighted_reaction_diffusion.py --state CA --geoid 06075 --DS 0.002 --DI 0.002 --DR 0.002 --dt 0.02
"""

import os
import argparse
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from county_graph import (
    load_counties, compute_centroids, build_adjacency_graph, OUT_DIR,
)
from sir_init import initialize_sir_compartments, check_sir_consistency
from population import load_county_population
from local_sir import plot_final_map, animate_local_sir
from reaction_diffusion_sir import (
    run_reaction_diffusion_sir, plot_timeseries, plot_infected_counties,
    plot_population_map, step_reaction_diffusion_sirs,
)
from numpy_backend import (
    run_reaction_diffusion_simulation_numpy, numpy_state_to_graph,
    compare_networkx_vs_numpy_backend,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

EPS = 1.0   # centroid 거리 하한(m) — 0 나눗셈 방지


def edge_weight_cache_path(state):
    """state 별 edge weight 캐시 CSV 경로."""
    tag = "US" if (state is None or str(state).lower() in {"us", "all", "conus"}) \
        else str(state).upper()
    return os.path.join(OUT_DIR, f"edge_weights_border_distance_{tag}.csv")


# ── 1. edge weight 계산 (경계길이 / 중심거리) ────────────────────
def compute_edge_weights_border_distance(gdf, G, cache_path=None, recompute=False):
    """
    각 node 에 area/centroid, 각 edge 에 weight 를 저장한다.
    - shared_border_length == 0 (corner touching) → weight 0 (diffusion 안 함)
    - raw_w>0 edge 평균이 1 이 되도록 정규화
    gdf 는 projected CRS(EPSG:5070) 상태여야 한다(load_counties 가 재투영함).

    캐시: cache_path 가 있고 파일이 존재하며 recompute=False 면 CSV 에서 불러온다.
          없으면 geometry 로 계산 후 저장한다(느린 경계교차 계산을 재사용).
    """
    geom = gdf.geometry

    # (3) node geometry 값 저장 (loop 밖, 항상 필요)
    for geoid in G.nodes:
        g = geom.loc[geoid]
        G.nodes[geoid]["area"] = g.area
        G.nodes[geoid]["centroid_x"] = G.nodes[geoid]["x"]
        G.nodes[geoid]["centroid_y"] = G.nodes[geoid]["y"]

    # 캐시 로드 시도
    if cache_path and (not recompute) and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, dtype={"geoid_a": str, "geoid_b": str})
        loaded = 0
        for _, r in df.iterrows():
            a, b = r["geoid_a"], r["geoid_b"]
            if G.has_edge(a, b):
                G.edges[a, b]["shared_border_length"] = r["shared_border_length"]
                G.edges[a, b]["centroid_distance"] = r["centroid_distance"]
                G.edges[a, b]["raw_weight"] = r["raw_weight"]
                G.edges[a, b]["weight"] = r["weight"]
                loaded += 1
        if loaded == G.number_of_edges():
            print(f"[weight] 캐시 로드: {cache_path} ({loaded} edges)")
            return G
        print(f"[weight] 캐시 불완전({loaded}/{G.number_of_edges()}) → 다시 계산")

    # (4)(5) edge 별 shared/distance/raw
    shared_d, dist_d, raw = {}, {}, {}
    for a, b in G.edges():
        ga, gb = geom.loc[a], geom.loc[b]
        shared = ga.boundary.intersection(gb.boundary).length
        dist = ga.centroid.distance(gb.centroid)
        shared_d[(a, b)] = shared
        dist_d[(a, b)] = dist
        raw[(a, b)] = (shared / max(dist, EPS)) if shared > 0 else 0.0

    # (6) raw>0 평균으로 정규화
    pos = [w for w in raw.values() if w > 0]
    mean_raw = (sum(pos) / len(pos)) if pos else 1.0
    for (a, b), w in raw.items():
        G.edges[a, b]["shared_border_length"] = shared_d[(a, b)]
        G.edges[a, b]["centroid_distance"] = dist_d[(a, b)]
        G.edges[a, b]["raw_weight"] = w
        G.edges[a, b]["weight"] = (w / mean_raw) if w > 0 else 0.0

    # 캐시 저장
    if cache_path:
        recs = [{"geoid_a": a, "geoid_b": b,
                 "shared_border_length": shared_d[(a, b)],
                 "centroid_distance": dist_d[(a, b)],
                 "raw_weight": raw[(a, b)],
                 "weight": G.edges[a, b]["weight"]}
                for a, b in G.edges()]
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pd.DataFrame(recs).to_csv(cache_path, index=False)
        print(f"[weight] 캐시 저장: {cache_path} ({len(recs)} edges)")

    return G


# ── 2. weight summary 출력 ───────────────────────────────────────
def print_weight_summary(G):
    edges = list(G.edges(data=True))
    weights = [d["weight"] for _, _, d in edges]
    pos = [(u, v, d["weight"]) for u, v, d in edges if d["weight"] > 0]
    zero_cnt = sum(1 for w in weights if w == 0)

    print("\n================ EDGE WEIGHT SUMMARY ================")
    print(f"total edge count          : {len(edges)}")
    print(f"positive-weight edge count: {len(pos)}")
    print(f"zero-weight edge count    : {zero_cnt}  (corner-touching 등)")
    if pos:
        pw = [w for _, _, w in pos]
        print(f"min / mean / max weight   : {min(pw):.3f} / "
              f"{sum(pw)/len(pw):.3f} / {max(pw):.3f}")

        def name(g):
            return G.nodes[g].get("name", g)
        strongest = sorted(pos, key=lambda e: e[2], reverse=True)[:10]
        weakest = sorted(pos, key=lambda e: e[2])[:10]
        print("── strongest 10 edges (w 큰 = 경계 길고 가까움) ──")
        for u, v, w in strongest:
            print(f"   {w:7.3f}  {name(u)} — {name(v)}")
        print("── weakest 10 positive edges ──")
        for u, v, w in weakest:
            print(f"   {w:7.3f}  {name(u)} — {name(v)}")
    print("====================================================\n")


# ── 3. weighted step (reaction + weighted pairwise flux) ─────────
def step_weighted_reaction_diffusion_sir(G, beta, gamma, D_S, D_I, D_R, dt):
    """unweighted step 과 동일하되 diffusion flux 에 edge weight w 를 곱한다."""
    # ① reaction (각 node 독립, simultaneous)
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

    # ② weighted diffusion (pairwise flux)
    delta = {i: [0.0, 0.0, 0.0] for i in G.nodes}
    for a, b, ed in G.edges(data=True):
        w = ed["weight"]
        if w == 0:
            continue
        da, db = G.nodes[a], G.nodes[b]
        flux_S = D_S * w * (da["S"] - db["S"]) * dt
        flux_I = D_I * w * (da["I"] - db["I"]) * dt
        flux_R = D_R * w * (da["R"] - db["R"]) * dt
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


# ── 3-b. weighted SIRS step (면역 소실 ω·R 추가) ────────────────
def step_weighted_reaction_diffusion_sirs(G, beta, gamma, omega,
                                          D_S, D_I, D_R, dt):
    """weighted SIR step 과 동일하되 reaction 에 면역소실 ω·R(회복→감수성) 추가."""
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

    delta = {i: [0.0, 0.0, 0.0] for i in G.nodes}
    for a, b, ed in G.edges(data=True):
        w = ed["weight"]
        if w == 0:
            continue
        da, db = G.nodes[a], G.nodes[b]
        flux_S = D_S * w * (da["S"] - db["S"]) * dt
        flux_I = D_I * w * (da["I"] - db["I"]) * dt
        flux_R = D_R * w * (da["R"] - db["R"]) * dt
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


# ── 4. weighted 시뮬레이션 (기존 run 을 weighted step 으로) ───────
def run_weighted_reaction_diffusion_simulation(G, beta, gamma, D_S, D_I, D_R,
                                               dt, days, frame_nodes=None,
                                               n_frames=150, verbose=True,
                                               step_fn=None):
    # step_fn 미지정 시 weighted SIR. SIRS 는 omega 캡처한 래퍼를 넘긴다.
    return run_reaction_diffusion_sir(
        G, beta, gamma, D_S, D_I, D_R, dt, days,
        frame_nodes=frame_nodes, n_frames=n_frames, verbose=verbose,
        step_fn=step_fn or step_weighted_reaction_diffusion_sir)


# ── 5. weight 지도 ───────────────────────────────────────────────
def plot_weighted_edges_map(gdf, G, out_path):
    """centroid 를 잇는 edge 를 weight 에 비례한 굵기/색으로 그린다."""
    segs, ws = [], []
    for a, b, d in G.edges(data=True):
        if d["weight"] <= 0:
            continue
        segs.append([(G.nodes[a]["centroid_x"], G.nodes[a]["centroid_y"]),
                     (G.nodes[b]["centroid_x"], G.nodes[b]["centroid_y"])])
        ws.append(d["weight"])
    ws = np.array(ws)

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.boundary.plot(ax=ax, color="0.85", linewidth=0.3, zorder=1)

    # 굵기: weight 에 비례(상한 clip), 색: weight
    lw = np.clip(ws, 0.2, np.percentile(ws, 99)) / max(ws.mean(), 1e-9) * 0.9
    lc = LineCollection(segs, array=ws, cmap="viridis",
                        linewidths=lw, alpha=0.8, zorder=2)
    ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, shrink=0.5)
    cb.set_label("edge weight (border length / centroid distance, mean=1)")

    ax.set_title(f"weighted diffusion edges (thicker = stronger link)  "
                 f"[n={len(ws)}, max={ws.max():.2f}]", fontsize=13)
    ax.set_axis_off(); ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장 완료: {out_path}")


# ── 6. 지표 요약(비교용) ─────────────────────────────────────────
def _metrics(hist):
    I = hist["I"]
    pk = max(range(len(I)), key=lambda i: I[i])
    return {"peak_day": hist["time"][pk], "peak_I": I[pk],
            "final_R": hist["R"][-1], "max_cnt": max(hist["infected"]),
            "cons_err": hist["max_cons_err"], "min_comp": hist["global_min_comp"]}


# ── main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="weighted graph reaction-diffusion SIRS (기본: US·numpy·weighted·sirs)")
    p.add_argument("--state", default="US", help="기본 US(미국 본토 전체). 특정 주는 예: CA")
    p.add_argument("--geoid", default="17031", help="seed GEOID(기본 17031=시카고)")
    p.add_argument("--pop", type=int, default=10000)
    p.add_argument("--real-pop", action="store_true", help="실제 census 인구 사용")
    p.add_argument("--pop-year", type=int, default=2025)
    p.add_argument("--infected", type=int, default=10)
    # ── 모델 선택 (SIR / SIRS) ──
    p.add_argument("--model", choices=["sir", "sirs"], default="sirs")
    p.add_argument("--recovery-days", type=float, default=30.0,
                   help="회복까지 평균 일수(길수록 gamma 작음=천천히 치료). 기본 30일")
    p.add_argument("--immunity-days", type=float, default=90.0,
                   help="면역 지속기간(짧을수록 R→S 많음). 기본 90일(≈3개월, 현실적)")
    p.add_argument("--omega", type=float, default=None)
    p.add_argument("--beta", type=float, default=None, help="기본 0.35")
    p.add_argument("--gamma", type=float, default=None,
                   help="없으면 sir=0.10, sirs=1/recovery_days")
    p.add_argument("--ground", choices=["prevalence", "absolute"], default="prevalence",
                   help="지상 확산 방식. prevalence(기본)=통근형, 인구 보존. "
                        "absolute=기존 절대수 라플라시안(인구가 이주함, 하위호환용)")
    p.add_argument("--D", type=float, default=None,
                   help="지상 이동 계수. prevalence 기본 0.05"
                        "(=실제 카운티간 통근 4천만명/일에 맞춘 값), absolute 기본 0.01")
    p.add_argument("--DS", type=float, default=None)
    p.add_argument("--DI", type=float, default=None)
    p.add_argument("--DR", type=float, default=None)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--days", type=int, default=None, help="기본 sir=200, sirs=365")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=True,
                   help="애니메이션 창 재생(기본 켜짐, 끄려면 --no-show)")
    p.add_argument("--fps", type=int, default=20)
    # ── 성능/캐시 옵션 ──
    p.add_argument("--backend", choices=["networkx", "numpy"], default="numpy",
                   help="weighted 시뮬레이션 백엔드(기본 numpy = 빠름)")
    p.add_argument("--compare-backends", action="store_true",
                   help="(선택) 짧은 기간으로 두 백엔드 일치 검증 + speedup")
    p.add_argument("--compare-unweighted", action="store_true",
                   help="(선택) unweighted 도 돌려서 weighted 와 비교(추가 실행이라 느림)")
    p.add_argument("--frame-interval-days", type=float, default=1.0)
    p.add_argument("--recompute-weights", action="store_true",
                   help="edge weight 캐시를 무시하고 다시 계산")
    # ── 항공 layer (T-100) ──
    p.add_argument("--flight-month", default=None,
                   help="항공 이동 layer 켜기. YYYYMM (예: 201910, 201902, 201906)")
    p.add_argument("--D-air", type=float, default=1.0,
                   help="항공 이동 배율(1.0=실제 이동량, 0=차단, 0.3=여행 70%% 감축)")
    p.add_argument("--catchment-km", type=float, default=100.0,
                   help="공항 상권 반경(km). 기본 100 (ATL 왜곡 보정에 필요)")
    p.add_argument("--stochastic-import", action="store_true",
                   help="확률적 유입: 감염자를 Poisson 정수로만 이동(0.001명 유입 제거). "
                        "결과가 매번 달라지므로 seed 를 바꿔 여러 번 돌려보세요")
    p.add_argument("--seed", type=int, default=0, help="확률적 유입 난수 시드")
    args = p.parse_args()

    # ── model 별 파라미터 해석 (reaction_diffusion_sir.py 와 동일 규칙) ──
    SMALLPOX_R0 = 5.0   # 천연두 기본 R0(문헌 3.5~7 중앙값). SIRS 기본에 사용
    model = args.model
    # 지상 이동 계수: prevalence 는 실제 통근량(약 4천만명/일)에 맞춘 0.05 가 기본
    D = args.D if args.D is not None else (0.05 if args.ground == "prevalence" else 0.01)
    days = args.days if args.days is not None else (365 if model == "sirs" else 200)
    # gamma 를 먼저 정해야 beta=R0*gamma 로 R0 를 고정할 수 있다
    if args.gamma is not None:
        gamma = args.gamma
        recovery_days = 1.0 / gamma if gamma > 0 else float("inf")
    elif model == "sirs":
        recovery_days = args.recovery_days
        gamma = 1.0 / recovery_days
    else:
        gamma = 0.10
        recovery_days = 1.0 / gamma
    # beta: SIRS 는 smallpox R0(=5) 기준 → beta=R0*gamma (gamma 바뀌어도 R0 유지). SIR 은 기존 0.35
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
    else:
        omega = 0.0
        immunity_days = float("inf")

    D_S = args.DS if args.DS is not None else D
    D_I = args.DI if args.DI is not None else D
    D_R = args.DR if args.DR is not None else D

    # SIRS 용 weighted / unweighted NetworkX step (omega 캡처)
    if model == "sirs":
        def nx_step_w(Gg, b, g, ds, di, dr, dtt):
            return step_weighted_reaction_diffusion_sirs(Gg, b, g, omega, ds, di, dr, dtt)

        def nx_step_u(Gg, b, g, ds, di, dr, dtt):
            return step_reaction_diffusion_sirs(Gg, b, g, omega, ds, di, dr, dtt)
    else:
        nx_step_w = step_weighted_reaction_diffusion_sir
        nx_step_u = None   # unweighted SIR 기본 step

    t_total0 = perf_counter()
    t0 = perf_counter()
    state = None if str(args.state).lower() in {"us", "all", "conus"} else args.state
    gdf = compute_centroids(load_counties(state=state))
    G = build_adjacency_graph(gdf)
    seeds = [g.strip() for g in args.geoid.split(",") if g.strip()]
    t_graph = perf_counter() - t0

    populations = None
    if args.real_pop:
        populations, _ = load_county_population(gdf, year=args.pop_year)

    # 유병률 기반 지상 확산은 numpy backend 전용
    if args.ground == "prevalence" and args.backend != "numpy":
        print("[ground] prevalence 방식은 numpy backend 전용 → --backend numpy 로 전환합니다.")
        args.backend = "numpy"

    # ── 항공 layer 준비 (요청 시에만; numpy backend 에서만 지원) ──
    flight = None
    if args.flight_month:
        from flight_coupling import build_flight_operator
        if args.backend != "numpy":
            print("[air] 항공 layer 는 numpy backend 전용 → --backend numpy 로 전환합니다.")
            args.backend = "numpy"
        fy, fm = int(args.flight_month[:4]), int(args.flight_month[4:])
        flight = build_flight_operator(
            gdf, fy, fm, radius_km=args.catchment_km,
            populations=(populations or {g: args.pop for g in gdf.index}),
            D_air=args.D_air, stochastic=args.stochastic_import, seed=args.seed)

    # (1) edge weight 계산(캐시 사용) + 요약 + 지도
    t0 = perf_counter()
    compute_edge_weights_border_distance(
        gdf, G, cache_path=edge_weight_cache_path(state),
        recompute=args.recompute_weights)
    t_weight = perf_counter() - t0
    print_weight_summary(G)
    plot_weighted_edges_map(gdf, G, os.path.join(OUT_DIR, "weighted_edges_map.png"))

    def fresh():
        initialize_sir_compartments(G, args.pop, seeds, args.infected,
                                    populations=populations)

    # 결과 파일명 prefix (SIRS 는 weighted_sirs)
    prefix = "weighted_sirs_reaction_diffusion" if model == "sirs" \
        else "weighted_reaction_diffusion"

    def wout(name, ext="png"):
        return os.path.join(OUT_DIR, f"{prefix}_{name}.{ext}")

    # (선택) weighted 백엔드 일치 검증
    if args.compare_backends:
        fresh()
        cmp_days = min(20, days)
        print(f"[compare] weighted model={model}, days={cmp_days} 로 두 백엔드 비교…")
        compare_networkx_vs_numpy_backend(
            G, beta, gamma, D_S, D_I, D_R, args.dt, cmp_days,
            networkx_step_fn=nx_step_w, model=model, omega=omega)

    # (2) weighted 실행 (선택 백엔드; numpy 는 edge weight 자동 사용)
    fresh()
    check_sir_consistency(G)
    want_anim = args.gif or args.show
    frame_nodes = list(gdf.index) if want_anim else None
    t0 = perf_counter()
    if args.backend == "numpy":
        histW, geoids, S, I, R = run_reaction_diffusion_simulation_numpy(
            G, beta, gamma, D_S, D_I, D_R, args.dt, days,
            frame_nodes=frame_nodes, frame_interval_days=args.frame_interval_days,
            model=model, omega=omega, flight=flight, ground=args.ground)
        numpy_state_to_graph(G, geoids, S, I, R)
    else:
        histW = run_weighted_reaction_diffusion_simulation(
            G, beta, gamma, D_S, D_I, D_R, args.dt, days,
            frame_nodes=frame_nodes, step_fn=nx_step_w)
    t_sim = perf_counter() - t0

    plot_timeseries(histW, wout("timeseries"))
    plot_infected_counties(histW, wout("infected_counties"))
    plot_final_map(gdf, G, days, wout("final_map"))
    plot_population_map(gdf, G, days, wout("population_map"))
    if want_anim:
        gif_path = wout("animation", "gif") if args.gif else None
        animate_local_sir(gdf, histW, days, out_path=gif_path, fps=args.fps,
                          show=args.show,
                          curve_title=f"Total epidemic curve ({model} weighted diffusion)")

    mW = _metrics(histW)

    # (3) (선택) unweighted 도 돌려서 비교 — 기본 OFF(추가 실행이라 느림)
    if args.compare_unweighted:
        fresh()
        histU = run_reaction_diffusion_sir(G, beta, gamma, D_S, D_I, D_R,
                                           args.dt, days, verbose=False,
                                           step_fn=nx_step_u)
        mU = _metrics(histU)
        print("\n============== unweighted vs weighted 비교 ==============")
        print(f"{'항목':<22}{'unweighted':>16}{'weighted':>16}")
        print("-" * 54)
        print(f"{'peak day':<22}{mU['peak_day']:>16.1f}{mW['peak_day']:>16.1f}")
        print(f"{'peak total infected':<22}{mU['peak_I']:>16,.0f}{mW['peak_I']:>16,.0f}")
        print(f"{'final recovered':<22}{mU['final_R']:>16,.0f}{mW['final_R']:>16,.0f}")
        print(f"{'max infected county':<22}{mU['max_cnt']:>16d}{mW['max_cnt']:>16d}")
        print(f"{'min S/I/R':<22}{mU['min_comp']:>16.1e}{mW['min_comp']:>16.1e}")
        print("=" * 54)

    # (4) 해석 (weighted 단독 — unweighted 재실행 불필요)
    strongest = max(G.edges(data=True), key=lambda e: e[2]["weight"])
    print("\n================ 결과 해석 ================")
    print(f"strongest edge: {G.nodes[strongest[0]]['name']} — "
          f"{G.nodes[strongest[1]]['name']} (w={strongest[2]['weight']:.2f}) "
          f"→ 경계 길고 가까운 이웃, 이 방향으로 확산이 강함")
    print(f"population 보존 오차: {mW['cons_err']:.3e}  (≈0 이면 보존)")
    print(f"음수 발생 여부: {'있음(불안정 → D/dt 줄이세요)' if mW['min_comp'] < -1e-9 else '없음 ✅'}")
    print("==========================================\n")

    t_total = perf_counter() - t_total0
    print("================ RUN SUMMARY ================")
    print(f"model / backend         : {model} / {args.backend} (weighted)")
    if model == "sirs":
        print(f"beta/gamma/omega        : {beta:.3f} / {gamma:.5f} / {omega:.6f} "
              f"(recovery={recovery_days:.0f}d, immunity={immunity_days:.0f}d)")
        print(f"R0 (=beta/gamma)        : {beta / gamma:.2f}  (smallpox 수준)")
    print(f"node / edge count       : {G.number_of_nodes()} / {G.number_of_edges()}")
    print(f"timestep count          : {int(round(days / args.dt))}")
    print(f"graph loading time      : {t_graph:.3f} s")
    print(f"weight compute/load time: {t_weight:.3f} s")
    print(f"simulation time         : {t_sim:.3f} s")
    print(f"total runtime           : {t_total:.3f} s")
    print(f"population 보존 오차     : {mW['cons_err']:.3e}")
    print(f"clamp 발생              : {histW.get('clamp_count', 0)}회")
    print(f"결과 폴더               : {OUT_DIR}")
    print("=============================================")
    return G, gdf, histW


if __name__ == "__main__":
    main()
