"""
ensemble.py
================================================================
확률적 유입(stochastic importation) 모델은 매 실행마다 결과가 달라진다.
단일 실행은 '하나의 가능한 시나리오'일 뿐이므로, N회 돌려서 '분포'로 봐야 한다.

이 스크립트가 하는 일:
  1) 같은 파라미터로 seed 만 바꿔 N회 시뮬레이션 (그래프·weight·항공 데이터는 1회만 로드)
  2) 각 실행에서 기록:
       · 주요 도시(county)별 '첫 감염자 1명' 도달일  ← 확률적 유입의 핵심 지표
       · 전체 유행 곡선 I(t), 감염 county 수
       · peak day / peak I / final R
  3) 산출:
       · result/ensemble/ensemble_arrival_times.png   도착일 분포(박스플롯)
       · result/ensemble/ensemble_epidemic_curves.png 유행곡선 중앙값 + 90% 밴드
       · result/ensemble/ensemble_runs.csv            실행별 지표
       · 콘솔: 도착일 중앙값과 90% 구간

모델 자체는 건드리지 않는다. 기존 step 함수와 항공 flux 를 그대로 호출한다.

실행:
  python graph/ensemble.py --runs 20 --geoid 36081 --days 200
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from county_graph import load_counties, compute_centroids, build_adjacency_graph
from weighted_reaction_diffusion import (
    compute_edge_weights_border_distance, edge_weight_cache_path,
)
from sir_init import initialize_sir_compartments
from population import load_county_population
from numpy_backend import (
    graph_to_numpy_state, step_reaction_diffusion_sir_numpy,
    step_reaction_diffusion_sirs_numpy,
)
from flight_coupling import build_flight_operator, apply_flight_flux

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "result", "ensemble")

# 기본 관찰 도시(주요 대도시 county)
DEFAULT_WATCH = {
    "06037": "Los Angeles", "17031": "Chicago(Cook)", "48113": "Dallas",
    "12086": "Miami-Dade", "06075": "San Francisco", "53033": "Seattle(King)",
    "04013": "Phoenix(Maricopa)", "08031": "Denver", "48201": "Houston(Harris)",
}


def run_one(G, gdf, pop, flight, seed, args, watch_idx):
    """seed 하나로 1회 시뮬레이션. 반환: (hist_I, hist_cnt, arrivals)"""
    initialize_sir_compartments(G, args.pop, [args.geoid], args.infected,
                                populations=pop)
    geoids, idx, S, I, R, eu, ev, ew = graph_to_numpy_state(G)
    if flight is not None:
        flight["rng"] = np.random.default_rng(seed)   # 실행마다 새 난수열

    gamma = 1.0 / args.recovery_days
    omega = (1.0 / args.immunity_days) if args.model == "sirs" else 0.0
    n_steps = int(round(args.days / args.dt))

    arrivals = {g: None for g in watch_idx}
    tI, tcnt = [], []
    for k in range(1, n_steps + 1):
        if args.model == "sirs":
            S, I, R, _, _ = step_reaction_diffusion_sirs_numpy(
                S, I, R, eu, ev, ew, args.beta, gamma, omega,
                args.D, args.D, args.D, args.dt)
        else:
            S, I, R, _, _ = step_reaction_diffusion_sir_numpy(
                S, I, R, eu, ev, ew, args.beta, gamma,
                args.D, args.D, args.D, args.dt)
        if flight is not None:
            S, I, R = apply_flight_flux(S, I, R, S + I + R, flight, args.dt)

        t = k * args.dt
        tI.append(float(I.sum()))
        tcnt.append(int((I > 0.5).sum()))
        for g, i in watch_idx.items():
            if arrivals[g] is None and I[i] >= 1.0:   # '실제 감염자 1명' 도달
                arrivals[g] = t
    return np.array(tI), np.array(tcnt), arrivals


def main():
    p = argparse.ArgumentParser(description="확률적 유입 앙상블 실행")
    p.add_argument("--state", default="US")
    p.add_argument("--geoid", default="36081", help="seed county (기본 Queens/NYC)")
    p.add_argument("--runs", type=int, default=20, help="앙상블 실행 횟수")
    p.add_argument("--days", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--model", choices=["sir", "sirs"], default="sirs")
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--recovery-days", type=float, default=30.0)
    p.add_argument("--immunity-days", type=float, default=90.0)
    p.add_argument("--D", type=float, default=0.01)
    p.add_argument("--pop", type=int, default=10000)
    p.add_argument("--infected", type=int, default=10)
    p.add_argument("--flight-month", default="201910")
    p.add_argument("--D-air", type=float, default=1.0)
    p.add_argument("--catchment-km", type=float, default=100.0)
    p.add_argument("--deterministic", action="store_true",
                   help="비교용: 확률적 유입 끄고 1회만 실행")
    args = p.parse_args()

    state = None if args.state.lower() in {"us", "all", "conus"} else args.state
    print("[ens] 그래프·weight·항공 데이터 로딩(1회)…")
    gdf = compute_centroids(load_counties(state=state))
    G = build_adjacency_graph(gdf)
    compute_edge_weights_border_distance(gdf, G, cache_path=edge_weight_cache_path(state))
    pop, _ = load_county_population(gdf, verbose=False)

    flight = None
    if args.flight_month:
        fy, fm = int(args.flight_month[:4]), int(args.flight_month[4:])
        flight = build_flight_operator(
            gdf, fy, fm, radius_km=args.catchment_km, populations=pop,
            D_air=args.D_air, stochastic=True, seed=0, verbose=True)

    idx_all = {g: i for i, g in enumerate(gdf.index)}
    watch = {g: n for g, n in DEFAULT_WATCH.items()
             if g in idx_all and g != args.geoid}
    watch_idx = {g: idx_all[g] for g in watch}
    print(f"[ens] seed={args.geoid}, 관찰 도시 {len(watch)}개, {args.runs}회 실행\n")

    curves, counts, rows, arr_rows = [], [], [], []
    for r in range(args.runs):
        tI, tcnt, arrivals = run_one(G, gdf, pop, flight, r, args, watch_idx)
        curves.append(tI); counts.append(tcnt)
        pk = int(np.argmax(tI))
        rows.append({"run": r, "seed": r,
                     "peak_day": (pk + 1) * args.dt, "peak_I": tI[pk],
                     "max_infected_county": int(tcnt.max())})
        arr_rows.append({"run": r, **{watch[g]: arrivals[g] for g in watch}})
        print(f"  run {r:2d}: peak {(pk+1)*args.dt:6.1f}일, peak I {tI[pk]:>12,.0f}, "
              f"감염 county {tcnt.max():4d}")

    os.makedirs(OUT_DIR, exist_ok=True)
    runs_df = pd.DataFrame(rows)
    arr_df = pd.DataFrame(arr_rows)
    runs_df.to_csv(os.path.join(OUT_DIR, "ensemble_runs.csv"), index=False)
    arr_df.to_csv(os.path.join(OUT_DIR, "ensemble_arrivals.csv"), index=False)

    # ── 도착일 분포 요약 ──
    print("\n================ 첫 감염자(1명) 도달일 분포 ================")
    print(f"{'도시':<20}{'중앙값':>8}{'90% 구간':>18}{'미도달':>8}")
    print("-" * 56)
    order = []
    for g, nm in watch.items():
        v = arr_df[nm].dropna().to_numpy()
        if len(v) == 0:
            print(f"{nm:<20}{'-':>8}{'-':>18}{args.runs:>8}")
            continue
        lo, hi = np.percentile(v, [5, 95])
        med = np.median(v)
        order.append((med, nm))
        print(f"{nm:<20}{med:>8.1f}{f'{lo:.0f} ~ {hi:.0f}일':>18}"
              f"{args.runs - len(v):>8}")
    print("=" * 56)

    # ── 그림 1: 도착일 박스플롯 ──
    order.sort()
    names = [nm for _, nm in order]
    data = [arr_df[nm].dropna().to_numpy() for nm in names]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(data, vert=False, tick_labels=names, showfliers=True)
    ax.set_xlabel("Day of first infection arrival (day)")
    ax.set_title(f"Stochastic importation — arrival day distribution by city ({args.runs}-run ensemble, "
                 f"seed county={args.geoid})")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    f1 = os.path.join(OUT_DIR, "ensemble_arrival_times.png")
    fig.savefig(f1, dpi=150); plt.close(fig)
    print(f"[ens] 저장: {f1}")

    # ── 그림 2: 유행 곡선 중앙값 + 90% 밴드 ──
    C = np.vstack(curves)
    t = np.arange(1, C.shape[1] + 1) * args.dt
    med = np.median(C, axis=0)
    lo, hi = np.percentile(C, [5, 95], axis=0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for c in C:
        a1.plot(t, c, color="crimson", alpha=0.15, lw=0.8)
    a1.plot(t, med, color="crimson", lw=2.5, label="Median")
    a1.fill_between(t, lo, hi, color="crimson", alpha=0.2, label="90% interval")
    a1.set_ylabel("Total infected I"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title(f"Stochastic importation ensemble ({args.runs} runs) — epidemic curve uncertainty")

    K = np.vstack(counts)
    a2.plot(t, np.median(K, axis=0), color="darkorange", lw=2.5, label="Median")
    a2.fill_between(t, *np.percentile(K, [5, 95], axis=0),
                    color="darkorange", alpha=0.2, label="90% interval")
    a2.set_xlabel("time (days)"); a2.set_ylabel("Infected counties")
    a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(OUT_DIR, "ensemble_epidemic_curves.png")
    fig.savefig(f2, dpi=150); plt.close(fig)
    print(f"[ens] 저장: {f2}")

    print(f"\npeak day: 중앙값 {runs_df.peak_day.median():.1f}일 "
          f"(범위 {runs_df.peak_day.min():.1f} ~ {runs_df.peak_day.max():.1f})")
    print(f"결과 폴더: {OUT_DIR}")


if __name__ == "__main__":
    main()
