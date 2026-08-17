"""
sir_init.py
================================================================
이전 단계에서 만든 county adjacency graph(G) 위에 SIR compartment 값을 붙이고
'초기 상태'만 확인하는 스크립트.

이번 단계에서 하는 것:
  1. 기존 GeoJSON → graph 생성 코드(county_graph.py)를 그대로 재사용.
  2. 각 county node 에 N, S, I, R 값을 추가.
  3. 인구 데이터가 아직 없으므로 모든 county 를 default_population(=10000)으로.
  4. 초기 감염지(geoid)를 지정 가능 → 그 county 만 I=initial_I, 나머지는 I=0.
  5. total N/S/I/R, 감염 county 수 출력.
  6. 모든 county 에서 S+I+R == N 인지 검증.
  7. 지도에 I/N(감염 비율)으로 색을 칠해 초기 감염 상태 시각화.

아직 하지 않는 것:
  · SIR 미분방정식 업데이트 (local dynamics)
  · county 간 diffusion / 이동
  · 시간에 따른 animation

실행:
  python graph/sir_init.py --state CA --geoid 06075          # SF 한 곳 감염
  python graph/sir_init.py --state CA --geoid 06075,06037    # 여러 곳 감염
  python graph/sir_init.py                                    # CONUS, 기본 감염지
"""

import os
import argparse
import geopandas as gpd
import matplotlib.pyplot as plt

# ── 이전 단계 코드 재사용 ────────────────────────────────────────
from county_graph import (
    load_counties,
    compute_centroids,
    build_adjacency_graph,
    OUT_DIR,
)

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ── 1. SIR compartment 초기화 ────────────────────────────────────
def initialize_sir_compartments(G, default_population=10000,
                                initial_infected_geoids=None, initial_I=10,
                                populations=None):
    """
    graph 의 각 node 에 N, S, I, R 값을 붙인다.

    - N: populations(GEOID→인구) 가 주어지면 그 값, 없으면 default_population.
         populations 에 없는 county 는 default_population 으로 대체.
    - initial_infected_geoids 에 든 county: I = min(initial_I, N)
    - 나머지: I = 0
    - S = N - I,  R = 0
    실제 인구 데이터(population.load_county_population)를 populations 로 넘기면
    초기조건이 실제 인구 기반으로 바뀐다.
    """
    if initial_infected_geoids is None:
        initial_infected_geoids = []
    infected_set = set(initial_infected_geoids)

    # 지정한 감염지가 graph 에 실제로 있는지 확인(오타 방지)
    missing = infected_set - set(G.nodes)
    if missing:
        print(f"[init] 경고: graph 에 없는 geoid 무시됨: {missing}")
    infected_set &= set(G.nodes)

    n_fallback = 0
    for geoid in G.nodes:
        if populations is not None and geoid in populations:
            N = populations[geoid]
        else:
            N = default_population
            if populations is not None:
                n_fallback += 1
        I = min(initial_I, N) if geoid in infected_set else 0
        G.nodes[geoid]["N"] = N
        G.nodes[geoid]["I"] = I
        G.nodes[geoid]["S"] = N - I
        G.nodes[geoid]["R"] = 0

    if populations is not None:
        print(f"[init] {G.number_of_nodes()}개 county SIR 초기화 완료 "
              f"(실제 인구 사용, 미매칭 {n_fallback}곳은 N={default_population}, "
              f"감염지 {len(infected_set)}곳, I={initial_I})")
    else:
        print(f"[init] {G.number_of_nodes()}개 county 에 SIR 초기화 완료 "
              f"(N={default_population}, 감염지 {len(infected_set)}곳, I={initial_I})")
    if infected_set:
        for g in sorted(infected_set):
            print(f"       감염 seed: {g} ({G.nodes[g]['name']})  I={G.nodes[g]['I']}")
    return G


# ── 2. 보존 조건 검증: S + I + R == N ────────────────────────────
def check_sir_consistency(G):
    """모든 county 에서 S+I+R == N 인지 확인. 어긋나면 목록 출력."""
    bad = []
    for geoid in G.nodes:
        d = G.nodes[geoid]
        if d["S"] + d["I"] + d["R"] != d["N"]:
            bad.append(geoid)

    if bad:
        print(f"[check] ❌ S+I+R != N 인 county {len(bad)}개: {bad[:10]}")
        return False
    print(f"[check] ✅ 모든 {G.number_of_nodes()}개 county 에서 S+I+R == N 성립")
    return True


# ── 3. 초기 상태 요약 출력 ───────────────────────────────────────
def print_sir_summary(G):
    """total N/S/I/R 와 감염 county 개수를 출력."""
    tot_N = tot_S = tot_I = tot_R = 0
    infected_counties = 0
    for geoid in G.nodes:
        d = G.nodes[geoid]
        tot_N += d["N"]; tot_S += d["S"]; tot_I += d["I"]; tot_R += d["R"]
        if d["I"] > 0:
            infected_counties += 1

    print("\n================ SIR INITIAL SUMMARY ================")
    print(f"total population (N) : {tot_N:,}")
    print(f"total S              : {tot_S:,}")
    print(f"total I              : {tot_I:,}")
    print(f"total R              : {tot_R:,}")
    print(f"감염자 있는 county 수 : {infected_counties} / {G.number_of_nodes()}")
    print(f"S+I+R 합계           : {tot_S + tot_I + tot_R:,}  (== total N 이어야 함)")
    print("====================================================\n")


# ── 4. 초기 감염 지도 시각화 ─────────────────────────────────────
def plot_initial_infection_map(gdf, G, out_path):
    """
    county 를 I/N(감염 비율)으로 색칠해 초기 감염 상태를 지도로 보여준다.
    - 감염 county 는 진한 빨강, 나머지는 흰색에 가깝게.
    - 감염지에는 별도 마커도 찍어 눈에 잘 띄게.
    """
    # graph 의 I/N 값을 gdf 에 매핑(gdf 는 index=GEOID)
    gdf = gdf.copy()
    gdf["I"] = [G.nodes[g]["I"] for g in gdf.index]
    gdf["N"] = [G.nodes[g]["N"] for g in gdf.index]
    gdf["inf_ratio"] = gdf["I"] / gdf["N"]

    fig, ax = plt.subplots(figsize=(14, 10))

    # (a) 감염 비율로 채색 (0 인 곳은 밝게, 감염지는 진하게)
    gdf.plot(
        column="inf_ratio", ax=ax, cmap="Reds",
        vmin=0.0, vmax=max(gdf["inf_ratio"].max(), 1e-9),
        edgecolor="0.7", linewidth=0.2,
        legend=True,
        legend_kwds={"label": "Infection ratio I / N", "shrink": 0.5},
    )

    # (b) 감염지 centroid 에 마커 표시(작아도 보이도록)
    infected = gdf[gdf["I"] > 0]
    if len(infected):
        cx = [G.nodes[g]["x"] for g in infected.index]
        cy = [G.nodes[g]["y"] for g in infected.index]
        ax.scatter(cx, cy, s=60, facecolor="none",
                   edgecolor="blue", linewidth=1.5, zorder=5,
                   label=f"Infection seed ({len(infected)} counties)")
        ax.legend(loc="lower left")

    ax.set_title(
        f"Initial infection state  ({len(infected)} infected counties, "
        f"total I={int(gdf['I'].sum())})", fontsize=14)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장 완료: {out_path}")


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="county graph 위 SIR 초기화 확인")
    parser.add_argument("--state", default=None,
                        help="특정 주만(예: CA). 생략 시 미국 본토 전체")
    parser.add_argument("--geoid", default=None,
                        help="초기 감염지 GEOID(쉼표로 여러 개). 생략 시 자동 선택")
    parser.add_argument("--pop", type=int, default=10000, help="임시 인구(기본 10000)")
    parser.add_argument("--infected", type=int, default=10, help="감염지 초기 I(기본 10)")
    args = parser.parse_args()

    # 1) 기존 코드로 graph 생성
    gdf = load_counties(state=args.state)
    gdf = compute_centroids(gdf)
    G = build_adjacency_graph(gdf)

    # 2) 초기 감염지 결정
    if args.geoid:
        seeds = [g.strip() for g in args.geoid.split(",") if g.strip()]
    else:
        # 지정 없으면: SF(06075)가 있으면 그걸, 아니면 첫 node 를 seed 로
        seeds = ["06075"] if "06075" in G.nodes else [list(G.nodes)[0]]
        print(f"[main] 감염지 미지정 → 기본 seed 사용: {seeds}")

    # 3) SIR 초기화 + 검증 + 요약
    initialize_sir_compartments(
        G, default_population=args.pop,
        initial_infected_geoids=seeds, initial_I=args.infected)
    check_sir_consistency(G)
    print_sir_summary(G)

    # 4) 시각화
    tag = (args.state or "conus").lower()
    out_path = os.path.join(OUT_DIR, f"sir_init_{tag}.png")
    plot_initial_infection_map(gdf, G, out_path)

    print("다음 단계 힌트: 이 G 에 시간 루프를 돌려 각 node 안에서 "
          "S,I,R 를 SIR 미분방정식으로 갱신하면 local dynamics 가 된다.")
    return G, gdf


if __name__ == "__main__":
    main()
