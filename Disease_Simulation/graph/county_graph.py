"""
county_graph.py
================================================================
Counties GeoJSON → county 인접(adjacency) graph 의 '기본 구조' 확인 스크립트.

이번 단계의 목표(시뮬레이션은 아직 X):
  1. counties GeoJSON 을 불러온다.
  2. 각 county 를 node 하나로 본다.
  3. 각 county 의 중심점(centroid)을 계산한다.
  4. 경계를 공유(border touch)하는 county 끼리 graph edge 로 잇는다.
  5. graph 기본 정보(노드/엣지 수·평균 연결·고립 노드)를 출력한다.
  6. 지도 위에 centroid 점 + 연결선을 그려 눈으로 검증한다.

아직 SIR / PDE / diffusion / infection 은 구현하지 않는다.
나중에 각 node 에 S,I,R 값을 붙이면 metapopulation 모델로 확장 가능하도록
전부 '함수 단위' 로 정리해 두었다.

실행:
  python graph/county_graph.py                 # 미국 본토(CONUS) 전체
  python graph/county_graph.py --state 06      # 캘리포니아만(STATEFP=06)
  python graph/county_graph.py --state CA       # 주 약어로도 가능
"""

import os
import argparse
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 경로 ─────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # Disease_Simulation/
DATA_PATH = os.path.join(ROOT, "data", "counties.geojson")
OUT_DIR = os.path.join(ROOT, "result", "county_graph")

# 미국 본토(CONUS)만 남기기 위해 제외할 STATEFP(알래스카·하와이·해외영토).
# 떨어져 있는 섬 주는 '경계 공유'가 없어 graph 검증을 방해하므로 기본 제외.
NON_CONUS_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

# 편의를 위한 주 약어 → STATEFP 매핑(자주 쓰는 것만; 숫자 코드도 그대로 허용)
STATE_ABBR = {
    "CA": "06", "NY": "36", "TX": "48", "FL": "12", "WA": "53",
    "OR": "41", "IL": "17", "PA": "42", "OH": "39", "GA": "13",
    "NC": "37", "MI": "26", "AZ": "04", "CO": "08", "MA": "25",
}


# ── 1. 데이터 로드 ────────────────────────────────────────────────
def load_counties(path=DATA_PATH, state=None, drop_non_conus=True):
    """
    counties GeoJSON 을 GeoDataFrame 으로 읽는다.

    - 정확한 centroid / 인접 계산을 위해 EPSG:5070(US Albers 등적투영)으로 재투영.
    - state 인자가 있으면 해당 주만 필터(그림을 보기 쉽게).
    - state 가 없으면 기본적으로 미국 본토(CONUS)만 남긴다.
    반환: GeoDataFrame (index = GEOID)
    """
    print(f"[load] reading: {path}")
    gdf = gpd.read_file(path)
    print(f"[load] raw feature 개수: {len(gdf)}")
    print(f"[load] 원본 CRS: {gdf.crs}")

    if state is not None:
        fips = STATE_ABBR.get(str(state).upper(), str(state))
        gdf = gdf[gdf["STATEFP"] == fips].copy()
        print(f"[load] state={state} (STATEFP={fips}) 필터 후: {len(gdf)} counties")
    elif drop_non_conus:
        gdf = gdf[~gdf["STATEFP"].isin(NON_CONUS_FIPS)].copy()
        print(f"[load] CONUS(본토)만 남김: {len(gdf)} counties")

    # 등적투영으로 재투영 → centroid / 거리 계산이 지리적으로 의미 있음
    gdf = gdf.to_crs(epsg=5070)
    print(f"[load] 재투영 CRS: {gdf.crs}")

    # GEOID 를 node id 로 사용
    gdf = gdf.set_index("GEOID")
    return gdf


# ── 2. centroid 계산 ─────────────────────────────────────────────
def compute_centroids(gdf):
    """각 county 의 중심점을 계산해 x, y 컬럼으로 붙인다(투영 좌표계 기준)."""
    cent = gdf.geometry.centroid
    gdf = gdf.copy()
    gdf["cx"] = cent.x
    gdf["cy"] = cent.y
    print(f"[centroid] {len(gdf)}개 county 의 중심점 계산 완료")
    print(f"[centroid] 예시: {gdf.index[0]} ({gdf['NAME'].iloc[0]}) "
          f"-> x={gdf['cx'].iloc[0]:.0f}, y={gdf['cy'].iloc[0]:.0f}")
    return gdf


# ── 3. 인접 graph 생성 ───────────────────────────────────────────
def build_adjacency_graph(gdf):
    """
    경계를 공유하는 county 끼리 edge 로 잇는 무방향 graph 를 만든다.

    - node : GEOID (속성: name, statefp, x, y)
    - edge : 두 county geometry 가 'touches'(경계 접촉) 관계이면 연결
             (꼭짓점 하나만 닿는 경우까지 포함하는 queen 방식 인접)
    - 공간 인덱스를 쓰는 sjoin 으로 빠르게 이웃 쌍을 찾는다.
    """
    G = nx.Graph()

    # (a) 모든 county 를 먼저 node 로 추가 → 고립 county 도 빠짐없이 포함
    for geoid, row in gdf.iterrows():
        G.add_node(
            geoid,
            name=row["NAME"],
            statefp=row["STATEFP"],
            x=row["cx"],
            y=row["cy"],
        )

    # (b) self-spatial-join 으로 경계 접촉 쌍 찾기.
    #     index 이름이 GEOID 라서 오른쪽 이웃 id 는 'GEOID_right' 컬럼에 담긴다
    #     (index 가 무명이면 'index_right'). 두 경우 모두 대응.
    joined = gpd.sjoin(gdf, gdf, predicate="touches", how="inner")
    right_col = "GEOID_right" if "GEOID_right" in joined.columns else "index_right"
    left = joined.index                       # GEOID (왼쪽)
    right = joined[right_col]                 # GEOID (오른쪽 이웃)

    # (a,b)와 (b,a) 중복 제거
    edges = {tuple(sorted((a, b))) for a, b in zip(left, right) if a != b}
    G.add_edges_from(edges)

    print(f"[graph] node 추가: {G.number_of_nodes()}")
    print(f"[graph] edge 추가: {G.number_of_edges()}")
    return G


# ── 4. graph 기본 정보 출력 ──────────────────────────────────────
def print_graph_summary(G):
    """county 개수·edge 개수·평균 연결·고립 노드 등 기본 통계를 출력."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_deg = (2 * m / n) if n else 0.0
    isolated = list(nx.isolates(G))
    components = nx.number_connected_components(G)

    print("\n================ GRAPH SUMMARY ================")
    print(f"county(node) 개수      : {n}")
    print(f"edge 개수              : {m}")
    print(f"평균 연결 개수(degree) : {avg_deg:.2f}")
    print(f"최소 / 최대 degree     : {min(degrees)} / {max(degrees)}")
    print(f"연결 요소(component) 수 : {components}")
    print(f"고립된 county 개수     : {len(isolated)}")
    if isolated:
        names = [G.nodes[i].get("name", "?") for i in isolated[:10]]
        print(f"  고립 예시(최대 10):  {names}")
    else:
        print("  → 고립된 county 없음 (모두 이웃과 연결됨)")

    # 가장 많이 연결된 county 몇 개(허브) 확인 → graph 가 그럴듯한지 감 잡기
    top = sorted(G.degree(), key=lambda kv: kv[1], reverse=True)[:5]
    print("연결 많은 county Top5   :")
    for geoid, deg in top:
        print(f"  {geoid} {G.nodes[geoid]['name']:<20s} degree={deg}")
    print("===============================================\n")


# ── 5. 시각화 ────────────────────────────────────────────────────
def plot_graph(G, gdf, out_path):
    """county 경계 + centroid 점 + 인접 연결선을 함께 그려 저장."""
    n = G.number_of_nodes()
    fig, ax = plt.subplots(figsize=(14, 10))

    # (a) county 경계(옅은 회색)
    gdf.boundary.plot(ax=ax, color="0.8", linewidth=0.3, zorder=1)

    # (b) 인접 연결선(파란색) — node 좌표를 이어서 그림
    node_xy = {i: (G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes}
    # 노드 수가 많으면 선을 얇게(전국) / 적으면 굵게(단일 주)
    edge_w = 0.25 if n > 500 else 0.8
    pt_s = 3 if n > 500 else 20
    for a, b in G.edges():
        xa, ya = node_xy[a]
        xb, yb = node_xy[b]
        ax.plot([xa, xb], [ya, yb], color="steelblue",
                linewidth=edge_w, alpha=0.5, zorder=2)

    # (c) centroid 점(빨강)
    xs = [xy[0] for xy in node_xy.values()]
    ys = [xy[1] for xy in node_xy.values()]
    ax.scatter(xs, ys, s=pt_s, color="crimson", zorder=3)

    ax.set_title(f"County adjacency graph  (nodes={n}, edges={G.number_of_edges()})",
                 fontsize=14)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 결과 폴더 규칙: 같은 결과물의 이전 png 는 덮어쓴다(같은 파일명이라 savefig 가 대체).
    # state 별로 파일명이 다르므로 CA/CONUS 결과는 함께 보존된다.
    if os.path.exists(out_path):
        os.remove(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] 저장 완료: {out_path}")


# ── main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="County adjacency graph 기본 구조 확인")
    parser.add_argument("--state", default=None,
                        help="특정 주만(예: CA 또는 06). 생략 시 미국 본토 전체")
    args = parser.parse_args()

    gdf = load_counties(state=args.state)
    gdf = compute_centroids(gdf)
    G = build_adjacency_graph(gdf)
    print_graph_summary(G)

    tag = (args.state or "conus").lower()
    out_path = os.path.join(OUT_DIR, f"county_graph_{tag}.png")
    plot_graph(G, gdf, out_path)

    print("다음 단계 힌트: 이 G 의 각 node 에 S,I,R 값을 붙이면 metapopulation 모델이 된다.")
    return G, gdf


if __name__ == "__main__":
    main()
