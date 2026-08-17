# ================================================================
#  설치 (터미널에서 먼저 실행):
#
#    pip install mesa mesa-geo networkx geopandas shapely matplotlib
#
# ================================================================
"""
nyc_network_metapop_sir.py
================================================================
네트워크 기반 메타포퓰레이션(metapopulation) ABM SIR 모델.

이전의 '좌표 평면 위 무작위 걷기'와 완전히 다른 접근이다.
  · 공간을 연속 좌표가 아니라 '동네(노드) + 길(에지)'의 그래프로 본다.
  · 사람은 (위도,경도) 실수 좌표를 갖지 않고, '어느 동네에 있는가'(current_node)
    만 갖는다. 이동은 오직 '길로 연결된 이웃 동네'로만 가능하다.
  · 감염은 같은 동네 안에 모인 사람들 사이에서 국소적으로 일어난다.

이는 전염병학에서 도시 간/구역 간 확산을 다루는 표준 틀인
'메타포퓰레이션 모델'이다. (각 동네 = 하나의 하위 인구집단 patch)

데이터:
  NYC 2020 Neighborhood Tabulation Areas(NTA) GeoJSON(폴더에서 자동 탐색).
  이 파일에는 인구가 없으므로, 실제 2020 센서스 '자치구별 인구'를
  각 자치구의 주거지 NTA에 '면적 비례'로 분배해 동네별 인구를 만든다.
  (실제 NTA별 인구 CSV가 있으면 assign_population() 한 곳만 바꾸면 됨)

실행:
  python3 nyc_network_metapop_sir.py
  → result/nyc_network_metapop_sir.gif   (동네 감염비율 코로플레스 + SIR 그래프)
"""

import os
import glob
from collections import defaultdict

import numpy as np
import geopandas as gpd
import networkx as nx

import mesa
import mesa_geo as mg

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ==================================================================
# 조절 파라미터
# ==================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

TARGET_CRS = "EPSG:32618"        # 미터 단위 투영(면적·인접 버퍼 계산용)

# 실제 2020 US 센서스 자치구별 인구(고정 공표값)
BOROUGH_POP_2020 = {
    "1": 1_694_251,   # Manhattan
    "2": 1_472_654,   # Bronx
    "3": 2_736_074,   # Brooklyn
    "4": 2_405_464,   # Queens
    "5":   495_747,   # Staten Island
}                     # 합계 8,804,190

POP_PER_AGENT = 1000             # 에이전트 1명 = 실제 1,000명 (스케일)
ADJ_BUFFER_M = 120.0             # 이 거리 안이면 '길로 연결'로 간주(좁은 공원/도로 다리 역할)

# --- 시간 축: 하루 = 낮(Day) + 밤(Night) 두 스텝이 번갈아 ---
STEPS_PER_DAY = 2                # 1일 = 낮 스텝 + 밤 스텝
COMMUTE_PROB = 0.80             # '낮'에 직장(work_node)으로 출근할 확률
                                 #   (나머지 20%는 집에 머무름: 재택/휴무/실직 등)

# --- 출퇴근/직장 배정 ---
WORK_CBD_PROB = 0.65            # 직장이 맨해튼 도심(CBD)에 있을 확률(요구: ≥60% 집중)
WORK_KEYWORDS = [               # 이 키워드가 이름에 들어간 맨해튼 동네 = 도심 직장지구
    "Midtown", "Financial District", "Hudson Yards", "Flatiron",
    "Hell's Kitchen", "Murray Hill", "Tribeca", "SoHo",
]

# 밀도 의존 감염률(force of infection). 같은 동네의 감염자가 많을수록 위험↑
#   P(감염) = 1 - exp(-BETA * I / N)   (I: 그 동네 감염자 수, N: 그 동네 전체 인원)
#   → 낮·밤 각 스텝이 끝날 때, 그 순간 각 노드에 모인 사람들을 기준으로 매번 계산
# 주의: 출퇴근 도입으로 도심(CBD)에 수백~수천 명이 매일 모여 초고밀도로 섞인다.
# 예전 국소 랜덤워크 기준의 BETA(2.5)를 그대로 쓰면 며칠 만에 전 도시가 감염된다.
# 그래서 출퇴근 허브 혼합에 맞춰 전파강도를 크게 낮춘다(수 주 규모 유행).
BETA = 0.10                      # 전파 강도(클수록 빠르게 퍼짐)
RECOVERY_DAYS = 14               # 감염 14일 뒤 감염 종료(회복 또는 사망 판정)
RECOVERY_STEPS = RECOVERY_DAYS * STEPS_PER_DAY   # = 28 스텝(낮·밤 포함)
FATALITY_RATE = 0.02             # 치사율: 감염 종료 시 2% 사망(D), 98% 회복(R)
INITIAL_INFECTED_NODES = 3       # 초기 감염 동네 수
INITIAL_INFECTED_PER_NODE = 8    # 각 초기 감염 동네에서 처음 감염되는 에이전트 수

MAX_DAYS = 300                   # 최대 시뮬레이션 일수(자가격리로 유행이 길어짐)
MAX_STEPS = MAX_DAYS * STEPS_PER_DAY
FRAME_STRIDE = 4                 # 애니메이션은 4스텝(2일)마다 1프레임(GIF 용량↓, 동일 시간대 샘플)
RANDOM_SEED = 42

CMAP = "Reds"                    # 감염비율 0→흰색, 1→진한 빨강


# ==================================================================
# 1. 데이터 로딩 · 인구 배정 · 네트워크 생성
# ==================================================================
def load_geojson(folder=HERE):
    """폴더에서 .geojson 자동 탐색 후 미터 투영으로 로드."""
    matches = sorted(glob.glob(os.path.join(folder, "*.geojson")))
    if not matches:
        raise FileNotFoundError(f"{folder} 에 .geojson 이 없습니다.")
    print(f"GeoJSON 로드: {os.path.basename(matches[0])}")
    gdf = gpd.read_file(matches[0])
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    return gdf.to_crs(TARGET_CRS)


def assign_population(gdf):
    """주거지(ntatype==0) NTA 에만, 자치구 실제 인구를 면적 비례로 분배.

    비주거지(공항·공원·묘지·교도소 등)는 인구 0 → 모델에서 제외.
    ※ 실제 NTA별 인구 CSV 가 있으면 이 함수만 교체하면 된다."""
    res = gdf[gdf["ntatype"] == "0"].copy().reset_index(drop=True)
    res["shape_area"] = res["shape_area"].astype(float)
    res["population"] = 0.0
    for code, pop in BOROUGH_POP_2020.items():
        m = res["borocode"] == code
        area = res.loc[m, "shape_area"]
        res.loc[m, "population"] = pop * area / area.sum()
    res["population"] = res["population"].round().astype(int)
    print(f"  주거지 NTA: {len(res)}개  |  배정 총인구: {res['population'].sum():,}명")
    return res


def build_network(res):
    """국경선이 맞닿은(또는 ADJ_BUFFER_M 이내로 근접한) 주거지끼리 에지로 연결.

    좁은 공원·도로로 갈라진 이웃도 이어지도록 버퍼를 살짝 준다.
    스테이튼 아일랜드처럼 물로 완전히 떨어진 곳은 자연히 별도 컴포넌트가 된다."""
    G = nx.Graph()
    G.add_nodes_from(res["nta2020"])

    # 버퍼링한 폴리곤끼리 교차하면 인접으로 판단(공간 인덱스로 빠르게)
    buffered = res[["nta2020", "geometry"]].copy()
    buffered["geometry"] = res.geometry.buffer(ADJ_BUFFER_M)
    joined = gpd.sjoin(buffered, res[["nta2020", "geometry"]],
                       predicate="intersects")
    for a, b in zip(joined["nta2020_left"], joined["nta2020_right"]):
        if a != b:
            G.add_edge(a, b)

    comps = list(nx.connected_components(G))
    print(f"  네트워크: 노드 {G.number_of_nodes()}개, 에지 {G.number_of_edges()}개, "
          f"연결요소 {len(comps)}개 (최대 요소 {max(len(c) for c in comps)}개 동네)")
    return G


# ==================================================================
# 2. 에이전트
# ==================================================================
class NeighborhoodAgent(mg.GeoAgent):
    """지도 위 한 동네(구역). 그래프의 노드이자 감염이 일어나는 patch.

    감염비율(ratio = I/전체)을 들고 있어 코로플레스 색을 결정한다."""

    def __init__(self, model, geometry, crs, node_id, population):
        super().__init__(model, geometry, crs)
        self.node_id = node_id
        self.population = population
        self.num_total = 0
        self.num_infected = 0

    @property
    def ratio(self):
        return self.num_infected / self.num_total if self.num_total else 0.0


class PersonAgent(mesa.Agent):
    """사람 에이전트. 실수 좌표 없이 '동네 노드'로만 위치를 표현.

    속성:
      unique_id     : 자동 부여
      condition     : 'S'/'I'/'R'
      home_node     : 집 — 밤에 돌아오는 곳(인구 비율대로 흩어져 배치)
      work_node     : 직장 — 낮에 가는 곳(60%+ 가 맨해튼 도심 CBD)
      current_node  : 지금 있는 동네(낮=직장/밤=집)
    """

    def __init__(self, model, home_node):
        super().__init__(model)            # unique_id 자동 부여(Mesa 3.x)
        self.condition = "S"
        self.home_node = home_node
        self.work_node = home_node         # 나중에 assign_work() 에서 배정
        self.current_node = home_node
        self.infected_at = None

    def infect(self, at_step):
        self.condition = "I"
        self.infected_at = at_step


# ==================================================================
# 3. 모델
# ==================================================================
class NYCNetworkSIRModel(mesa.Model):
    def __init__(self, seed=RANDOM_SEED):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.t = 0

        # --- 지도 · 인구 · 네트워크 ---
        gdf = load_geojson()
        self.res = assign_population(gdf)            # 주거지 GeoDataFrame(코로플레스 순서 기준)
        self.res_ids = list(self.res["nta2020"])     # 노드 순서 고정
        self.G = build_network(self.res)
        self.adj = {n: list(self.G.neighbors(n)) for n in self.res_ids}  # 빠른 이웃조회
        self.bounds = self.res.total_bounds

        # --- mesa-geo GeoSpace + NeighborhoodAgent ---
        self.space = mg.GeoSpace(crs=str(self.res.crs), warn_crs_conversion=False)
        self.nbhd = {}                               # node_id -> NeighborhoodAgent
        nbhd_agents = []
        for _, row in self.res.iterrows():
            na = NeighborhoodAgent(self, row.geometry, self.res.crs,
                                   row["nta2020"], int(row["population"]))
            self.nbhd[row["nta2020"]] = na
            nbhd_agents.append(na)
        self.space.add_agents(nbhd_agents)

        # --- 사람 에이전트 생성(집=home_node, 인구 비율대로 흩어짐) ---
        self.persons = self.create_people()

        # --- 직장(work_node) 배정: 60%+ 를 맨해튼 도심(CBD)으로 집중 ---
        self.cbd_nodes = self.identify_cbd()
        self.assign_work()

        # --- 초기 감염 ---
        self.seed_infection()

        # --- 기록 ---
        self.history = {"S": [], "I": [], "R": [], "D": [], "label": []}
        self.update_neighborhood_counts()
        self.record(label="Initial (Home)")

    # ----- 사람 생성 -----
    def create_people(self):
        persons = []
        for node_id in self.res_ids:
            pop = self.nbhd[node_id].population
            n_agents = max(1, round(pop / POP_PER_AGENT)) if pop > 0 else 0
            for _ in range(n_agents):
                persons.append(PersonAgent(self, node_id))
        print(f"  사람 에이전트: {len(persons):,}명 "
              f"(1명 = {POP_PER_AGENT:,}명 스케일)")
        return persons

    # ----- 직장(CBD) 식별 & 배정 -----
    def identify_cbd(self):
        """이름에 WORK_KEYWORDS 가 들어간 맨해튼(borocode=1) 동네 = 도심 직장지구."""
        name_of = dict(zip(self.res["nta2020"], self.res["ntaname"]))
        boro_of = dict(zip(self.res["nta2020"], self.res["borocode"]))
        cbd = [n for n in self.res_ids
               if boro_of[n] == "1"
               and any(kw in name_of[n] for kw in WORK_KEYWORDS)]
        print(f"  도심(CBD) 직장지구 {len(cbd)}곳: "
              f"{', '.join(name_of[n] for n in cbd)}")
        return cbd

    def assign_work(self):
        """각 사람에게 직장을 배정.
          · WORK_CBD_PROB(65%)  → 맨해튼 도심(CBD) 노드(도심 내 인구 비례로 선택)
          · 나머지              → 집 근처(이웃 동네)에서 근무"""
        cbd = self.cbd_nodes
        w = np.array([self.nbhd[n].population for n in cbd], dtype=float)
        w = w / w.sum()
        n_cbd = 0
        for p in self.persons:
            if cbd and self.rng.random() < WORK_CBD_PROB:
                p.work_node = cbd[self.rng.choice(len(cbd), p=w)]
                n_cbd += 1
            else:
                nbrs = self.adj[p.home_node]
                p.work_node = nbrs[self.rng.integers(len(nbrs))] if nbrs else p.home_node
        print(f"  직장 배정: {n_cbd:,}명 "
              f"({n_cbd / len(self.persons) * 100:.0f}%)이 도심(CBD) 근무")

    def seed_infection(self):
        # 인구가 있는 동네 중 무작위로 초기 감염 동네 선택(재현 가능)
        populated = [n for n in self.res_ids if self.nbhd[n].population > 0]
        seeds = self.rng.choice(populated, size=INITIAL_INFECTED_NODES, replace=False)
        by_node = defaultdict(list)
        for p in self.persons:
            by_node[p.current_node].append(p)
        for node in seeds:
            ppl = by_node[node]
            k = min(INITIAL_INFECTED_PER_NODE, len(ppl))
            for p in self.rng.choice(ppl, size=k, replace=False):
                p.infect(at_step=0)
        names = [self.res.loc[self.res.nta2020 == n, "ntaname"].iloc[0] for n in seeds]
        print(f"  초기 감염 동네: {', '.join(names)}")

    # ----- 시간대(낮/밤) -----
    @property
    def phase(self):
        return "Day" if self.t % STEPS_PER_DAY == 0 else "Night"

    # ----- ① 출퇴근 이동 -----
    def commute(self):
        """낮/밤에 따라 사람을 직장/집으로 옮긴다.
          · 낮(Day)  : 건강자(S)·회복자(R)만 80% 확률로 출근.
                       감염자(I)는 즉시 '자가격리' → 출근하지 않고 집에만 머무름.
          · 밤(Night): 모두 집으로 귀가.
          · 사망자(D): 낮·밤 모두 이동하지 않음(비활성)."""
        if self.phase == "Day":
            for p in self.persons:
                if p.condition == "D":
                    continue                       # 사망자: 이동 없음
                if p.condition == "I":
                    p.current_node = p.home_node   # 감염자: 자가격리(집)
                else:                              # S, R 만 출근
                    p.current_node = (p.work_node
                                      if self.rng.random() < COMMUTE_PROB
                                      else p.home_node)
        else:
            for p in self.persons:
                if p.condition == "D":
                    continue                       # 사망자: 이동 없음
                p.current_node = p.home_node

    # ----- ② 감염: 같은 동네 안에서 밀도 의존 국소 전파 -----
    def spread_infection(self):
        """이동이 끝난 뒤, 동네별로 S/I/R 을 집계하고
        감염자 수 I 에 비례해 커지는 확률로 S 를 감염시킨다.

            P(감염) = 1 - exp(-BETA * I / N)
              I : 그 동네에 현재 있는 감염자 수
              N : 그 동네의 '살아있는' 현재 인원 (S+I+R) — 사망자 D 는 제외
        감염자가 많을수록(=I/N 이 클수록) 감염 확률이 1 에 가까워진다.
        사망자(D)는 접촉·집계 어디에도 포함되지 않는다(비활성)."""
        members = defaultdict(list)
        for p in self.persons:
            if p.condition == "D":
                continue                       # 사망자는 감염 계산에서 제외
            members[p.current_node].append(p)
        for node, ppl in members.items():
            N = len(ppl)                       # 살아있는 인원(S+I+R)만
            n_inf = sum(1 for q in ppl if q.condition == "I")
            if n_inf == 0 or N == 0:
                continue
            p_infect = 1.0 - np.exp(-BETA * n_inf / N)   # 밀도 의존 감염 확률
            for q in ppl:
                if q.condition == "S" and self.rng.random() < p_infect:
                    q.infect(at_step=self.t)

    # ----- ③ 감염 종료: 14일 경과 시 2% 사망(D) / 98% 회복(R) -----
    def resolve_infections(self):
        for p in self.persons:
            if p.condition == "I" and (self.t - p.infected_at) >= RECOVERY_STEPS:
                if self.rng.random() < FATALITY_RATE:
                    p.condition = "D"      # 사망(치사율 2%) — 이후 완전 비활성
                else:
                    p.condition = "R"      # 회복(98%)

    # ----- 동네별 집계(코로플레스용, 사망자 제외) -----
    def update_neighborhood_counts(self):
        for na in self.nbhd.values():
            na.num_total = 0
            na.num_infected = 0
        for p in self.persons:
            if p.condition == "D":
                continue                   # 사망자는 살아있는 인구 집계에서 제외
            na = self.nbhd[p.current_node]
            na.num_total += 1
            if p.condition == "I":
                na.num_infected += 1

    def get_ratios(self):
        """res_ids 순서대로 각 동네의 감염비율 배열."""
        return np.array([self.nbhd[n].ratio for n in self.res_ids])

    def get_sir_counts(self):
        c = {"S": 0, "I": 0, "R": 0, "D": 0}
        for p in self.persons:
            c[p.condition] += 1
        return c

    def record(self, label=""):
        c = self.get_sir_counts()
        for k in "SIRD":
            self.history[k].append(c[k])
        self.history["label"].append(label)

    # ----- 한 스텝(낮 또는 밤) -----
    def step(self):
        day = self.t // STEPS_PER_DAY + 1
        ph = "Day" if self.phase == "Day" else "Night"
        self.commute()              # ① 출퇴근 이동(감염자 자가격리, 사망자 정지)
        self.spread_infection()     # ② 모인 곳에서 밀도 의존 감염(N=생존자)
        self.resolve_infections()   # ③ 14일 경과 시 2% 사망 / 98% 회복
        self.update_neighborhood_counts()
        self.record(label=f"Day {day} · {ph}")
        self.t += 1


# ==================================================================
# 4. 시뮬레이션 실행 + 시각화(코로플레스 + SIR 그래프)
# ==================================================================
def run_simulation(model):
    """유행이 끝날 때까지(또는 MAX_STEPS) 매 스텝 진행하되,
    애니메이션 프레임은 FRAME_STRIDE 스텝마다만 저장한다(GIF 용량 절감).
    S/I/R/D 곡선용 history 는 매 스텝 기록되므로 정확도는 유지된다."""
    print("\n시뮬레이션 실행 중...")
    frames, frame_steps = [], []
    def snap():
        frames.append(model.get_ratios())
        frame_steps.append(len(model.history["S"]) - 1)   # 대응하는 history 인덱스
    snap()                                                # 초기 상태
    while model.t < MAX_STEPS:
        model.step()
        done = model.get_sir_counts()["I"] == 0
        if model.t % FRAME_STRIDE == 0 or done:
            snap()
        if done:
            break
    print(f"  {model.t}스텝({model.t//STEPS_PER_DAY}일)에서 종료, {len(frames)}프레임")
    return frames, frame_steps


def animate_simulation(model, filename="nyc_network_metapop_sir.gif"):
    frames, frame_steps = run_simulation(model)
    total = len(model.persons)
    H = model.history

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1])

    # --- (좌) 코로플레스 지도 ---
    # 배경: 비주거지 포함 전체 경계를 옅게
    model.res.plot(ax=ax_map, color="#f4f4f4", edgecolor="0.8", linewidth=0.3)
    norm = Normalize(vmin=0, vmax=1)
    # 감염비율로 색칠할 레이어(아티스트를 잡아 매 프레임 색만 갱신)
    choropleth = model.res.plot(ax=ax_map, column=frames[0], cmap=CMAP,
                                norm=norm, edgecolor="0.7", linewidth=0.3)
    art = ax_map.collections[-1]
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=ax_map,
                        fraction=0.035, pad=0.02)
    cbar.set_label("Neighborhood infection rate  I / (living S+I+R)")
    map_title = ax_map.set_title("")

    # 실시간 모니터: 누적 사망자 D + 생존 인구
    monitor = ax_map.text(
        0.02, 0.98, "", transform=ax_map.transAxes, va="top", ha="left",
        fontsize=12, family="AppleGothic",
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

    # --- (우) SIRD 누적 그래프 ---
    xdata = list(range(len(H["S"])))
    (lineS,) = ax_curve.plot([], [], color="#2f6fd0", lw=2.2, label="S Susceptible")
    (lineI,) = ax_curve.plot([], [], color="#d1352c", lw=2.2, label="I Infected")
    (lineR,) = ax_curve.plot([], [], color="#2e8b57", lw=2.2, label="R Recovered")
    (lineD,) = ax_curve.plot([], [], color="#000000", lw=2.2, label="D Deaths (cumulative)")
    ax_curve.set_xlim(0, len(H["S"]) - 1)
    ax_curve.set_ylim(0, total * 1.02)
    ax_curve.set_xlabel("Step (Day/Night cycle, 2 steps = 1 day)")
    ax_curve.set_ylabel("Number of person agents")
    ax_curve.set_title("Total S / I / R Trajectory")
    ax_curve.legend(loc="center right", fontsize=10)
    ax_curve.grid(True, alpha=0.25)

    def update(i):
        art.set_array(frames[i])                     # 코로플레스 색 갱신
        s = frame_steps[i]                           # 이 프레임의 history 인덱스
        c = {k: H[k][s] for k in "SIRD"}
        living = c["S"] + c["I"] + c["R"]            # 생존 인구
        map_title.set_text(
            f"NYC Commute SIRD   {H['label'][s]}   "
            f"S={c['S']:,}  I={c['I']:,}  R={c['R']:,}")
        monitor.set_text(
            f"Cumulative deaths D : {c['D']:,}\n"
            f"Living population    : {living:,}\n"
            f"CFR (cumulative)     : {c['D']/(c['R']+c['D'])*100 if (c['R']+c['D'])>0 else 0:.1f}%")
        lineS.set_data(xdata[:s + 1], H["S"][:s + 1])
        lineI.set_data(xdata[:s + 1], H["I"][:s + 1])
        lineR.set_data(xdata[:s + 1], H["R"][:s + 1])
        lineD.set_data(xdata[:s + 1], H["D"][:s + 1])
        return art, lineS, lineI, lineR, lineD, monitor, map_title

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=200)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, filename)
    anim.save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"저장: {out}")


# ==================================================================
# 5. 메인
# ==================================================================
if __name__ == "__main__":
    model = NYCNetworkSIRModel()
    animate_simulation(model)
    print("완료.")
