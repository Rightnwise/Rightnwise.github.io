# ================================================================
#  설치 (터미널에서 먼저 실행):
#
#    pip install mesa mesa-geo networkx numpy geopandas shapely matplotlib
#
# ================================================================
"""
nyc_metapop_sird_final.py
================================================================
뉴욕시 동(Neighborhood) 단위 '네트워크 기반 메타포퓰레이션 SIRD' 최종본.

핵심 아이디어
  · 외부 고용 데이터 없이, 지도 파일 정보만으로 출퇴근 확률분포(PDF)를 자동 생성.
  · '방법 B(중심지 거리 감쇠)': 인구밀도가 가장 높은 동네를 중심지로 자동 탐색하고,
    거기서 멀어질수록 감소하는 일자리 지수(job_index)를 만든 뒤, 중력모형으로
    각 동네의 목적지 확률분포(destination_pdf)를 빌드한다.
  · 낮/밤 확률적 출퇴근 + SIRD(사망 포함) + 감염자 자가격리.

이식성(다른 도시 지도 교체)
  · geojson 을 자동 탐색.  좌표계는 거리 계산을 위해 미터 투영으로 재투영.
  · 인구: (1) 인구 컬럼이 있으면 사용 → (2) borocode 가 있으면 실제 센서스
    자치구 인구를 면적 비례 분배 → (3) 둘 다 없으면 면적 비례(균일 밀도).
  · 'ntatype' 컬럼이 있으면 주거지(0)만 사용(공항/공원 제외), 없으면 전체 사용.
  → NYC 전용 하드코딩을 최소화해 다른 도시 지도로 바꿔 끼워도 동작한다.

실행:
  python3 nyc_metapop_sird_final.py
  → result/nyc_metapop_sird_final.gif
"""

import os
import glob
import numpy as np
import geopandas as gpd

import mesa
import mesa_geo as mg

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# ==================================================================
# [1] 글로벌 상수 (Configuration)
# ==================================================================
# ※ 상수는 nyc_network_metapop_sir.py 에서 튜닝했던 값으로 맞춤
POP_PER_AGENT = 1000           # 에이전트 1명 = 실제 약 1,000명 → 동네별 인구 비례 배치(총 ~8,800명)
INITIAL_INFECTED = 20          # 초기 감염자 수(네트워크판 3동네×8명)
INFECTION_BETA = 0.10          # 감염 계수(출퇴근 혼합에 맞춰 낮춘 값)
FATALITY_RATE = 0.02           # 치사율(2%)
INCUBATION_PERIOD = 8          # 잠복기 틱수(4일 = 낮·밤 2틱×4)
                               #   이 기간엔 무증상 → 정상 출퇴근하되 전염은 그대로 시킴
RECOVERY_PERIOD = 28           # 감염→회복/사망 판정까지 틱수(14일 = 낮·밤 2틱×14)
DISTANCE_DECAY_ALPHA = 2.0     # 출퇴근 거리 감쇠 계수(중력모형 지수)

# 시뮬레이션/시각화
MAX_TICKS = 600               # 최대 틱(낮·밤 1틱씩). 보통 그 전에 유행 종료
FRAME_STRIDE = 4              # 애니메이션 프레임 샘플 간격(GIF 용량↓)
RANDOM_SEED = 42
TARGET_CRS = "EPSG:32618"     # 미터 투영(거리·면적 계산용)
CMAP = "Reds"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

# (폴백용) 실제 2020 US 센서스 자치구 인구
BOROUGH_POP_2020 = {
    "1": 1_694_251, "2": 1_472_654, "3": 2_736_074,
    "4": 2_405_464, "5": 495_747,
}
STATE_COLOR = {"S": "#2f6fd0", "I": "#d1352c", "R": "#2e8b57", "D": "#000000"}


# ==================================================================
# 데이터 로딩 (이식성 있는 자동 처리)
# ==================================================================
def load_geojson(folder=HERE):
    matches = sorted(glob.glob(os.path.join(folder, "*.geojson")))
    if not matches:
        raise FileNotFoundError(f"{folder} 에 .geojson 이 없습니다.")
    print(f"지도 로드: {os.path.basename(matches[0])}")
    gdf = gpd.read_file(matches[0])
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(TARGET_CRS)

    # 주거지만 사용(가능하면). ntatype==0 = 주거지
    if "ntatype" in gdf.columns:
        gdf = gdf[gdf["ntatype"] == "0"]
    gdf = gdf.reset_index(drop=True)

    # 동네 이름 컬럼 자동 탐색(없으면 인덱스로)
    name_col = next((c for c in ["ntaname", "name", "NAME", "neighborhood"]
                     if c in gdf.columns), None)
    gdf["_name"] = gdf[name_col] if name_col else [f"node_{i}" for i in range(len(gdf))]
    return gdf


def assign_population(gdf):
    """인구 3단 폴백: (1)인구컬럼 (2)센서스×면적 (3)면적비례."""
    gdf = gdf.copy()
    gdf["_area"] = gdf.geometry.area.astype(float)          # m^2

    pop_col = next((c for c in ["population", "pop", "Pop", "totpop", "pop2020"]
                    if c in gdf.columns), None)
    if pop_col:
        gdf["_pop"] = gdf[pop_col].astype(float).clip(lower=1.0)
        print(f"  인구: 컬럼 '{pop_col}' 사용")
    elif "borocode" in gdf.columns:
        gdf["_pop"] = 0.0
        for code, pop in BOROUGH_POP_2020.items():
            m = gdf["borocode"] == code
            if m.any():
                a = gdf.loc[m, "_area"]
                gdf.loc[m, "_pop"] = pop * a / a.sum()
        gdf["_pop"] = gdf["_pop"].clip(lower=1.0)
        print("  인구: 센서스 자치구 인구를 면적 비례로 분배")
    else:
        gdf["_pop"] = gdf["_area"] / gdf["_area"].mean() * 1000.0
        print("  [경고] 인구 정보 없음 → 면적 비례(균일 밀도) 가정")
    return gdf


# ==================================================================
# [3] 에이전트
# ==================================================================
class NeighborhoodAgent(mg.GeoAgent):
    """고정된 공간 노드(동네). 인구·면적·job_index·destination_pdf 보유."""

    def __init__(self, model, geometry, crs, idx, name, population, area):
        super().__init__(model, geometry, crs)
        self.idx = idx
        self.name = name
        self.population = population
        self.area = area
        self.job_index = 0.0
        self.destination_pdf = None        # 이 동네 사람들의 출근지 확률분포


class PersonAgent(mesa.Agent):
    """사람. condition('S'/'I'/'R'/'D'), home(고정 동네 idx), current(현재 idx)."""

    def __init__(self, model, home_idx):
        super().__init__(model)
        self.condition = "S"
        self.home = home_idx
        self.current = home_idx
        self.infected_at = None


# ==================================================================
# 모델
# ==================================================================
class MetapopSIRDModel(mesa.Model):
    def __init__(self, seed=RANDOM_SEED):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.tick = 0

        # --- 지도 · 인구 ---
        gdf = assign_population(load_geojson())
        self.gdf = gdf
        self.N = len(gdf)
        self.names = list(gdf["_name"])
        self.pop = gdf["_pop"].to_numpy(float)
        self.area = gdf["_area"].to_numpy(float)
        self.bounds = gdf.total_bounds

        # 노드(GeoSpace + NeighborhoodAgent)
        self.space = mg.GeoSpace(crs=str(gdf.crs), warn_crs_conversion=False)
        self.nodes = []
        for i, row in gdf.iterrows():
            na = NeighborhoodAgent(self, row.geometry, gdf.crs, i,
                                   row["_name"], self.pop[i], self.area[i])
            self.nodes.append(na)
        self.space.add_agents(self.nodes)

        # --- [2] 방법 B: 중심지 탐색 + job_index + 출퇴근 PDF 자동 생성 ---
        self.build_method_b()

        # --- 사람 에이전트 (인구 비례로 집 배정) ---
        self.persons = self.create_agents()
        self.home_groups = [[] for _ in range(self.N)]
        for p in self.persons:
            self.home_groups[p.home].append(p)

        # --- 초기 감염 ---
        self.seed_infection()

        # --- 기록 ---
        self.history = {"S": [], "I": [], "R": [], "D": [], "label": []}
        self.record("Initial (Home)")

    # ---------------------------------------------------------------
    # [2] 방법 B: 중심지 자동탐색 → job_index → destination_pdf
    # ---------------------------------------------------------------
    def build_method_b(self):
        cent = self.gdf.geometry.centroid
        xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])  # m

        # 노드 간 거리행렬(km)
        diff = xy[:, None, :] - xy[None, :, :]
        Dm = np.sqrt((diff ** 2).sum(-1)) / 1000.0
        # 자기 자신 거리(대각선)=동네 유효반경(km): 0 나눗셈 방지 + 지역 내 근무 허용
        radius_km = np.sqrt(self.area / np.pi) / 1000.0
        Dm[np.arange(self.N), np.arange(self.N)] = radius_km
        self.centroids = xy

        # ① 중심지 자동탐색: 인구밀도(pop/area) 최대 동네
        density = self.pop / self.area
        top = np.where(density >= density.max() * 0.999)[0]   # 최고밀도(동률) 집합
        # 동률이면 그 집합의 기하 중심에 가장 가까운 노드를 중심지로(도심 대표점)
        c = xy[top].mean(axis=0)
        self.center = int(top[np.argmin(((xy[top] - c) ** 2).sum(1))])
        print(f"  [방법B] 중심지 자동탐색 → '{self.names[self.center]}' "
              f"(밀도 {density[self.center]*1e6:.0f} 명/km², 후보 {len(top)}개)")

        # ② 일자리 지수: 중심지와 가까울수록 1.0, 멀수록 0 (거리 km 기준)
        d_center = np.sqrt(((xy - xy[self.center]) ** 2).sum(1)) / 1000.0
        self.job_index = 1.0 / (1.0 + d_center)

        # ③ 출퇴근 PDF(중력모형): Weight = pop_j * job_j / dist_ij^alpha, 행별 정규화
        W = (self.pop[None, :] * self.job_index[None, :]) / (Dm ** DISTANCE_DECAY_ALPHA)
        self.pdf = W / W.sum(axis=1, keepdims=True)

        # 노드 객체에도 저장(요구 사항)
        for i, na in enumerate(self.nodes):
            na.job_index = float(self.job_index[i])
            na.destination_pdf = self.pdf[i]

        # 참고 출력: 중심지로 출근이 얼마나 몰리는지
        share_to_center = float((self.pdf[:, self.center] * self.pop).sum()
                                / self.pop.sum())
        print(f"  [방법B] 평균 {share_to_center*100:.0f}% 가 중심지로 출근하는 무게")

    def create_agents(self):
        # 각 동네에 '실제 인구 / POP_PER_AGENT' 명씩 배치 (실제 인구 비례)
        persons = []
        for idx in range(self.N):
            n = max(1, round(self.pop[idx] / POP_PER_AGENT)) if self.pop[idx] > 0 else 0
            persons.extend(PersonAgent(self, idx) for _ in range(n))
        print(f"  사람 에이전트 {len(persons):,}명 "
              f"(1명 = {POP_PER_AGENT:,}명 스케일, 실제 인구 비례 배치)")
        return persons

    def seed_infection(self):
        for p in self.rng.choice(self.persons, size=INITIAL_INFECTED, replace=False):
            p.condition = "I"
            p.infected_at = 0
        print(f"  초기 감염 {INITIAL_INFECTED}명")

    # ---------------------------------------------------------------
    # [4] 스텝 규칙
    # ---------------------------------------------------------------
    @property
    def phase(self):
        return "Day" if self.tick % 2 == 0 else "Night"

    def is_incubating(self, a):
        """감염됐지만 아직 증상 발현 전(잠복기): 무증상 → 정상 출퇴근하되 전염은 시킴."""
        return (a.condition == "I"
                and (self.tick - a.infected_at) < INCUBATION_PERIOD)

    def commute(self):
        """낮: S·R + '잠복기(무증상) 감염자'는 destination_pdf 로 확률적 출근.
              증상 발현한 감염자만 자가격리(집), 사망자(D)는 정지.
           밤: 사망자 제외 전원 귀가."""
        if self.phase == "Day":
            for hidx, group in enumerate(self.home_groups):
                # 출근자 = 건강자·회복자 + 잠복기(무증상) 감염자
                movers = [a for a in group
                          if a.condition in ("S", "R") or self.is_incubating(a)]
                if movers:
                    dests = self.rng.choice(self.N, size=len(movers),
                                            p=self.pdf[hidx])
                    for a, d in zip(movers, dests):
                        a.current = int(d)
                for a in group:
                    # 증상 발현한 감염자(잠복기 종료)만 자가격리
                    if a.condition == "I" and not self.is_incubating(a):
                        a.current = a.home
                    # D: 이동 없음(집에 정지)
        else:
            for a in self.persons:
                if a.condition != "D":
                    a.current = a.home

    def spread_infection(self):
        """현재 모인 곳 기준 밀도 의존 감염.
           P = 1 - exp(-BETA * I / N), N = 사망자 제외 살아있는 현재 인원."""
        present = [[] for _ in range(self.N)]
        for a in self.persons:
            if a.condition != "D":
                present[a.current].append(a)
        for ppl in present:
            n = len(ppl)
            if n == 0:
                continue
            n_inf = sum(1 for q in ppl if q.condition == "I")
            if n_inf == 0:
                continue
            p_inf = 1.0 - np.exp(-INFECTION_BETA * n_inf / n)
            for q in ppl:
                if q.condition == "S" and self.rng.random() < p_inf:
                    q.condition = "I"
                    q.infected_at = self.tick

    def resolve(self):
        """감염 RECOVERY_PERIOD틱 경과 → 2% 사망(D) / 98% 회복(R)."""
        for p in self.persons:
            if p.condition == "I" and (self.tick - p.infected_at) >= RECOVERY_PERIOD:
                p.condition = "D" if self.rng.random() < FATALITY_RATE else "R"

    # ---------------------------------------------------------------
    # 집계 · 스텝
    # ---------------------------------------------------------------
    def node_infection_rates(self):
        """각 동네의 실시간 감염률 = I / 살아있는 현재 인원."""
        Icnt = np.zeros(self.N)
        tot = np.zeros(self.N)
        for a in self.persons:
            if a.condition == "D":
                continue
            tot[a.current] += 1
            if a.condition == "I":
                Icnt[a.current] += 1
        return np.divide(Icnt, tot, out=np.zeros(self.N), where=tot > 0)

    def counts(self):
        c = {"S": 0, "I": 0, "R": 0, "D": 0}
        for p in self.persons:
            c[p.condition] += 1
        return c

    def record(self, label):
        c = self.counts()
        for k in "SIRD":
            self.history[k].append(c[k])
        self.history["label"].append(label)

    def step(self):
        day = self.tick // 2 + 1
        ph = "Day" if self.phase == "Day" else "Night"
        self.commute()            # 이동(낮=확률출근/밤=귀가)
        self.spread_infection()   # 감염(낮/밤 각각)
        self.resolve()            # 회복/사망 판정
        self.record(f"Day {day} · {ph}")
        self.tick += 1


# ==================================================================
# [5] 시뮬레이션 실행 + 시각화
# ==================================================================
def run(model):
    print("\n시뮬레이션 실행 중...")
    frames, frame_ticks = [], []

    def snap():
        frames.append(model.node_infection_rates())
        frame_ticks.append(len(model.history["S"]) - 1)

    snap()
    while model.tick < MAX_TICKS:
        model.step()
        done = model.counts()["I"] == 0
        if model.tick % FRAME_STRIDE == 0 or done:
            snap()
        if done:
            break
    print(f"  {model.tick}틱({model.tick//2}일)에서 종료, {len(frames)}프레임")
    return frames, frame_ticks


def animate(model, filename="nyc_metapop_sird_final.gif"):
    frames, frame_ticks = run(model)
    total = len(model.persons)
    H = model.history

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax_map, ax_curve = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # (좌) 코로플레스: 동네별 실시간 감염률
    model.gdf.plot(ax=ax_map, color="#f4f4f4", edgecolor="0.8", linewidth=0.3)
    norm = Normalize(0, 1)
    model.gdf.plot(ax=ax_map, column=frames[0], cmap=CMAP, norm=norm,
                   edgecolor="0.7", linewidth=0.3)
    art = ax_map.collections[-1]
    # 자동탐색된 중심지 표시(별)
    cx, cy = model.centroids[model.center]
    ax_map.scatter([cx], [cy], marker="*", s=320, c="gold",
                   edgecolors="black", linewidths=0.8, zorder=5)
    ax_map.annotate(f"Center: {model.names[model.center]}", (cx, cy),
                    textcoords="offset points", xytext=(8, 8), fontsize=10)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), ax=ax_map,
                 fraction=0.035, pad=0.02).set_label("Neighborhood live infection rate  I / (living N)")
    map_title = ax_map.set_title("")
    monitor = ax_map.text(0.02, 0.98, "", transform=ax_map.transAxes, va="top",
                          fontsize=12, family="AppleGothic",
                          bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

    # (우) SIRD 곡선
    x = list(range(len(H["S"])))
    lines = {k: ax_curve.plot([], [], color=STATE_COLOR[k], lw=2.2, label=lab)[0]
             for k, lab in [("S", "S Susceptible"), ("I", "I Infected"),
                            ("R", "R Recovered"), ("D", "D Deaths (cumulative)")]}
    ax_curve.set_xlim(0, len(H["S"]) - 1)
    ax_curve.set_ylim(0, total * 1.02)
    ax_curve.set_xlabel("Tick (Day/Night cycle, 2 ticks = 1 day)")
    ax_curve.set_ylabel("Number of person agents")
    ax_curve.set_title("Total SIRD Trajectory")
    ax_curve.legend(loc="center right", fontsize=10)
    ax_curve.grid(True, alpha=0.25)

    def update(i):
        art.set_array(frames[i])
        s = frame_ticks[i]
        c = {k: H[k][s] for k in "SIRD"}
        living = c["S"] + c["I"] + c["R"]
        map_title.set_text(f"NYC Metapopulation SIRD (Method B Commute)   {H['label'][s]}   "
                           f"S={c['S']:,} I={c['I']:,} R={c['R']:,}")
        cfr = c["D"] / (c["R"] + c["D"]) * 100 if (c["R"] + c["D"]) else 0
        monitor.set_text(f"Cumulative deaths D : {c['D']:,}\n"
                         f"Living population   : {living:,}\n"
                         f"Cumulative CFR      : {cfr:.1f}%")
        for k in "SIRD":
            lines[k].set_data(x[:s + 1], H[k][:s + 1])
        return (art, monitor, map_title, *lines.values())

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=200)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, filename)
    anim.save(out, writer=PillowWriter(fps=5))
    plt.close(fig)
    print(f"저장: {out}")


# ==================================================================
# 메인
# ==================================================================
if __name__ == "__main__":
    model = MetapopSIRDModel()
    animate(model)
    print("완료.")
